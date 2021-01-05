# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time, random
import numpy as np
from absl import app, flags, logging
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes
from deep_sort import iou_matching
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image


def calc_speed(i_fps, i_total_frames, i_ref_distance):
    l_total_time = i_total_frames / (i_fps * 1.0)
    rtn_speed = (i_ref_distance * 1.0 / l_total_time) * 3600 / 1600
    return round(rtn_speed)


if __name__ == '__main__':
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    speed_detect_bbox_top = ((950, 780), (1350, 860))
    speed_detect_bbox_top_tlwh = np.array([950, 780, 400, 80]).reshape(1, 4)
    speed_detect_bbox_bottom = ((150, 1280), (1185, 1390))
    speed_detect_bbox_bottom_tlwh = np.array([150, 1280, 1035, 110]).reshape(1, 4)
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3(classes=80)

    class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
    logging.info('classes loaded')

    yolo.load_weights('./weights/yolov3.tf')
    logging.info('weights loaded')
    vid = cv2.VideoCapture('./data/video/h1.mp4')
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('myOut.avi', codec, fps, (width, height))
    list_file = open('detection.txt', 'w')
    frame_index = -1
    fps = 0.0
    count = 0
    cur_frame = 0
    skip_frame = 3
    ref_speed = 70
    ref_distance = 53
    ref_fps = 60
    while True:
        _, img = vid.read()
        cur_frame = cur_frame + 1

        if (cur_frame < 100) or (cur_frame % skip_frame == 0):
            continue

        print("The current frame is " + str(cur_frame))
        # if cur_frame == 200 or cur_frame == 259 or cur_frame == 260:
        #    print("Time since update: -> " + str(tracker.tracks[0].time_since_update))
        #    print("The debugging point")
        # try:
        #    print("Time since update: -> "+str(tracker.tracks[0].time_since_update))
        # except:
        #    print("No lucas yet")
        # if 1 in tracker.metric.samples:
        #    print("Tracker has lucas")
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else:
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        # img_in = transform_images(img_in, FLAGS.size)
        img_in = transform_images(img_in, 416)
        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        zip_val = zip(converted_boxes, scores[0], names, features)
        object_detected = [(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip_val if
                           class_name in ('car', 'truck')]
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      object_detected]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            track_bbox = track.to_tlwh()
            iou_bbox_top_score = iou_matching.iou(track_bbox, speed_detect_bbox_top_tlwh)
            iou_bbox_bottom_score = iou_matching.iou(track_bbox, speed_detect_bbox_bottom_tlwh)
            if iou_bbox_top_score > 0:
                if track.bbox_cross_at_frame == -1:
                    track.bbox_cross_at_frame = cur_frame
                tag_str = class_name + "-" + str(track.track_id) + " (entering speed detect zone) "
            elif iou_bbox_bottom_score > 0:
                if track.estimated_speed == -1:
                    # track.estimated_speed = round(ref_speed * 100 / (cur_frame - track.bbox_cross_at_frame))
                    track.estimated_speed = calc_speed(ref_fps, (cur_frame - track.bbox_cross_at_frame), ref_distance)
                tag_str = class_name + "-" + str(track.track_id) + " Speed: " + str(track.estimated_speed) + " mph (exiting speed detect zone)"
            else:
                tag_str = class_name + "-" + str(track.track_id)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(img, tag_str, (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        # for det in detections:
        #    bbox = det.to_tlbr()
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

        # print fps on screen
        cv2.rectangle(img, speed_detect_bbox_top[0], speed_detect_bbox_top[1], (200.0, 0, 0), 10)
        cv2.rectangle(img, speed_detect_bbox_bottom[0], speed_detect_bbox_bottom[1], (200.0, 0, 0), 10)
        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)
        out.write(img)
        frame_index = frame_index + 1
        list_file.write(str(frame_index) + ' ')
        if len(converted_boxes) != 0:
            for i in range(0, len(converted_boxes)):
                list_file.write(str(converted_boxes[i][0]) + ' ' + str(converted_boxes[i][1]) + ' ' + str(
                    converted_boxes[i][2]) + ' ' + str(converted_boxes[i][3]) + ' ')
        list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    out.release()
    list_file.close()
    cv2.destroyAllWindows()
