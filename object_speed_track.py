#!/usr/bin/env python
import os
import pika
import sys
import numpy as np
import cv2
from tools import generate_detections as gdet
from deep_sort import iou_matching
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import matplotlib.pyplot as plt
import time


class object_tracker:
    def __init__(self, video_loc):
        """
        Note: All these parameter initialization needs to be driven by configuration files
        Part 1 initialization:
        a. model_filename: Define the location of embedding model which gives vector representation of images
        b. nms_max_overlap: Bounding box overlap threshold.  In Keras, seems need to set it lower value
        c. nn_budget: how many historical images for a given object needs to be stored.  None means no limit as
            long as objects on the screen
        d. metric: use cosine distance to calculate distances from embedding model
        """
        self.model_filename = 'model_data/mars-small128.pb'
        self.nms_max_overlap = 1.0
        self.max_cosine_distance = 0.5
        self.nn_budget = None
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        """
        Part 2: video initialization
        a. skip_frame: skip frames, for example if video is 60 frames per second, we may only need to process
            odd frames to enhance the performance.
            Note: make sure at least process 25 frames per second.
        b. ref_fps Video fps 
        """
        self.ref_fps = 60
        self.vid = cv2.VideoCapture(video_loc)
        self.cur_frame = 0
        self.skip_frame = 2
        self.tracker = Tracker(self.metric)
        """
        Part 3: Detection Box setup
        Hard code for detection zone setup
        a. crop_*_*: Since detection region is between two bars.  No need to track any objects outside that region to 
           speed up object tracking performance.  (Note: no impact to Yolo object detection performance)
        b. *bbox*:  entrance and exit of speed detection zone
        """
        self.crop_x_from = 170
        self.crop_x_to = 1480
        self.crop_y_from = 620
        self.crop_y_to = 2040
        self.crop_bbox = ((self.crop_x_from, self.crop_y_from), (self.crop_x_to, self.crop_y_to))
        self.crop_bbox_tlwh = self.calc_speed_bbox_tlwh(self.crop_bbox)
        self.speed_detect_bbox_top = ((950, 780), (1350, 860))
        self.speed_detect_bbox_top_tlwh = np.array([950, 780, 400, 80]).reshape(1, 4)
        self.speed_detect_bbox_bottom = ((150, 1280), (1185, 1390))
        self.speed_detect_bbox_bottom_tlwh = np.array([150, 1280, 1035, 110]).reshape(1, 4)
        """
        Part 4: Others
        a. ref_distance: the distance of speed detection zone
        b. max_frames_missing: if Yolo fails to detect an object, we can use Kalman Filter to approximate the location
           of the object.  But this can only be done if Yolo continuously fails detect objects for max_frames_missing
           frames 
        """
        self.ref_distance = 53
        self.start_time = None
        self.max_frames_missing = 8

    """
        Speed estimation calculation
    """
    def calc_speed(self, i_fps, i_total_frames, i_ref_distance):
        l_total_time = i_total_frames / (i_fps * 1.0)
        rtn_speed = (i_ref_distance * 1.0 / l_total_time) * 3600 / 1600
        return round(rtn_speed)

    def calc_speed_bbox_tlwh(self, bbox):
        return np.array([bbox[0][0], bbox[0][1], (bbox[1][0] - bbox[0][0]), (bbox[1][1] - bbox[0][1])]).reshape(1, 4)

    """
    Step1: Read BBOx from Yolo
    Step2: Get image embedding from embedding model
    Step3: Track objects within speed detection zone using:
           Image embedding (Track objects by measuring how similar they look)
           Kalman Filter (Track objects by estimate object trajectory of next time in the time series)
    Step4: Estimate Vehicle speed
    """
    def video_read(self, frame_pos, detections):

        self.cur_frame = self.cur_frame + 1
        if self.cur_frame == 1:
            self.start_time = time.time()

        _, img = self.vid.read()
        if self.cur_frame % self.skip_frame == 0:
            _, img = self.vid.read()
            self.cur_frame = self.cur_frame + 1

        detections = [[[int(float(j)) for j in i[2:]], i[0], i[1]] for i in detections]
        bboxes = []
        names = []
        scores = []
        """
            Get all the bounding boxes within the detection zone.
            These boxes only need to be tracked to speed up tracking performance.
            Not all BBox needs to be tracked.
        """
        for item in detections:
            cur_bbox_tlwh = [int(float(eachRec)) for eachRec in item[0]]
            if iou_matching.iou(np.array(cur_bbox_tlwh), self.crop_bbox_tlwh) == 0:
                continue
            bboxes.append(cur_bbox_tlwh)
            names.append(item[1])
            scores.append(item[2])
        # Use embedding model to get vector representation of images within each of bounding boxes
        features = self.encoder(img, bboxes)
        names = np.array(names)
        scores = np.array([float(eachRec) for eachRec in scores])
        zip_val = zip(bboxes, scores, names, features)
        # For now, lets detect car and truck
        object_detected = [(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip_val if
                           class_name in ('car', 'truck')]
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      object_detected]
        # initialize color map

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        print("*** Total time since start @ current frame " + str(self.cur_frame) + " : " +
              str(time.time() - self.start_time))

        for track in self.tracker.tracks:
            """
            if Yolo fails detect object, try to use the Kalman Filter to track for at most
            max_frames_missing frames, if beyond, then give up
            """
            if not track.is_confirmed() or track.time_since_update > self.max_frames_missing:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            if track.bbox_cross_at_frame != -1:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                              -1)
            track_bbox = track.to_tlwh()
            """
            Detect if an object enters the speed detection zone
            Calculate number of frames the vehicle remains in the detection zone
            then use the number of frames and reference distance to derive speed 
            """
            iou_bbox_top_score = iou_matching.iou(track_bbox, self.speed_detect_bbox_top_tlwh)
            iou_bbox_bottom_score = iou_matching.iou(track_bbox, self.speed_detect_bbox_bottom_tlwh)
            if iou_bbox_top_score > 0 and iou_bbox_bottom_score <= 0:
                if track.bbox_cross_at_frame == -1:
                    track.bbox_cross_at_frame = self.cur_frame
                tag_str = class_name + "-" + str(track.track_id) + " (entering speed detect zone) "
            elif iou_bbox_bottom_score > 0:
                if track.estimated_speed == -1:
                    track.estimated_speed = self.calc_speed(self.ref_fps,
                                                            (self.cur_frame - track.bbox_cross_at_frame),
                                                            self.ref_distance)
                tag_str = class_name + "-" + str(track.track_id) \
                          + " Speed: " + str(track.estimated_speed) \
                          + " mph (exiting speed detect zone)"
            else:
                tag_str = class_name + "-" + str(track.track_id)
            if track.bbox_cross_at_frame != -1:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                              -1)
                cv2.putText(img, tag_str, (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

        cv2.rectangle(img, self.speed_detect_bbox_top[0], self.speed_detect_bbox_top[1], (200.0, 0, 0), 2)
        cv2.rectangle(img, self.speed_detect_bbox_bottom[0], self.speed_detect_bbox_bottom[1], (200.0, 0, 0), 2)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            exit()


def main():
    video_loc = '/home/lilun/PycharmProjects/object_detect_track/data/video/h1.mp4'
    """
    Step0:  Initialization of object tracker
    Step1:  Connect to RabbitMQ and listen to 'hwy_traffic_1' queue
            Requires the queue to maintain the correct order
    Step2:  When a message is delivered, callback method is triggered.
            In this case, Pytorch Yolo detects bounding box of each frames and send it through RabbitMQ
            Then it will be picked up my object tracking code
    Note:   Video stream is read by object detection and tracking at the same time
    
    This is just a proof of concept code.  A lot of thought needs to put on fault tolerance and defensive
    programming to make this robust.  Right now, just assume happy path.  
    """
    hwy_obj_tacker = object_tracker(video_loc)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hwy_traffic_1')

    def callback(ch, method, properties, body):
        # Get bounding box and pass to Video_read
        message_str = body.decode("utf-8")
        cur_frame = message_str.split('|')[0]
        detections = [eachRec.split(',') for eachRec in message_str.split('|')[1:]]
        hwy_obj_tacker.video_read(int(cur_frame), detections)

    channel.basic_consume(queue='hwy_traffic_1', on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
