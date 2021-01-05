import os.path
from os import path
from time import sleep
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


def calc_speed_bbox_tlwh(bbox):
    return np.array([bbox[0][0], bbox[0][1], (bbox[1][0] - bbox[0][0]), (bbox[1][1] - bbox[0][1])]).reshape(1, 4)


class object_tracker:

    def calc_speed(self, i_total_frames):
        l_total_time = i_total_frames / (self.ref_fps * 1.0)
        rtn_speed = (self.ref_distance * 1.0 / l_total_time) * 3600 / 1600
        return round(rtn_speed)

    def __init__(self):
        # Initiate deep sort parameter
        self.max_distance = 0.5
        self.distance_measure_type = 'cosine'
        self.nn_budget = None
        self.nms_max_overlap = 0.78
        self.model_filename = 'model_data/mars-small128.pb'
        # Initiate YOLO parameter
        self.yolo_label_loc = './data/labels/coco.names'
        self.yolo_weight_loc = './weights/yolov3.tf'
        self.yolo_class_nbr = 80
        self.crop_x_from = 170
        self.crop_x_to = 1480
        self.crop_y_from = 620
        self.crop_y_to = 2040
        self.speed_detect_bbox_top = ((950, 780), (1350, 860))
        self.speed_detect_bbox_bottom = ((150, 1280), (1185, 1390))
        self.max_frames_missing = 8
        self.skip_frame = 2
        self.ref_distance = 53
        self.ref_fps = 60
        self.detection_classes_str = 'truck,train,bus,car'
        self.yolo_predict_time = 0
        self.deep_sort_predict_time = 0
        self.cv_draw_time = 0
        self.feature_engineering_time = 0
        self.embedding_time = 0
        self.file_io_time = 0
        self.code_segment_start = time.time()
        self.detection_classes = self.detection_classes_str.split(',')
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric(self.distance_measure_type, self.max_distance,
                                                                self.nn_budget)
        self.tracker = Tracker(self.metric)
        self.physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(self.physical_devices) > 0:
            tf.config.experimental.set_memory_growth(self.physical_devices[0], True)

        self.speed_detect_bbox_top_tlwh = self.calc_speed_bbox_tlwh(self.speed_detect_bbox_top)
        self.speed_detect_bbox_bottom_tlwh = self.calc_speed_bbox_tlwh(self.speed_detect_bbox_bottom)
        self.video_loc = './data/video/h1.mp4'
        self.vid = cv2.VideoCapture(self.video_loc)
        self.fl_base = '/home/lilun/dockers/data/out/labels/h1_'
        self.fl_ctr = 1
        self.fl_wait_ctr = 0
        self.cur_frame = 0
        self.count = 0

    def object_track_test(self, img, converted_boxes, scores, names):
        self.cur_frame = self.cur_frame + 1
        print('cur_frame is '+str(self.cur_frame))
        cv2.imshow('output', img)

    def object_track(self, img, converted_boxes, scores, names):

        self.cur_frame = self.cur_frame + 1
        if not ((self.cur_frame < 60) or (self.cur_frame % self.skip_frame == 0)):
            features = self.encoder(img, converted_boxes)
            zip_val = zip(converted_boxes, scores, names, features)
            object_detected = [(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip_val if
                               class_name in self.detection_classes]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          object_detected]
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
            # run non-maxima suppresion
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)
            code_segment_start = time.time()
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > self.max_frames_missing:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                if track.bbox_cross_at_frame != -1:
                    if track.class_name == 'truck' and track.track_id == 6:
                        print("Truck detected at " + str(self.cur_frame))
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                                  (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])),
                                  color,
                                  -1)
                track_bbox = track.to_tlwh()
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
                                  (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])),
                                  color,
                                  -1)
                    cv2.putText(img, tag_str, (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

            cv2.rectangle(img, self.speed_detect_bbox_top[0], self.speed_detect_bbox_top[1], (200.0, 0, 0), 2)
            cv2.rectangle(img, self.speed_detect_bbox_bottom[0], self.speed_detect_bbox_bottom[1], (200.0, 0, 0), 2)
            cv2.imshow('output', img)
