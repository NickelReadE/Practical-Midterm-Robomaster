import torch
import os
#import serial
import numpy as np
from turtle import color
import matplotlib
import statistics
import cv2
import pyrealsense2.pyrealsense2 as rs
import cv2
import time
import argparse
import struct

matplotlib.use('TKAgg')
# Disable tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




# ColorDetection Classdepth_frame = frames.get_depth_frame()

class ColorDetection:
    def __init__(self):
        self.red_lower = np.array([0, 4, 226])
        self.red_upper = np.array([60, 255, 255])
        self.blue_lower = np.array([68, 38, 131])
        self.blue_upper = np.array([113, 255, 255])

    # Checks if more red or blue then decides if plate is red or blue
    def red_or_blue(self, color_frame):
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, self.red_lower, self.red_upper)
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(red_contours) > 0 and len(blue_contours) > 0:
            r_area = 0
            b_area = 0
            for c in red_contours:
                r_area += cv2.contourArea(c)
            for c in blue_contours:
                b_area += cv2.contourArea(c)
            if r_area > b_area:
                return 'r'
            else:
                return 'b'
        elif len(red_contours) > 0:
            return 'r'

        return 'b'




# Initialize CV Camera




class Capture:
    # Constructor with depth camera
    def __init__(self, dc=None, camera_index=0, is_realsense=True):
        # Check if realsense class depth camera object is passed or an integer for the index of a regular camera
        self.cap = cv2.VideoCapture(
            "./CV_Detection-code-refactor-arjun/utils/vid.mp4")
        self.load_model()
        self.robot_list = []
        self.color_detected = ColorDetection()

    def load_model(self):
        # or yolov5m, yolov5l, yolov5x, custom
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='./Algorithm/pt_files/best.pt')

    # Get Depth and Color Frame
    def capture_pipeline(self, debug=False, display=False):
        while True:
            # Get frame from camera
            try:
                ret, color_image = self.cap.read()
            except:
                print("Error getting frame iojhoiuh0i7")

            if ret:
                key = cv2.waitKey(1)
                if key == 27:
                    break

                # Frame is valid
                self.process_frame(color_image=color_image,
                                   debug=debug, display=display)

    # Process a color frame
    def process_frame(self, color_image, debug=False, display=False):
        conf_thres = 0.25  # Confidence threshold
        # Get bounding boxes
        results = self.model(color_image)

        # Post process bounding boxes
        rows = results.pandas().xyxy[0].to_numpy()

        detections_rows = results.pandas().xyxy

        #rows = [x.to_numpy for x in detections_rows]

#         for i in range(len(detections_rows)):
#             rows = detections_rows[i].to_numpy()

        # Go through all detections

        for i in range(len(rows)):
            if len(rows) > 0:
                # Get the bounding box of the first object (most confident)
                x_min, y_min, x_max, y_max, conf, cls, label = rows[i]
                # Coordinate system is as follows:
                # 0,0 is the top left corner of the image
                # x is the horizontal axis
                # y is the vertical axis
                # x_max, y_max is the bottom right corner of the screen

                # (0,0) --------> (x_max, 0)
                # |               |
                # |               |
                # |               |
                # |               |
                # |               |
                # (0, y_max) ----> (x_max, y_max)
                if debug:
                    print("({},{}) \n\n\n                     ({},{})".format(
                        x_min, y_min, x_max, y_max))
                    # os.system('cls')
                    # os.system('clear')

                if display and self.color_detected.red_or_blue(color_image) == "b":
                    bbox = [x_min, y_min, x_max, y_max]
                    self.write_bbx_frame(color_image, bbox, label, conf)
        # Display the image
        #cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

    def write_bbx_frame(self, color_image, bbxs, label, conf):
        # Display the bounding box
        x_min, y_min, x_max, y_max = bbxs
        cv2.rectangle(color_image, (int(x_min), int(y_min)), (int(
            x_max), int(y_max)), (0, 255, 0), 2)  # Draw with green color

        # Display the label with the confidence
        label_conf = label + " " + str(conf)
        cv2.putText(color_image, label_conf, (int(x_min), int(
            y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('RealSense', color_image)




capture_stream = Capture()


capture_stream.capture_pipeline(debug=True, display=True)
