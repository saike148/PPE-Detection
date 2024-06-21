import numpy as np
import random
import os
import cv2
import time
import ultralytics
from PIL import Image
from contextlib import contextmanager
from ultralytics import YOLO

ultralytics.checks()
from ultralytics import YOLO

class YOLOv8_ObjectDetector:
    def __init__(self, model_file='yolov8n.pt', labels=None, classes=None, conf=0.25, iou=0.45):
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.model = YOLO(model_file)
        self.model_name = model_file.split('.')[0]
        self.results = None

        if labels is None:
            self.labels = self.model.names
        else:
            self.labels = labels
    
    def predict_img(self, img, verbose=True):
        results = self.model(img, classes=self.classes, conf=self.conf, iou=self.iou, verbose=False)
        self.orig_img = img
        self.results = results[0]
        return results[0]
 
    def default_display(self, show_conf=True, line_width=None, font_size=None, font='Arial.ttf', pil=False, example_frame=None):
        if self.results is None:
            raise ValueError('No detected objects to display. Call predict_img() method first.')

        # Convert the provided example_frame (video frame) to a NumPy array
        img_array = np.asarray(example_frame)

        # Display the image using the plot method
        display_img = self.results.plot(show_conf, line_width, font_size, font, pil, img_array)
        return display_img