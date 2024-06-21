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
    def __init__(self, model_file='yolov8s.pt', labels=None, classes=None, conf=0.25, iou=0.45):
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
        
    def predict_video(self, video_path, save_dir, save_format="avi", display='custom', verbose=True, **display_args):
        
        cap = cv2.VideoCapture(video_path)

        vid_name = os.path.basename(video_path)
        width = int(cap.get(3)) 
        height = int(cap.get(4))  
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_name = self.model_name + ' -- ' + vid_name.split('.')[0] + '.' + save_format
        save_file = os.path.join(save_dir, save_name)
        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vid_name} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + save_file)
        out = cv2.VideoWriter(save_file,
                              cv2.VideoWriter_fourcc(*"MPEG"),
                              30, (width, height))
        if not cap.isOpened():
            print("Error opening video stream or file")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            beg = time.time()
            results = self.predict_img(frame, verbose=False)
            if results is None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)
            if display == 'default':
                frame = self.default_display(**display_args)
            elif display == 'custom':
                frame == self.custom_display(**display_args)
            frame = cv2.putText(frame, f"FPS : {fps:,.2f}",
                                (5, 15), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
class YOLOv8_ObjectCounter(YOLOv8_ObjectDetector):
    def __init__(self, model_file = 'yolov8n.pt', labels= None, classes = None, conf = 0.25, iou = 0.45, 
                 track_max_age = 45, track_min_hits= 15, track_iou_threshold = 0.3 ):

        super().__init__(model_file , labels, classes, conf, iou)

        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold
    def predict_video(self, video_path, save_dir, save_format = "avi", 
                      display = 'custom', verbose = True, **display_args):
        cap = cv2.VideoCapture(video_path)
        vid_name = os.path.basename(video_path)
        width  = int(cap.get(3) ) 
        height = int(cap.get(4) )  
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_name = self.model_name + ' -- ' + vid_name.split('.')[0] + '.' + save_format
        save_file = os.path.join(save_dir, save_name)
        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vid_name} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + save_file)
        out = cv2.VideoWriter(save_file,
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            30,(width,height))
        if not cap.isOpened():
            print("Error opening video stream or file")
        tracker = sort.Sort(max_age = self.track_max_age, min_hits= self.track_min_hits , 
                            iou_threshold = self.track_iou_threshold)
        totalCount = []
        currentArray = np.empty((0, 5))
        while cap.isOpened():
            detections = np.empty((0, 5))
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            beg = time.time()
            results = self.predict_img(frame, verbose = False)
            if results == None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)
            for box in results.boxes:
                score = box.conf.item() * 100
                class_id = int(box.cls.item())
                x1 , y1 , x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)
                currentArray = np.array([x1, y1, x2, y2, score])
                detections = np.vstack((detections, currentArray))
            resultsTracker = tracker.update(detections)
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                id_txt = f"ID: {str(id)}"
                cv2.putText(frame, id_txt, (cx, cy), 4, 0.5, (0, 0, 255), 1)
                if totalCount.count(id) == 0:
                    totalCount.append(id)
            if display == 'default':
                frame = self.default_display(**display_args)
            elif display == 'custom':
                frame == self.custom_display( **display_args)
            frame = cv2.putText(frame,f"FPS : {fps:,.2f}" , 
                                (5,55), cv2.FONT_HERSHEY_COMPLEX, 
                            0.5,  (0,255,255), 1, cv2.LINE_AA)
            count_txt = f"TOTAL COUNT : {len(totalCount)}"
            frame = cv2.putText(frame, count_txt, (5,45), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        print(len(totalCount))
        print(totalCount)