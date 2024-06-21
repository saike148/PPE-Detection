import numpy as np
import sys
import cv2
import pandas as pd
from datetime import datetime
from ppe_detect import YOLOv8_ObjectDetector
import os
import win32com.client as win32
import time

ppe_labels = ["Arc Flash Suit", "Helmet", "Safety shoes", "Electric hand gloves", "Voltage regulator", "No Gloves", "No Shoes", "No Arc Suit", "No Helmet"]

def send_email_with_frame(subject, body, recipient, frame):
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.Subject = subject
    mail.Body = body
    mail.To = recipient

    temp_frame_filename = 'temp_frame.jpg'
    cv2.imwrite(temp_frame_filename, frame)
    
    attachment = os.path.abspath(temp_frame_filename)
    mail.Attachments.Add(attachment)
    
    mail.Send()

def process_video(video_filename, ppe_detector, frame_skip):
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        sys.exit(f"Error opening video file: {video_filename}")

    frame_count = 0
    detection_results = {}
    time_interval = 10  # Time interval in seconds
    last_detection_time = time.time() - time_interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            current_time = time.time()
            if current_time - last_detection_time >= time_interval:
                last_detection_time = current_time

                detections = ppe_detector.predict_img(frame)
                display_img = frame.copy()

                frame_detections = set()
                for result in detections:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_name = ppe_detector.labels[cls]
                        frame_detections.add(class_name)

                        x1, y1, x2, y2 = box.xyxy[0]
                        cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(display_img, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if frame_detections:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                    if timestamp not in detection_results:
                        detection_results[timestamp] = []

                    email_subject = "PPE Violation Detected"
                    email_body = f"Detected violations: {', '.join(frame_detections)}\nTimestamp: {timestamp}\nCamera: Camera 1\nSite ID: 1\n"
                    recipient_email = "recipient@example.com"  # Replace with recipient's email address

                    send_email_with_frame(email_subject, email_body, recipient_email, frame)

                cv2.imshow('Video with Detections', display_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return detection_results

if __name__ == "__main__":
    ppe_detector = YOLOv8_ObjectDetector(model_file="D:/runs/detect/train15/weights/best.pt", labels=ppe_labels)
    video_filename = "D:/Saikrishna/Downloads/videosppe/958f8a00-34ee-488b-bc96-c7dc4863a611.mp4"  # Use the desired video filename
    frame_skip = 120

    all_detection_results = {}

    print(f"Processing video: {video_filename}")
    detection_results = process_video(video_filename, ppe_detector, frame_skip)

    df_data = []
    for idx, (timestamp, violations) in enumerate(detection_results.items()):
        unique_violations = list(set(violations))  # Remove duplicates

        # Determine severity based on violations
        severity = "High" if any(label in unique_violations for label in ["No Arc Suit", "No Hand Glove", "No Helmet"]) else "Low"

        # Create DataFrame row
        row = {'ID': idx, 'Timestamp': timestamp, 'Violations': ', '.join(unique_violations),
               'Shift': np.random.choice(['morning', 'evening', 'night']),
               'Camera': np.random.choice(['Camera 1', 'Camera 2']),
               'Site_ID': np.random.choice([1, 2, 3]),
               'Severity': severity,
               'Weight': 'NA',
               'Image_Path': 'NA'}
        df_data.append(row)