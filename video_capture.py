import numpy as np
import cv2
import sys
import os
from ppe_detect import YOLOv8_ObjectDetector

# Labels and object detector instantiation
ppe_labels = ["Arc Flash Suit", "Helmet", "Safety shoes", "Electric hand gloves", "Voltage regulator","No Gloves","No shoes","No Arc Suit","No Helmet"]
ppe_detector = YOLOv8_ObjectDetector(model_file=r"D:/runs/detect/train8/weights/best.pt", labels=ppe_labels)

# Webcam capture
cap = cv2.VideoCapture(0)  # 0 represents the default webcam index

# Check if the webcam capture was successful
if not cap.isOpened():
    sys.exit("Error opening webcam capture.")

# Process every 50th frame
frame_skip = 200  # Process every 50th frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        detections = ppe_detector.predict_img(frame)
        display_img = ppe_detector.default_display(show_conf=False, pil=False, example_frame=frame)

        # Display the frame with detections
        cv2.imshow('Frame with Detections', display_img)

        # Print labels that are not detected
        detected_labels = [label for label in detections]
        missing_labels = [label for label in ppe_labels if label not in detected_labels]
        if missing_labels:
            print("Frame", frame_count, "- Missing labels:", ", ".join(missing_labels), flush=True)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_count += 1

# Release the webcam capture
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()