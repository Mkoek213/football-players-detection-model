from ultralytics import YOLO
import cv2
import cvzone
import math
import supervision as sv
from typing import Iterator, List
import numpy as np

# Define annotators for drawing boxes and labels on frames
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FF6347', '#FFD700']),
    thickness=2
)

BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FF6347', '#FFD700']),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)


def live_predict(model_path, setting, wait_key, video_path=None):
    """
    Perform live object detection using YOLO model and save the output to a video file.

    Parameters:
    - model_path (str): Path to the YOLO model weights file.
    - setting (str): Mode of operation, either 'live' for webcam or 'static' for video file.
    - wait_key (int): Time in milliseconds to wait between frames.
    - video_path (str, optional): Path to the video file for 'static' setting.
    """

    # Initialize video capture based on the setting (live or static)
    if setting == 'live':
        cap = cv2.VideoCapture(0)  # Open default webcam
        cap.set(3, 640)  # Set the width of the frame to 640 pixels
        cap.set(4, 480)  # Set the height of the frame to 480 pixels
    elif setting == 'static':
        # If static, ensure video path is provided and generate frames
        if video_path is None:
            raise ValueError("In 'static' setting you must pass video_path")
        cap = cv2.VideoCapture(video_path)  # Open the video file
    else:
        raise ValueError(f"Invalid setting '{setting}'. Expected 'live' or 'static'.")

    # Load the YOLO model from the specified path
    model = YOLO(model_path)


    # Set up the VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('res.mp4',fourcc, 20.0, (1200, 900))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frame is captured

        # Run YOLO detection on the frame
        result = model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)

        # Resize and display the frame
        annotated_frame = cv2.resize(annotated_frame, (1200, 900))
        cv2.imshow("frame", annotated_frame)

        # Write the processed frame to the output video
        out.write(annotated_frame)

        if cv2.waitKey(wait_key) & 0xFF == ord("q"):
            break  # Exit if 'q' key is pressed

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    # Run the live_predict function with the YOLO model and specified settings
    live_predict(
        model_path='models/yolov8n_transfer_road_model.pt',
        setting='static',
        wait_key=1,
        video_path='test_videos/test_video.mp4'
    )
