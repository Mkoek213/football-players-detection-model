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


def run_player_detection(source_video_path: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    # Load the YOLO model with specified weights on a CUDA device
    player_detection_model = YOLO("models/yolov8n_transfer_road_model.pt").to(device="cuda")

    # Generate frames from the source video
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    # Iterate over frames
    for frame in frame_generator:
        # Perform player detection on the frame
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]

        # Convert the detection results to a usable format
        detections = sv.Detections.from_ultralytics(result)

        # Copy and annotate the frame with detection boxes and labels
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)

        # Yield the annotated frame
        yield annotated_frame


def live_predict(model_path, setting, wait_key, video_path=None):
    """
    Perform live object detection using YOLO model.

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
        frame_generator = sv.get_video_frames_generator(source_path=video_path)
    else:
        raise ValueError(f"Invalid setting '{setting}'. Expected 'live' or 'static'.")

    # Load the YOLO model from the specified path
    model = YOLO(model_path)

    # Set up frame generation for static video
    frame_generator = run_player_detection(
        source_video_path=video_path)

    # Get video information and configure display window size
    video_info = sv.VideoInfo.from_video_path(video_path)

    # Initialize video sink and process each frame
    with sv.VideoSink("result", video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)  # Write frame to the output video
            frame = cv2.resize(frame, (1200, 900))
            cv2.imshow("frame", frame)  # Display the frame
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  # Exit if 'q' key is pressed

        cv2.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    # Run the live_predict function with the YOLO model and specified settings
    live_predict(
        model_path='models/yolov8n_transfer_road_model.pt',
        setting='static',
        wait_key=1,
        video_path='test_videos/test_video.mp4'
    )
