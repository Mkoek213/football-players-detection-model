from ultralytics import YOLO
import cv2
import cvzone
import math

def live_predict(model_path, setting, wait_key, classNames, video_path=None):
    """
    Perform live object detection using YOLO model.

    Parameters:
    - model_path (str): Path to the YOLO model weights file.
    - setting (str): Mode of operation, either 'live' for webcam or 'static' for video file.
    - wait_key (int): Time in milliseconds to wait between frames. A value of 0 means wait indefinitely.
    - classNames (list of str): List of class names that the model has been trained to recognize.
    - video_path (str, optional): Path to the video file for 'static' setting. Required if setting is 'static'.

    Raises:
    - ValueError: If 'setting' is not 'live' or 'static', or if 'video_path' is not provided for 'static' setting.
    """

    # Initialize video capture based on the setting
    if setting == 'live':
        # For live webcam feed
        cap = cv2.VideoCapture(0)  # Open default webcam
        cap.set(3, 640)  # Set the width of the frame to 640 pixels
        cap.set(4, 480)  # Set the height of the frame to 480 pixels
    elif setting == 'static':
        # For video file
        if video_path is None:
            raise ValueError("In 'static' setting you must pass video_path")
        cap = cv2.VideoCapture(video_path)  # Load video file
    else:
        # Raise an error if setting is invalid
        raise ValueError(f"Invalid setting '{setting}'. Expected 'live' or 'static'.")

    # Load the YOLO model from the specified path
    model = YOLO(model_path)

    # Define specific colors for selected classes
    classColors = {
        "ball": (255, 100, 50),  # Blue
        "goalkeeper": (128, 0, 128),  # Purple
        "player": (0, 0, 255),  # Red
        "referee": (255, 192, 203)  # Pink
    }
    while True:
        # Read a frame from the video capture
        success, img = cap.read()
        img = cv2.resize(img, (1280, 960))
        if not success:
            break  # End of video or cannot read frame

        # Perform object detection on the current frame
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box coordinates and convert to integers
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the color for the bounding box based on the detected class
                cls = classNames[int(box.cls[0])]
                color = classColors.get(cls, (255, 255, 255))  # Default to white if class not found in color map

                # Draw a thin rectangle around the detected object
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Thickness set to 2 for thin rectangles

                # Calculate the confidence score and format it
                conf = math.floor(box.conf[0] * 100) / 100

                # Display class name and confidence score
                cvzone.putTextRect(img, "", (max(0, x1), max(35, y1)), scale=1.5, thickness=2, offset=3, colorR=color, colorT=(0, 0, 0))

        # Display the resulting frame in a window
        cv2.imshow("Image", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(wait_key) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define class names for different settings
    class_names = [
        'ball', 'goalkeeper', 'player', 'referee'
    ]

    # Run the live_predict function with the fine-tuned model and specified settings
    live_predict(
        model_path='models/yolov8n_.pt',
        setting='static',
        wait_key=10,
        classNames=class_names,
        video_path='test_videos/test_video.mp4'
    )


