# src/detect.py

import cv2
from ultralytics import YOLO
from src.config import Config
from src.utils import draw_bounding_boxes, classify_car_color, count_objects
from src.color_ranges import COLOR_RANGES
import os

def detect_objects(source_path, model_path='models/best.pt'):
    """
    Detects objects in an image or video stream and performs color classification for cars.
    Args:
        source_path (str): Path to the image file, video file, or '0' for webcam.
        model_path (str): Path to the trained YOLO model weights.
    """
    # Load the trained YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure the model path is correct and the model file exists.")
        return

    # Define colors for bounding boxes (BGR format)
    # These can be extended or made more dynamic
    BOX_COLORS = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "silver": (192, 192, 192),
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "pedestrian": (255, 0, 255), # Magenta for pedestrians
        "car": (0, 255, 0) # Default green for cars if color not classified
    }

    # Get class names from the model
    # If the model was trained on a custom dataset, these will be from data.yaml
    class_names = model.names

    # Determine if the source is a webcam, video, or image
    is_webcam = source_path == '0'
    if is_webcam:
        cap = cv2.VideoCapture(0)
    elif os.path.exists(source_path):
        cap = cv2.VideoCapture(source_path)
    else:
        print(f"Error: Source '{source_path}' not found.")
        return

    if not cap.isOpened() and not is_webcam:
        print(f"Error: Could not open video source {source_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame, conf=Config.CONF_THRESHOLD, iou=Config.IOU_THRESHOLD, verbose=False)

        processed_detections = []
        for r in results:
            for *box, conf, cls in r.boxes.data:
                det = {
                    'box': box.tolist(),
                    'conf': conf.item(),
                    'cls': cls.item()
                }
                label = class_names[int(cls.item())]

                # If it's a car, try to classify its color
                if label == 'car':
                    x1, y1, x2, y2 = map(int, box)
                    car_roi = frame[y1:y2, x1:x2]
                    car_color = classify_car_color(car_roi, COLOR_RANGES)
                    det['color'] = car_color
                processed_detections.append(det)

        # Draw bounding boxes and labels
        annotated_frame = draw_bounding_boxes(frame.copy(), processed_detections, class_names, BOX_COLORS)

        # Count objects
        current_counts = count_objects(processed_detections, class_names)

        # Display counts on the frame
        y_offset = 30
        for key, value in current_counts.items():
            if key == 'car_colors':
                cv2.putText(annotated_frame, "Car Colors:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 25
                for color, count in value.items():
                    cv2.putText(annotated_frame, f"  {color}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 20
            else:
                cv2.putText(annotated_frame, f"{key}: {value}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 25

        cv2.imshow("Traffic Analysis", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage:
    # To detect from a webcam:
    # detect_objects('0')
    # To detect from an image file:
    # detect_objects('path/to/your/image.jpg')
    # To detect from a video file:
    # detect_objects('path/to/your/video.mp4')

    print("Starting object detection. Please provide a source path (e.g., '0' for webcam, or a file path).")
    # For now, let's assume a placeholder for testing.
    # User will need to replace this with actual image/video path or '0' for webcam.
    # detect_objects('path/to/your/test_image.jpg') # Placeholder
    # You can uncomment one of the lines below for testing after training
    # detect_objects('0') # For webcam
    # detect_objects('data/test_images/test1.jpg') # Example test image path
    pass # Placeholder to avoid immediate error if no source is provided
