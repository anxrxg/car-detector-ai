# src/utils.py

import cv2
import numpy as np
from src.color_ranges import COLOR_RANGES

def draw_bounding_boxes(image, detections, class_names, colors):
    """
    Draws bounding boxes and labels on the image.
    Args:
        image (np.array): The input image.
        detections (list): List of dictionaries, each containing 'box', 'conf', 'cls', 'color' (optional).
        class_names (dict): Dictionary mapping class IDs to names.
        colors (dict): Dictionary mapping class names or colors to BGR tuples for default colors.
    Returns:
        np.array: Image with bounding boxes drawn.
    """
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        conf = det['conf']
        cls = int(det['cls'])
        label = class_names.get(cls, 'unknown')
        
        detected_color_name = det.get('color', None)

        # --- New, cleaner bounding box color logic ---
        if label == 'car':
            if detected_color_name == 'blue':
                box_color = (0, 0, 255)  # RED for blue cars
            elif detected_color_name and detected_color_name != 'unknown':
                box_color = (255, 0, 0)  # BLUE for other CLASSIFIED cars
            else:
                box_color = colors.get('car', (0, 255, 0)) # Default color for UNCLASSIFIED cars
        else:
            # Default behavior for non-car objects (e.g., pedestrians)
            box_color = colors.get(label, (0, 255, 0)) # Default to green if not specified

        # --- Label Formatting (Always show the real detected color) ---
        if detected_color_name and detected_color_name != 'unknown':
            display_label = f"{detected_color_name} {label} {conf:.2f}"
        else:
            display_label = f"{label} {conf:.2f}"

        # Draw the bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(image, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
    return image

def classify_car_color(car_roi, color_ranges):
    """
    Classifies the color of a car's region of interest (ROI).
    Args:
        car_roi (np.array): The image region containing the car.
        color_ranges (dict): Dictionary of HSV color ranges.
    Returns:
        str: The classified color name (e.g., 'red', 'blue'), or 'unknown'.
    """
    if car_roi is None or car_roi.size == 0:
        return "unknown"

    hsv_roi = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)

    # Calculate the average HSV values for the ROI
    # This can be a simple average or more sophisticated (e.g., dominant color)
    # For simplicity, let's use a simple average for now
    h_mean, s_mean, v_mean = np.mean(hsv_roi, axis=(0, 1))

    # Iterate through predefined color ranges to find a match
    for color_name, ranges in color_ranges.items():
        if color_name == "red": # Red wraps around in HSV
            lower1 = np.array(ranges["lower1"])
            upper1 = np.array(ranges["upper1"])
            lower2 = np.array(ranges["lower2"])
            upper2 = np.array(ranges["upper2"])

            mask1 = cv2.inRange(hsv_roi, lower1, upper1)
            mask2 = cv2.inRange(hsv_roi, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower = np.array(ranges["lower"])
            upper = np.array(ranges["upper"])
            mask = cv2.inRange(hsv_roi, lower, upper)

        # Check if a significant portion of the ROI falls within the color range
        # You might need to adjust this threshold (e.g., 10% of pixels)
        if np.sum(mask) > (car_roi.shape[0] * car_roi.shape[1] * 0.1):
            return color_name

    return "unknown"

def count_objects(detections, class_names):
    """
    Counts detected objects by class and car color.
    Args:
        detections (list): List of dictionaries, each containing 'box', 'conf', 'cls', 'color' (optional).
        class_names (dict): Dictionary mapping class IDs to names.
    Returns:
        dict: A dictionary with counts for each class and car color.
    """
    # Initialize counts for each class name (the values in the class_names dict)
    counts = {name: 0 for name in class_names.values()}
    counts['car_colors'] = {}

    for det in detections:
        cls = int(det['cls'])
        # Ensure the class ID from the detection exists in the model's class list
        if cls in class_names:
            label = class_names[cls]
            counts[label] += 1

            if label == 'car' and 'color' in det and det['color'] != 'unknown':
                color_name = det['color']
                counts['car_colors'][color_name] = counts['car_colors'].get(color_name, 0) + 1
    return counts
