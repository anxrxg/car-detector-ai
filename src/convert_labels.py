
import os
import cv2
import numpy as np
import pandas as pd

def mask_to_yolo_bbox(mask_path, output_dir, class_index):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not read mask image {mask_path}")
        return

    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        bbox_width = w / width
        bbox_height = h / height
        yolo_lines.append(f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}")

    if yolo_lines:
        base_filename = os.path.splitext(os.path.basename(mask_path))[0]
        output_path = os.path.join(output_dir, f"{base_filename}.txt")
        with open(output_path, 'w') as f:
            f.write("\n".join(yolo_lines))

def process_directory(base_dir, output_base_dir):
    labels_dir = os.path.join(base_dir, 'label')
    
    # Create a list of all subdirectories in the labels_dir
    class_names = [d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))]

    for class_name in class_names:
        class_index = class_names.index(class_name)
        mask_dir = os.path.join(labels_dir, class_name)
        
        # Determine if we are processing for training or testing based on the csv file
        train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'), header=None, names=['filepath'])
        test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'), header=None, names=['filepath'])

        train_files = set(train_df['filepath'].apply(lambda x: os.path.basename(x).split('.')[0]))
        test_files = set(test_df['filepath'].apply(lambda x: os.path.basename(x).split('.')[0]))

        for filename in os.listdir(mask_dir):
            if filename.lower().endswith('.png'):
                file_id = os.path.splitext(filename)[0]
                
                if file_id in train_files:
                    output_dir = os.path.join(output_base_dir, 'labels', 'train')
                elif file_id in test_files:
                    output_dir = os.path.join(output_base_dir, 'labels', 'val')
                else:
                    continue # Skip files that are not in train or test csv

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                mask_path = os.path.join(mask_dir, filename)
                mask_to_yolo_bbox(mask_path, output_dir, class_index)

if __name__ == '__main__':
    BASE_INPUT_DIR = 'datasets/2024-Synset-Blvd-Open-Data-unzipped/Synset-Blvd-Original'
    LABELS_OUTPUT_DIR = 'data'
    
    # Create necessary directories
    os.makedirs(os.path.join(LABELS_OUTPUT_DIR, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(LABELS_OUTPUT_DIR, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(LABELS_OUTPUT_DIR, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(LABELS_OUTPUT_DIR, 'images', 'val'), exist_ok=True)

    process_directory(BASE_INPUT_DIR, LABELS_OUTPUT_DIR)
    print("Conversion complete.")
