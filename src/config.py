# src/config.py

class Config:
    # Model and Training Parameters
    IMAGE_SIZE = 640  # Input image size for the model (e.g., 640x640)
    BATCH_SIZE = 16   # Batch size for training
    EPOCHS = 2      # Number of training epochs
    MODEL_NAME = 'yolov8n.pt' # Pre-trained YOLO model to use (e.g., yolov8n, yolov8s)

    # Paths
    DATA_YAML_PATH = 'data/data.yaml' # Path to the YOLO dataset configuration file
    SAVE_DIR = 'models/'              # Directory to save trained model weights

    # Detection Parameters
    CONF_THRESHOLD = 0.25 # Confidence threshold for object detection
    IOU_THRESHOLD = 0.45  # IoU threshold for Non-Maximum Suppression (NMS)

    # Class Names (will be overridden by data.yaml during training)
    # These are just placeholders for initial setup
    CLASSES = ['car', 'pedestrian']
