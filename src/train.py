# src/train.py

from ultralytics import YOLO
from config import Config
import os
import yaml

def train_model():
    """
    Trains the YOLO model using the configuration defined in src/config.py.
    """
    # Load a model
    # If Config.MODEL_NAME is a path to a .pt file, it loads that.
    # If it's a model name (e.g., 'yolov8n.pt'), it downloads it.
    model = YOLO(Config.MODEL_NAME)

    # Ensure the save directory exists
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    # Train the model
    results = model.train(
        data=Config.DATA_YAML_PATH,
        imgsz=Config.IMAGE_SIZE,
        epochs=Config.EPOCHS,
        batch=Config.BATCH_SIZE,
        name='traffic_analysis_model', # Name for the training run
        project=Config.SAVE_DIR,      # Save results in the models directory
        conf=Config.CONF_THRESHOLD,   # Confidence threshold for validation
        iou=Config.IOU_THRESHOLD,      # IoU threshold for validation
        device='cpu' # Use CPU
    )

    print("Training complete. Results saved to:", os.path.join(Config.SAVE_DIR, 'traffic_analysis_model'))

if __name__ == "__main__":
    print("Starting model training...")
    train_model()
