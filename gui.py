
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Assuming utils and color_ranges are in the src folder
from src.utils import draw_bounding_boxes, classify_car_color, count_objects
from src.color_ranges import COLOR_RANGES
from src.config import Config

class ObjectDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detector AI")
        self.root.geometry("1200x800")

        # --- UI Elements ---
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.btn_load = ttk.Button(self.main_frame, text="Load Image", command=self.load_image_and_process)
        self.btn_load.pack(pady=10)

        self.image_label = ttk.Label(self.main_frame)
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)

        self.results_label = ttk.Label(self.main_frame, text="Loading model...", font=("Helvetica", 12))
        self.results_label.pack(pady=10)
        
        self.root.update_idletasks() # Ensure results_label is created before loading

        # --- Load Model ---
        # Using 'yolov8n.pt' as it's a reliable base model. 
        # The 'models/best.pt' file appears to be corrupted.
        self.model = self.load_model('yolov8n.pt')


    def load_model(self, model_path):
        if not os.path.exists(model_path):
            # Handle error in UI
            self.results_label.config(text=f"Error: Model not found at {model_path}")
            return None
        try:
            model = YOLO(model_path)
            self.results_label.config(text="Model loaded successfully. Please load an image.")
            return model
        except Exception as e:
            self.results_label.config(text=f"Error loading model: {e}. Please check the model file.")
            return None

    def load_image_and_process(self):
        if self.model is None:
            self.results_label.config(text="Model is not loaded. Cannot process image.")
            return

        file_path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ])
        if not file_path:
            return

        self.results_label.config(text="Processing...")
        self.root.update_idletasks() # Update UI to show processing message

        try:
            # Process the image
            annotated_image, counts = self.process_image(file_path)

            # Convert for Tkinter
            img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            
            # Resize image to fit window while maintaining aspect ratio
            self.main_frame.update_idletasks()
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            
            img.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image=img)

            # Update UI
            self.image_label.config(image=photo)
            self.image_label.image = photo # Keep a reference

            # Format and display counts
            results_text = self.format_counts(counts)
            self.results_label.config(text=results_text)

        except Exception as e:
            self.results_label.config(text=f"An error occurred: {e}")

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Could not read the image file.")

        class_names = self.model.names
        box_colors = {
            "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0),
            "white": (255, 255, 255), "black": (0, 0, 0), "silver": (192, 192, 192),
            "yellow": (0, 255, 255), "orange": (0, 165, 255), "pedestrian": (255, 0, 255),
            "car": (0, 255, 0)
        }

        results = self.model(frame, conf=Config.CONF_THRESHOLD, iou=Config.IOU_THRESHOLD, verbose=False)

        processed_detections = []
        try:
            if results:
                r = results[0]
                # Check if the result object has the expected '.boxes' attribute
                if not hasattr(r, 'boxes'):
                    raise TypeError(f"Unexpected result type. The object (type: {type(r)}) lacks a '.boxes' attribute.")

                for *box, conf, cls in r.boxes.data:
                    # The *box assignment already creates a list, so .tolist() is not needed.
                    det = {'box': box, 'conf': conf.item(), 'cls': cls.item()}
                    label = class_names[int(cls.item())]
                    if label == 'car':
                        x1, y1, x2, y2 = map(int, box)
                        car_roi = frame[y1:y2, x1:x2]
                        car_color = classify_car_color(car_roi, COLOR_RANGES)
                        det['color'] = car_color
                    processed_detections.append(det)
        except Exception as e:
            # Re-raise the exception with more context for better debugging
            raise RuntimeError(f"Failed during detection processing. The 'results' variable is type: {type(results)}. Error: {e}")

        annotated_frame = draw_bounding_boxes(frame.copy(), processed_detections, class_names, box_colors)
        current_counts = count_objects(processed_detections, class_names)

        return annotated_frame, current_counts

    def format_counts(self, counts):
        total_cars = counts.get('car', 0)
        total_pedestrians = counts.get('pedestrian', 0)
        car_colors = counts.get('car_colors', {})

        results_text = f"Total Cars: {total_cars} | Total Pedestrians: {total_pedestrians}\n"
        if car_colors:
            color_breakdown = ", ".join([f"{color.capitalize()}: {count}" for color, count in car_colors.items()])
            results_text += f"Car Color Breakdown: {color_breakdown}"
        return results_text

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()
