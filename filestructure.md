car-detector-ai/
�
��� data/                      # Dataset folder
�   ��� train/
�   �   ��� images/
�   �   ��� labels/
�   ��� val/
�   �   ��� images/
�   �   ��� labels/
�   ��� data.yaml               # YOLO dataset config file
�
��� models/                     # Saved model weights
�   ��� best.pt                  # Trained model
�
��� src/                        # All source code
�   ��� __init__.py
�   ��� train.py                 # Training script
�   ��� detect.py                # Inference + drawing rectangles
�   ��� utils.py                 # Helper functions (color check, drawing, counting)
�   ��� config.py                # Configurations (paths, thresholds, etc.)
�   ��� color_ranges.py          # HSV ranges for color detection
�
�
��� requirements.txt             # Python dependencies
��� README.md                    # Project description & setup guide
��� .gitignore                   # Ignore unnecessary files
ewew
