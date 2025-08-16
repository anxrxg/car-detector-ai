import os
import yaml

# Get the class names from the directory structure
labels_dir = 'datasets/2024-Synset-Blvd-Open-Data-unzipped/Synset-Blvd-Original/label'
class_names = [d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))]

# Create the data dictionary
data = {
    'train': '../data/images/train',
    'val': '../data/images/val',
    'nc': len(class_names),
    'names': class_names
}

# Write the data to the data.yaml file
with open('data/data.yaml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False)

print("data.yaml file created successfully.")