import os
import pandas as pd
import shutil

def organize_images(base_dir, output_base_dir):
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'), header=None, names=['filepath'])
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'), header=None, names=['filepath'])

    # Create output directories if they don't exist
    train_output_dir = os.path.join(output_base_dir, 'images', 'train')
    val_output_dir = os.path.join(output_base_dir, 'images', 'val')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    # Copy training images
    for filepath in train_df['filepath']:
        source_path_bayer = os.path.join(base_dir, 'bayer_good', filepath)
        source_path_bloom = os.path.join(base_dir, 'bloom_good', filepath)
        
        if os.path.exists(source_path_bayer):
            shutil.copy(source_path_bayer, os.path.join(train_output_dir, os.path.basename(filepath)))
        elif os.path.exists(source_path_bloom):
            shutil.copy(source_path_bloom, os.path.join(train_output_dir, os.path.basename(filepath)))

    # Copy validation images
    for filepath in test_df['filepath']:
        source_path_bayer = os.path.join(base_dir, 'bayer_good', filepath)
        source_path_bloom = os.path.join(base_dir, 'bloom_good', filepath)

        if os.path.exists(source_path_bayer):
            shutil.copy(source_path_bayer, os.path.join(val_output_dir, os.path.basename(filepath)))
        elif os.path.exists(source_path_bloom):
            shutil.copy(source_path_bloom, os.path.join(val_output_dir, os.path.basename(filepath)))

if __name__ == '__main__':
    BASE_INPUT_DIR = 'datasets/2024-Synset-Blvd-Open-Data-unzipped/Synset-Blvd-Original'
    OUTPUT_DIR = 'data'
    organize_images(BASE_INPUT_DIR, OUTPUT_DIR)
    print("Image organization complete.")