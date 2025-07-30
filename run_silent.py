#!/usr/bin/env python3
"""
Silent version of run.py for testing multi-threading functionality
"""

import os
import sys
import logging

# Suppress all logging
logging.basicConfig(level=logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['GLOG_minloglevel'] = '3'  # Suppress MediaPipe INFO and WARNING logs

# Import the main run module
from run import process_images

def main():
    """Main function to run the face verification test silently"""
    folder_path = "test"
    
    if not os.path.exists(folder_path):
        print(f"Error: {folder_path} directory not found")
        sys.exit(1)
    
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    if not folders:
        print(f"No folders found in {folder_path}")
        sys.exit(1)
    
    print(f"Processing {len(folders)} folders with multi-threading...")
    
    total_images = 0
    for folder in folders:
        folder_full_path = os.path.join(folder_path, folder)
        os.makedirs(f"output/{folder}", exist_ok=True)
        output_base_dir = f"output/{folder}/"
        
        # Count images in this folder
        image_count = 0
        for root, _, files in os.walk(folder_full_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_count += 1
        
        total_images += image_count
        print(f"Folder '{folder}': {image_count} images")
        
        # Process the folder
        results = process_images(folder_full_path, output_base_dir, max_workers=4)
        print(f"  Completed: {len(results)} images processed")
    
    print(f"\nTotal processing complete: {total_images} images across {len(folders)} folders")

if __name__ == "__main__":
    main() 