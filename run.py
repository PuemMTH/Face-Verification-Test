import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe INFO and WARNING logs
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # Additional filter for MediaPipe
import pandas as pd
import shutil
import yaml
import pandas as pd
import shutil
import yaml
from func.check_head_pose import check_head_pose
from func.check_face_blur import check_face_blur
from func.check_face_size import check_face_min_size
from func.check_light_pollution import check_lightpol
from func.check_eye import check_eye_status
from func.get_landmarks import get_lm
from func.check_head_fully import analyze_single_image
import time
from pandas import ExcelWriter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import csv

def save_results_incrementally(results, timing_results, output_base_dir, timing_totals):
    """Save results to CSV files incrementally"""
    # Save main results
    df_results = pd.DataFrame(results, columns=["image_name", "face_message", "eye_message", "light_message", "blur_message", "head_fully_message", "head_pose_message"])
    df_results.to_csv(os.path.join(output_base_dir, "results.csv"), index=False)
    
    # Save timing per image
    df_timing = pd.DataFrame(timing_results, columns=[
        "image_name",
        "get_lm_time",
        "check_face_min_size_time",
        "check_eye_status_time",
        "check_lightpol_time",
        "check_face_blur_time",
        "check_head_fully_time",
        "check_head_pose_time"
    ])
    df_timing.to_csv(os.path.join(output_base_dir, "timing_per_image.csv"), index=False)
    
    # Calculate and save summary
    total_all_functions = sum(timing_totals.values())
    summary_data = [
        {"Function": func, "Total_Time_Seconds": total}
        for func, total in timing_totals.items()
    ]
    summary_data.append({"Function": "Total_All_Functions", "Total_Time_Seconds": total_all_functions})
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_base_dir, "summary.csv"), index=False)

def process_single_image(image_path, config):
    """Process a single image and return results and timing"""
    result = {
        "image_name": os.path.basename(image_path),
        "face_message": "",
        "eye_message": "",
        "light_message": "",
        "blur_message": "",
        "head_fully_message": "",
        "head_pose_message": ""
    }
    timing = {
        "image_name": os.path.basename(image_path),
        "get_lm_time": 0.0,
        "check_face_min_size_time": 0.0,
        "check_eye_status_time": 0.0,
        "check_lightpol_time": 0.0,
        "check_face_blur_time": 0.0,
        "check_head_fully_time": 0.0,
        "check_head_pose_time": 0.0
    }
    
    try:
        # Get landmarks and bounding box
        start_time = time.time()
        success, msg, landmarks, bbox = get_lm(image_path)
        timing["get_lm_time"] = time.time() - start_time
        
        # Check face size
        start_time = time.time()
        face_success, face_message = check_face_min_size(bbox, config['threshold']['face_size'])
        timing["check_face_min_size_time"] = time.time() - start_time
        result["face_message"] = face_message
        
        # Check eye status
        start_time = time.time()
        eye_success, eye_message = check_eye_status(landmarks, success, msg, config['threshold']['EAR_THRESHOLD'])
        timing["check_eye_status_time"] = time.time() - start_time
        result["eye_message"] = eye_message
        
        # Check lighting
        start_time = time.time()
        light_success, light_status = check_lightpol(
            image_path,
            config['threshold']['dark_threshold'],
            config['threshold']['bright_threshold'],
            config['threshold']['diff_threshold'],
            config['threshold']['margin']
        )
        timing["check_lightpol_time"] = time.time() - start_time
        result["light_message"] = light_status
        
        # Check face blur
        start_time = time.time()
        blur_success, blur_status = check_face_blur(
            image_path,
            config['threshold']['blur']
        )
        timing["check_face_blur_time"] = time.time() - start_time
        result["blur_message"] = blur_status
        
        # Check head fully
        start_time = time.time()
        head_fully_success, head_fully_status = analyze_single_image(image_path)
        timing["check_head_fully_time"] = time.time() - start_time
        result["head_fully_message"] = head_fully_status
        
        # Check head pose
        start_time = time.time()
        head_pose_result = check_head_pose(image_path)
        if isinstance(head_pose_result, str):
            head_pose_success, head_pose_status = False, head_pose_result
        elif isinstance(head_pose_result, tuple) and len(head_pose_result) >= 2:
            head_pose_success, head_pose_status = head_pose_result[:2]
        else:
            head_pose_success, head_pose_status = False, "Error: Invalid head pose result"
        timing["check_head_pose_time"] = time.time() - start_time
        result["head_pose_message"] = head_pose_status
        
    except Exception as e:
        result["face_message"] = f"Error: {str(e)}"
        result["eye_message"] = f"Error: {str(e)}"
        result["light_message"] = f"Error: {str(e)}"
        result["blur_message"] = f"Error: {str(e)}"
        result["head_fully_message"] = f"Error: {str(e)}"
        result["head_pose_message"] = f"Error: {str(e)}"
    
    return result, timing

def process_images(folder_path, output_base_dir, max_workers=4):
    """Process all images in the input folder and its subfolders using multi-threading with incremental saving"""
    # Load config from yml file
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    
    # Collect all image files
    image_files = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                image_files.append(image_path)
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return []
    
    results = []
    timing_results = []
    timing_totals = {
        "get_lm": 0.0,
        "check_face_min_size": 0.0,
        "check_eye_status": 0.0,
        "check_lightpol": 0.0,
        "check_face_blur": 0.0,
        "check_head_fully": 0.0,
        "check_head_pose": 0.0
    }
    
    # Thread lock for safe file writing
    file_lock = threading.Lock()
    
    def process_and_save(image_path):
        """Process single image and save results immediately"""
        result, timing = process_single_image(image_path, config)
        
        # Update timing totals
        with file_lock:
            timing_totals["get_lm"] += timing["get_lm_time"]
            timing_totals["check_face_min_size"] += timing["check_face_min_size_time"]
            timing_totals["check_eye_status"] += timing["check_eye_status_time"]
            timing_totals["check_lightpol"] += timing["check_lightpol_time"]
            timing_totals["check_face_blur"] += timing["check_face_blur_time"]
            timing_totals["check_head_fully"] += timing["check_head_fully_time"]
            timing_totals["check_head_pose"] += timing["check_head_pose_time"]
            
            results.append(result)
            timing_results.append(timing)
            
            # Save results incrementally every 10 images or immediately for small batches
            if len(results) % 10 == 0 or len(results) == len(image_files):
                save_results_incrementally(results, timing_results, output_base_dir, timing_totals)
        
        return result, timing
    
    # Process images with multi-threading and progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_and_save, image_path): image_path 
            for image_path in image_files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(image_files), desc=f"Processing {os.path.basename(folder_path)}", 
                  unit="image") as pbar:
            for future in as_completed(future_to_image):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing {future_to_image[future]}: {str(e)}")
                    pbar.update(1)
    
    # Final save to ensure all results are written
    save_results_incrementally(results, timing_results, output_base_dir, timing_totals)
    
    return results

# Example usage with multi-threading and progress tracking
if __name__ == "__main__":
    folder_path = r"/project/lt200384-ff_bio/datasets/ff_mix_crop"
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    print(f"Found {len(folders)} folders to process")
    
    # Process each folder with progress tracking
    for folder in tqdm(folders, desc="Processing folders", unit="folder"):
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):  # Only process directories
            os.makedirs(f"output/{folder}", exist_ok=True)
            output_base_dir = f"output/{folder}/"
            results = process_images(folder_full_path, output_base_dir, max_workers=4)
            print(f"Completed processing {folder}: {len(results)} images processed")