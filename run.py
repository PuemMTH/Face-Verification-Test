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

def process_images(folder_path, output_base_dir):
    """Process all images in the input folder and its subfolders for check_face_min_size, check_eye_status, and check_lightpol"""
    # Load config from yml file
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    
    results = []
    timing_results = []
    
    # Initialize timing totals for summary
    timing_totals = {
        "get_lm": 0.0,
        "check_face_min_size": 0.0,
        "check_eye_status": 0.0,
        "check_lightpol": 0.0,
        "check_face_blur": 0.0,
        "check_head_fully": 0.0,
        "check_head_pose": 0.0
    }
    
    # Walk through the folder
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, filename)
                result = {
                    "image_name": filename,
                    "face_message": "",
                    "eye_message": "",
                    "light_message": "",
                    "blur_message": "",
                    "head_fully_message": "",
                    "head_pose_message": ""
                }
                timing = {
                    "image_name": filename,
                    "get_lm_time": 0.0,
                    "check_face_min_size_time": 0.0,
                    "check_eye_status_time": 0.0,
                    "check_lightpol_time": 0.0,
                    "check_face_blur_time": 0.0,
                    "check_head_fully_time": 0.0,
                    "check_head_pose_time": 0.0
                }
                
                # Get landmarks and bounding box
                start_time = time.time()
                success, msg, landmarks, bbox = get_lm(image_path)
                timing["get_lm_time"] = time.time() - start_time
                timing_totals["get_lm"] += timing["get_lm_time"]
                
                # Check face size
                start_time = time.time()
                face_success, face_message = check_face_min_size(bbox, config['threshold']['face_size'])
                timing["check_face_min_size_time"] = time.time() - start_time
                timing_totals["check_face_min_size"] += timing["check_face_min_size_time"]
                result["face_message"] = face_message
                
                # Check eye status
                start_time = time.time()
                eye_success, eye_message = check_eye_status(landmarks, success, msg, config['threshold']['EAR_THRESHOLD'])
                timing["check_eye_status_time"] = time.time() - start_time
                timing_totals["check_eye_status"] += timing["check_eye_status_time"]
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
                timing_totals["check_lightpol"] += timing["check_lightpol_time"]
                result["light_message"] = light_status
                
                # Check face blur
                start_time = time.time()
                blur_success, blur_status = check_face_blur(
                    image_path,
                    config['threshold']['blur']
                )
                timing["check_face_blur_time"] = time.time() - start_time
                timing_totals["check_face_blur"] += timing["check_face_blur_time"]
                result["blur_message"] = blur_status
                
                # Check head fully
                start_time = time.time()
                head_fully_success, head_fully_status = analyze_single_image(image_path)
                timing["check_head_fully_time"] = time.time() - start_time
                timing_totals["check_head_fully"] += timing["check_head_fully_time"]
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
                timing_totals["check_head_pose"] += timing["check_head_pose_time"]
                result["head_pose_message"] = head_pose_status
                
                results.append(result)
                timing_results.append(timing)
    
    
    # Calculate total time across all functions
    total_all_functions = sum(timing_totals.values())
    
    # Save results to CSV files
    # Main results
    df_results = pd.DataFrame(results, columns=["image_name", "face_message", "eye_message", "light_message", "blur_message", "head_fully_message", "head_pose_message"])
    df_results.to_csv(os.path.join(output_base_dir, "results.csv"), index=False)
    
    # Timing per image
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
    
    # Summary of total time per function
    summary_data = [
        {"Function": func, "Total_Time_Seconds": total}
        for func, total in timing_totals.items()
    ]
    summary_data.append({"Function": "Total_All_Functions", "Total_Time_Seconds": total_all_functions})
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_base_dir, "summary.csv"), index=False)
    
    return results

# Example usage
folder_path = r"test"
for folder in os.listdir(folder_path):
    folder_full_path = os.path.join(folder_path, folder)
    if os.path.isdir(folder_full_path):  # Only process directories
        os.makedirs(f"output/{folder}", exist_ok=True)
        output_base_dir = f"output/{folder}/"
        results = process_images(folder_full_path, output_base_dir)