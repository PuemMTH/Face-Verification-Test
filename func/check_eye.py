import numpy as np
from typing import Tuple, List, Optional

def calculate_ear(landmarks: List[Tuple[int, int, float]], eye_indices: List[int]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for a given eye using specified landmark indices.
    
    Args:
        landmarks: List of (x, y, z) coordinates from get_lm
        eye_indices: List of 6 indices for eye landmarks [p1, p2, p3, p4, p5, p6]
    
    Returns:
        float: Eye Aspect Ratio
    """
    try:
        # Extract the 6 points for the eye
        p1 = np.array(landmarks[eye_indices[0]][:2])  # Horizontal point 1
        p2 = np.array(landmarks[eye_indices[1]][:2])  # Vertical point 1
        p3 = np.array(landmarks[eye_indices[2]][:2])  # Vertical point 2
        p4 = np.array(landmarks[eye_indices[3]][:2])  # Horizontal point 2
        p5 = np.array(landmarks[eye_indices[4]][:2])  # Vertical point 3
        p6 = np.array(landmarks[eye_indices[5]][:2])  # Vertical point 4

        # Calculate distances
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)

        # Avoid division by zero
        if horizontal == 0:
            return 0.0

        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    except Exception as e:
        return 0.0

def check_eye_status(landmarks, success, message,EAR_THRESHOLD) -> Tuple[bool, str]:
    """
    Check if both eyes are open or closed using landmarks from get_lm function.
    
    Args:
        landmarks: List of (x, y, z) coordinates from get_lm
        success: Boolean indicating if landmark detection was successful
        message: Status or error message from get_lm
    
    Returns:
        Tuple[bool, str]: (success, message)
        - success: True if both eyes are open, False otherwise
        - message: Status or error message
    """
    # Eye landmark indices (from MediaPipe Face Mesh)
    LEFT_EYE_INDICES = [33, 160, 159, 133, 158, 157]
    RIGHT_EYE_INDICES = [362, 387, 386, 263, 385, 384]
    
    # EAR threshold (adjust based on testing, typically 0.2-0.3)
    EAR_THRESHOLD = EAR_THRESHOLD

    if not success or landmarks is None:
        return (False, message)
    
    try:
        # Calculate EAR for both eyes
        left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)

        # Check if both eyes are open
        if left_ear > EAR_THRESHOLD and right_ear > EAR_THRESHOLD:
            return (True, "Both eyes are open")
        else:
            return (False, "One or both eyes are closed")

    except Exception as e:
        return (False, f"Error during eye status detection: {str(e)}")