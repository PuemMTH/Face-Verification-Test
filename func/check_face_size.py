import os
import cv2

def check_face_min_size(bbox, min_size):
    """
    Checks if the face bounding box meets the minimum size criteria.
    Args:
        bbox: Tuple of (x, y, w, h) representing the bounding box
        min_size: Minimum size (in pixels) for width and height
    Returns:
        (success, message)
        - success: Boolean indicating if the face size meets the criteria
        - message: String with status message
    """
    if bbox is None:
        return (False, "No bounding box provided")

    x, y, w, h = bbox
    if w > min_size and h > min_size:
        return (True, "The face size passes the specified criteria.")
    else:
        return (False, "The face size does not meet the specified criteria.")