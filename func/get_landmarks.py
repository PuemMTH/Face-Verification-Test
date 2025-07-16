import mediapipe as mp
import cv2
import numpy as np

def get_lm(img_path):
    """
    Detects face landmarks using MediaPipe Face Mesh, extracts landmarks and bounding box.
    Returns: (success, message, landmarks, bbox, num_landmarks)
    - success: Boolean indicating if detection was successful
    - message: String with status or error message
    - landmarks: List of landmark coordinates [(x, y, z), ...] or None
    - bbox: Tuple of (x, y, w, h) or None
    - num_landmarks: Integer indicating the number of landmarks detected
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    try:
        # Read the image
        image = cv2.imread(img_path)
        if image is None:
            return (False, "Failed to load image", None, None, 0)

        # Convert to RGB as MediaPipe expects RGB images
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # Initialize face mesh model
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5,
            refine_landmarks=True
        ) as face_mesh:
            # Process the image
            results = face_mesh.process(image_rgb)

            # Check if faces are detected
            if not results.multi_face_landmarks:
                return (False, "No faces detected", None, None)

            # Get the first detected face
            face_landmarks = results.multi_face_landmarks[0]

            # Extract landmarks
            landmarks = []
            for landmark in face_landmarks.landmark:
                # Convert relative coordinates to pixel coordinates
                landmark_x = int(landmark.x * width)
                landmark_y = int(landmark.y * height)
                landmark_z = landmark.z  # Keep z in relative units (depth)
                landmarks.append((landmark_x, landmark_y, landmark_z))

            # Calculate bounding box from landmarks with margin
            x_coords = [lm[0] for lm in landmarks]
            y_coords = [lm[1] for lm in landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            w = x_max - x_min
            h = y_max - y_min

            # Add margin (10% of width/height) to ensure bbox covers the entire face
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            x_min = max(0, x_min - margin_x)
            y_min = max(0, y_min - margin_y)
            x_max = min(width, x_max + margin_x)
            y_max = min(height, y_max + margin_y)
            w = x_max - x_min
            h = y_max - y_min
            bbox = (x_min, y_min, w, h)

            return (True, "Face detected successfully", landmarks, bbox)

    except Exception as e:
        return (False, f"Error during face detection: {str(e)}", None,None)