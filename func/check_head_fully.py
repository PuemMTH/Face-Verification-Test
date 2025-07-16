import cv2
import mediapipe as mp

def is_top_of_head_cut(landmarks, image_height):
    top_y = landmarks[10].y * image_height
    return top_y < 10

def is_chin_cut(landmarks, image_height):
    chin_y = landmarks[152].y * image_height
    return chin_y > image_height - 10

def analyze_single_image(image_path):
    mp_face_mesh = mp.solutions.face_mesh

    image = cv2.imread(image_path)
    if image is None:
        return False, "Failed to read image"

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        top_cut = is_top_of_head_cut(face_landmarks, h)
        chin_cut = is_chin_cut(face_landmarks, h)

        if top_cut and chin_cut:
            return False, "Top of head and chin might be cut"
        elif top_cut:
            return False, "Top of head might be cut"
        elif chin_cut:
            return False, "Chin might be cut"
        else:
            return True, "Head is fully visible"
    else:
        return False, "No face detected"
