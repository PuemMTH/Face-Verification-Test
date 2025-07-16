import cv2
import numpy as np
import mediapipe as mp

def check_lightpol(
    image_path: str, 
    dark_threshold, 
    bright_threshold, 
    diff_threshold,
    margin  # ตัดขอบหน้า 10%
) -> tuple[bool, str]:
    image = cv2.imread(image_path)
    if image is None:
        return False, "invalid_image"

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        result = detector.process(rgb_image)

    if not result.detections:
        return False, "no_face"

    h, w, _ = image.shape
    bbox = result.detections[0].location_data.relative_bounding_box

    # แปลงเป็นพิกัด pixel
    x_min = int(bbox.xmin * w)
    y_min = int(bbox.ymin * h)
    box_width = int(bbox.width * w)
    box_height = int(bbox.height * h)

    # ตัดขอบ (เฉพาะส่วนกลางใบหน้า)
    x_start = max(0, int(x_min + box_width * margin))
    y_start = max(0, int(y_min + box_height * margin))
    x_end = min(w, int(x_min + box_width * (1 - margin)))
    y_end = min(h, int(y_min + box_height * (1 - margin)))

    # ตรวจสอบขนาดว่ามีข้อมูลหรือไม่
    if x_end <= x_start or y_end <= y_start:
        return False, "invalid_face_crop"

    face_region_v = hsv_image[y_start:y_end, x_start:x_end, 2]
    if face_region_v.size == 0:
        return False, "empty_face_region"

    # Mask สำหรับ background
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[y_min:y_min + box_height, x_min:x_min + box_width] = 0

    background_v = cv2.bitwise_and(hsv_image[:, :, 2], hsv_image[:, :, 2], mask=mask)
    background_values = background_v[background_v > 0]

    face_brightness = float(np.mean(face_region_v))
    background_brightness = float(np.mean(background_values)) if background_values.size > 0 else None
    brightness_diff = abs(face_brightness - background_brightness) if background_brightness is not None else None

    # สถานะตามเกณฑ์
    if face_brightness < dark_threshold:
        status = "too_dark"
    elif face_brightness > bright_threshold:
        status = "too_bright"
    elif brightness_diff is not None and brightness_diff > diff_threshold:
        status = "backlight"
    else:
        status = "normal"

    return (status == "normal"), status
