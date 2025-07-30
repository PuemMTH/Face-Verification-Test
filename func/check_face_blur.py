import cv2
import numpy as np
import mediapipe as mp


def _patch_from_contour(img, contour):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)

    xmin = max(0, int(np.min(contour[:, 0])))
    xmax = min(img.shape[1], int(np.max(contour[:, 0])))
    ymin = max(0, int(np.min(contour[:, 1])))
    ymax = min(img.shape[0], int(np.max(contour[:, 1])))

    if xmax <= xmin or ymax <= ymin:
        return None, (0, 0)

    out = img.copy()
    out[mask == 0] = (255, 255, 255)

    cropped_img = out[ymin:ymax, xmin:xmax]
    if cropped_img.size == 0:
        return None, (0, 0)
    if len(cropped_img.shape) == 3:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    return cropped_img, (xmin, ymin)


def check_face_blur(image, threshold):
    """
    ตรวจสอบว่าบริเวณใบหน้าในภาพเบลอหรือไม่ โดยใช้ MediaPipe และ Laplacian variance
    
    Parameters
    ----------
    image : str หรือ numpy.ndarray
        path ของภาพ หรืออาเรย์ภาพแบบ BGR
    threshold : float
        ค่า threshold ที่ใช้ตัดสินความเบลอ
    
    Returns
    -------
    variance : float หรือ None
        ค่าความแปรปรวนของ Laplacian หรือ None ถ้ามีปัญหา
    message : str
        ข้อความแจ้งผลลัพธ์
    """
    if threshold <= 0:
        return None, "Threshold must be positive"

    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            return None, "Cannot read image"
    else:
        img = image

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ย้ายการสร้าง face_detection เข้าในฟังก์ชัน
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_rgb)

        if not results.detections:
            return None, "No face detected"

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = img.shape

        xmin = int(bbox.xmin * w)
        ymin = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        contour = np.array([
            [xmin, ymin],
            [xmin + width, ymin],
            [xmin + width, ymin + height],
            [xmin, ymin + height]
        ], dtype=np.int32)

        face_img, _ = _patch_from_contour(img, contour)
        if face_img is None:
            return None, "Invalid face region"

        variance = cv2.Laplacian(face_img, cv2.CV_64F).var()

        if variance < threshold:
            return False, f"Image is blurry"
        return True, f"Image isn't blurry"