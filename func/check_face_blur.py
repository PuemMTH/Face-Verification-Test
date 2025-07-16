import cv2
def check_face_blur(image_path,threshold):
    # อ่านภาพและแปลงเป็น grayscale
    img = cv2.imread(image_path)
    if img is None:
        return False, "Cannot read image"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # คำนวณ Laplacian และ variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # ตรวจสอบว่าเบลอหรือไม่
    if variance < threshold:
        return False, f"Image is blurry"
    return True, f"Image isn't blurry"