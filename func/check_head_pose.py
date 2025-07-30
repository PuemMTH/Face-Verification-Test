import cv2
import mediapipe as mp
import numpy as np
import os
import yaml

# โหลดค่า config จากไฟล์ yml
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)



def check_head_pose(image_path):
    # ตั้งค่า MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    # อ่านภาพจาก path
    if not os.path.exists(image_path):
        return "Error: Image path does not exist"
    
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Cannot read image"

    # เตรียมภาพ: แปลงเป็น RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    # เก็บขนาดภาพ
    img_h, img_w, img_c = image.shape

    # ตัวแปรเก็บ landmarks
    face_2d = []
    face_3d = []

    # หากเจอใบหน้า
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:  # จุดสำคัญ: ตา, จมูก, ปาก, คาง
                    if idx == 1:  # จมูก
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            # แปลงเป็น array
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # ตั้งค่า focal length และ camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1]
            ])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # คำนวณการหมุนและการเคลื่อนที่
            success, rot_vec, tran_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            if not success:
                return "Error: solvePnP failed"

            # แปลงเวกเตอร์การหมุนเป็นเมทริกซ์
            rmat, _ = cv2.Rodrigues(rot_vec)

            # คำนวณมุม Pitch, Yaw, Roll
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch = angles[0] * 360
            yaw = angles[1] * 360
            roll = angles[2] * 360

            

            # ตรวจสอบทิศทางศีรษะ
            if yaw < config['threshold']['left_th']:
                success = False
                direction = "Looking Left"
            elif yaw > config['threshold']['right_th']:
                success = False
                direction = "Looking Right"
            elif pitch < config['threshold']['down_th']:
                success = False
                direction = "Looking Down"
            elif pitch > config['threshold']['up_th']:
                success = False
                direction = "Looking Up"
            elif roll < config['threshold']['til_left_th']:
                success = False
                direction = "Tilting Left"
            elif roll > config['threshold']['til_right_th']:
                success = False
                direction = "Tilting Right"
            else:
                success = True
                direction = "Forward"

            # สร้างข้อความผลลัพธ์
            result = (success,direction)
            face_mesh.close()
            return result

    face_mesh.close()
    return "Error: No face detected"