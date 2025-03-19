import cv2
import numpy as np
from deepface import DeepFace
from src.antispoofing import anti_spoofing

MODEL_NAME = "Facenet"
THRESHOLD = 0.65
TARGET_SIZE = (160, 160)

def process_frame(frame, registered_embeddings):
    try:
        faces = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False, align=False)
        frame_h, frame_w = frame.shape[:2]
        results = []
        for face in faces:
            facial_area = face.get("facial_area")
            if not facial_area:
                continue
            x = max(0, facial_area.get("x", 0))
            y = max(0, facial_area.get("y", 0))
            w = max(0, facial_area.get("w", 0))
            h = max(0, facial_area.get("h", 0))
            if w <= 10 or h <= 10 or w >= 0.8 * frame_w or h >= 0.8 * frame_h:
                continue
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            face_img_resized = cv2.resize(face_img, TARGET_SIZE)
            # Caso o anti-spoofing esteja ativo, pode causar falsos negativos – ajuste ou comente para teste:
            if not anti_spoofing(face_img_resized):
                continue
            face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
            norm_img = (face_img_rgb - 127.5) / 128.0
            embedding_data = DeepFace.represent(img_path=norm_img, model_name=MODEL_NAME, enforce_detection=False, detector_backend='skip')
            recognized_name = "Desconhecido"
            if embedding_data:
                current_embedding = np.array(embedding_data[0]["embedding"])
                current_embedding /= np.linalg.norm(current_embedding)
                best_similarity = -1
                for user in registered_embeddings:
                    user_embedding = user['embedding'] / np.linalg.norm(user['embedding'])
                    similarity = np.dot(current_embedding, user_embedding)
                    if similarity > THRESHOLD and similarity > best_similarity:
                        best_similarity = similarity
                        recognized_name = user['name']
            results.append({"box": (x, y, w, h), "recognized": recognized_name})
        return results
    except Exception as e:
        print(f"Erro crítico: {str(e)[:100]}")
        return []
