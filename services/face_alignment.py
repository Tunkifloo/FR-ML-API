import cv2
import numpy as np
from typing import Optional, Tuple
import mediapipe as mp


class FaceAlignmentService:
    """
    Alineación facial robusta usando MediaPipe Face Mesh
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("✅ MediaPipe Face Mesh inicializado")

    def align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Alinea rostro basado en landmarks de MediaPipe
        """
        try:
            # Convertir BGR a RGB si es necesario
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Procesar imagen
            results = self.face_mesh.process(image_rgb)

            if not results.multi_face_landmarks:
                print("⚠️ No se detectaron landmarks faciales")
                return None

            # Obtener landmarks del primer rostro
            face_landmarks = results.multi_face_landmarks[0]

            # Extraer puntos de ojos
            # Índices de MediaPipe: ojo izquierdo 33, ojo derecho 263
            h, w = image.shape[:2]

            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

            # Calcular ángulo de rotación
            dy = right_eye_coords[1] - left_eye_coords[1]
            dx = right_eye_coords[0] - left_eye_coords[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # Centro entre ojos
            eyes_center = (
                (left_eye_coords[0] + right_eye_coords[0]) // 2,
                (left_eye_coords[1] + right_eye_coords[1]) // 2
            )

            # Matriz de rotación
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

            # Aplicar rotación
            aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

            print(f"✅ Rostro alineado (ángulo: {angle:.2f}°)")
            return aligned

        except Exception as e:
            print(f"❌ Error en alineación facial: {e}")
            return None

    def __del__(self):
        """Liberar recursos"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()