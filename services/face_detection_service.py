import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class FaceDetectionService:
    """
    Servicio para detección de rostros usando Haar Cascades de OpenCV
    (Algoritmo clásico, no pre-entrenado por nosotros pero sí un algoritmo establecido)
    """

    def __init__(self):
        """
        Inicializa el detector de rostros
        """
        # Cargar clasificadores Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

        # Parámetros de detección
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
        self.max_size = (300, 300)

    def detect_faces(self, image: np.ndarray, detect_profile: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostros en una imagen

        Args:
            image: Imagen en formato numpy array
            detect_profile: Si detectar también rostros de perfil

        Returns:
            Lista de tuplas (x, y, w, h) con las coordenadas de los rostros detectados
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detectar rostros frontales
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            maxSize=self.max_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_list = list(faces)

        # Detectar rostros de perfil si está habilitado
        if detect_profile:
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                maxSize=self.max_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Añadir rostros de perfil que no se solapen con frontales
            for (px, py, pw, ph) in profile_faces:
                is_overlap = False
                for (fx, fy, fw, fh) in face_list:
                    # Verificar solapamiento
                    if self._calculate_overlap((px, py, pw, ph), (fx, fy, fw, fh)) > 0.3:
                        is_overlap = True
                        break

                if not is_overlap:
                    face_list.append((px, py, pw, ph))

        return face_list

    def extract_face_roi(self, image: np.ndarray, face_coords: Tuple[int, int, int, int],
                         margin: float = 0.2) -> np.ndarray:
        """
        Extrae la región de interés (ROI) de un rostro

        Args:
            image: Imagen original
            face_coords: Coordenadas del rostro (x, y, w, h)
            margin: Margen adicional alrededor del rostro (porcentaje)

        Returns:
            Imagen recortada del rostro
        """
        x, y, w, h = face_coords

        # Calcular margen
        margin_x = int(w * margin)
        margin_y = int(h * margin)

        # Coordenadas expandidas
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)

        # Extraer ROI
        face_roi = image[y1:y2, x1:x2]

        return face_roi

    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Obtiene el rostro más grande de una lista de rostros detectados

        Args:
            faces: Lista de rostros detectados

        Returns:
            Coordenadas del rostro más grande o None si la lista está vacía
        """
        if not faces:
            return None

        largest_face = max(faces, key=lambda face: face[2] * face[3])  # w * h
        return largest_face

    def filter_faces_by_quality(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                                min_quality_score: float = 50.0) -> List[Tuple[int, int, int, int, float]]:
        """
        Filtra rostros por calidad y retorna con puntuación

        Args:
            image: Imagen original
            faces: Lista de rostros detectados
            min_quality_score: Puntuación mínima de calidad

        Returns:
            Lista de tuplas (x, y, w, h, quality_score) filtradas por calidad
        """
        quality_faces = []

        for face_coords in faces:
            # Extraer ROI del rostro
            face_roi = self.extract_face_roi(image, face_coords)

            # Calcular puntuación de calidad
            quality_score = self._calculate_face_quality(face_roi)

            if quality_score >= min_quality_score:
                quality_faces.append((*face_coords, quality_score))

        # Ordenar por calidad descendente
        quality_faces.sort(key=lambda x: x[4], reverse=True)

        return quality_faces

    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Dibuja rectángulos alrededor de los rostros detectados

        Args:
            image: Imagen original
            faces: Lista de rostros detectados
            color: Color del rectángulo (B, G, R)
            thickness: Grosor de la línea

        Returns:
            Imagen con rectángulos dibujados
        """
        result_image = image.copy()

        for i, (x, y, w, h) in enumerate(faces):
            # Dibujar rectángulo
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)

            # Añadir etiqueta
            label = f"Face {i + 1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(result_image, (x, y - label_size[1] - 10),
                          (x + label_size[0], y), color, -1)
            cv2.putText(result_image, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result_image

    def _calculate_overlap(self, face1: Tuple[int, int, int, int],
                           face2: Tuple[int, int, int, int]) -> float:
        """
        Calcula el solapamiento entre dos rostros detectados
        """
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2

        # Coordenadas de intersección
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        # Área de intersección
        intersection_area = (xi2 - xi1) * (yi2 - yi1)

        # Área de unión
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        # IoU (Intersection over Union)
        return intersection_area / union_area if union_area > 0 else 0.0

    def _calculate_face_quality(self, face_roi: np.ndarray) -> float:
        """
        Calcula una puntuación de calidad para un rostro

        Args:
            face_roi: Región de interés del rostro

        Returns:
            Puntuación de calidad (0-100)
        """
        if face_roi.size == 0:
            return 0.0

        # Convertir a escala de grises si es necesario
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi

        # Factores de calidad
        quality_scores = []

        # 1. Tamaño del rostro (rostros más grandes = mejor calidad)
        size_score = min(100, (gray_face.shape[0] * gray_face.shape[1]) / 100)
        quality_scores.append(size_score)

        # 2. Nitidez (usando varianza de Laplaciano)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 10)
        quality_scores.append(sharpness_score)

        # 3. Contraste
        contrast = gray_face.std()
        contrast_score = min(100, contrast * 2)
        quality_scores.append(contrast_score)

        # 4. Brillo (evitar imágenes muy oscuras o muy claras)
        brightness = gray_face.mean()
        brightness_score = 100 - abs(128 - brightness) * 2
        brightness_score = max(0, brightness_score)
        quality_scores.append(brightness_score)

        # Puntuación final (promedio ponderado)
        weights = [0.2, 0.4, 0.2, 0.2]  # Más peso a la nitidez
        final_score = sum(score * weight for score, weight in zip(quality_scores, weights))

        return max(0, min(100, final_score))

    def enhance_face_image(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Mejora la calidad de la imagen del rostro

        Args:
            face_roi: Región de interés del rostro

        Returns:
            Imagen del rostro mejorada
        """
        enhanced = face_roi.copy()

        # Convertir a escala de grises si es necesario
        if len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro de reducción de ruido
        enhanced = cv2.medianBlur(enhanced, 3)

        # Ecualización adaptiva de histograma
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)

        # Aplicar filtro de nitidez
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced

    def batch_detect_faces(self, images: List[np.ndarray]) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detecta rostros en un lote de imágenes

        Args:
            images: Lista de imágenes

        Returns:
            Lista de listas con rostros detectados para cada imagen
        """
        all_faces = []

        for i, image in enumerate(images):
            faces = self.detect_faces(image)
            all_faces.append(faces)

            if (i + 1) % 10 == 0:
                print(f"Procesadas {i + 1}/{len(images)} imágenes para detección")

        return all_faces