import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from skimage.feature import local_binary_pattern
import pickle
import os
from typing import List, Tuple, Optional
from datetime import datetime


class LBPService:
    """
    Implementación del algoritmo Local Binary Patterns (LBP) para reconocimiento facial
    Implementado desde cero sin modelos pre-entrenados
    """

    def __init__(self, radius: int = 2, n_points: int = 16, grid_size: Tuple[int, int] = (8, 8)):
        """
        Inicializa el servicio LBP

        Args:
            radius: Radio del patrón circular LBP
            n_points: Número de puntos en el patrón circular
            grid_size: Tamaño de la grilla para dividir la imagen (filas, columnas)
        """
        self.radius = radius
        self.n_points = n_points
        self.grid_size = grid_size
        self.method = 'uniform'  # Usar patrones uniformes para reducir dimensionalidad

        # Datos de entrenamiento
        self.trained_histograms = []
        self.trained_labels = []
        self.is_trained = False

        # Configuración
        self.threshold_similarity = 0.7  # Umbral de similitud para considerar coincidencia
        self.model_path = "storage/models/lbp_model.pkl"
        self.image_size = (100, 100)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        CORREGIDO: Preprocesa imagen para LBP con validación robusta
        """
        print(f"🔧 LBP preprocess input: {image.shape}, dtype: {image.dtype}")

        processed = image.copy()

        # Manejar diferentes tipos de entrada
        if len(processed.shape) == 1:
            # Vector 1D - reformatear
            target_pixels = self.image_size[0] * self.image_size[1]
            if processed.shape[0] == target_pixels:
                processed = processed.reshape(self.image_size)
                print(f"🔧 Vector 1D reformateado: {processed.shape}")
            else:
                size = int(np.sqrt(processed.shape[0]))
                if size * size == processed.shape[0]:
                    processed = processed.reshape(size, size)
                    print(f"🔧 Vector 1D a cuadrado: {processed.shape}")
                else:
                    raise ValueError(f"Vector 1D no compatible: {processed.shape}")

        elif len(processed.shape) == 3:
            # Imagen color
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            print(f"🔧 Convertida a escala de grises: {processed.shape}")

        # Asegurar que sea 2D
        if len(processed.shape) != 2:
            raise ValueError(f"Error: imagen debe ser 2D, recibida: {processed.shape}")

        # Redimensionar a tamaño estándar
        if processed.shape != self.image_size:
            processed = cv2.resize(processed, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            print(f"🔧 Redimensionada: {processed.shape}")

        # Aplicar filtro gaussiano
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # Ecualización adaptiva para LBP
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

        # Asegurar tipo uint8 para LBP
        if processed.dtype != np.uint8:
            if processed.max() <= 1.0:
                processed = (processed * 255).astype(np.uint8)
            else:
                processed = processed.astype(np.uint8)

        print(f"✅ LBP preprocessing completado: {processed.shape}, dtype={processed.dtype}")
        return processed

    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        CORREGIDO: Extrae características LBP con manejo robusto de dimensiones
        """
        print(f"🔍 LBP extract_lbp_features input: {image.shape}, dtype: {image.dtype}")

        # CRÍTICO: Asegurar que la imagen sea 2D
        processed_image = image.copy()

        if len(processed_image.shape) == 1:
            # Vector 1D - intentar reformatear
            target_pixels = self.image_size[0] * self.image_size[1]

            if processed_image.shape[0] == target_pixels:
                processed_image = processed_image.reshape(self.image_size)
                print(f"🔧 Vector 1D reformateado a image_size: {processed_image.shape}")
            else:
                # Intentar cuadrado perfecto
                size = int(np.sqrt(processed_image.shape[0]))
                if size * size == processed_image.shape[0]:
                    processed_image = processed_image.reshape(size, size)
                    print(f"🔧 Vector 1D reformateado a cuadrado: {processed_image.shape}")
                else:
                    raise ValueError(
                        f"Vector 1D {processed_image.shape} no compatible con LBP. Esperado: {target_pixels} píxeles")

        elif len(processed_image.shape) == 3:
            # Imagen color - convertir a escala de grises
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            print(f"🔧 Convertida a escala de grises: {processed_image.shape}")

        # Verificar que ahora sea 2D
        if len(processed_image.shape) != 2:
            raise ValueError(f"LBP requiere imagen 2D, recibida: {processed_image.shape}")

        print(f"✅ Imagen 2D confirmada: {processed_image.shape}")

        # Aplicar preprocesamiento específico de LBP si no está ya procesada
        if not (processed_image.dtype == np.uint8 and processed_image.shape == self.image_size):
            processed_image = self.preprocess_image(processed_image)
            print(f"🔧 Preprocesamiento LBP aplicado: {processed_image.shape}")

        # Calcular LBP
        try:
            lbp_image = local_binary_pattern(
                processed_image,
                self.n_points,
                self.radius,
                method=self.method
            )
            print(f"✅ LBP calculado: shape={lbp_image.shape}")
        except Exception as e:
            print(f"❌ Error calculando LBP: {e}")
            print(f"   Parámetros: radius={self.radius}, n_points={self.n_points}, method={self.method}")
            print(f"   Imagen: shape={processed_image.shape}, dtype={processed_image.dtype}")
            raise

        # Dividir en grilla y calcular histogramas
        height, width = lbp_image.shape
        cell_height = height // self.grid_size[0]
        cell_width = width // self.grid_size[1]

        histograms = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                start_row = i * cell_height
                end_row = (i + 1) * cell_height
                start_col = j * cell_width
                end_col = (j + 1) * cell_width

                cell = lbp_image[start_row:end_row, start_col:end_col]

                # Calcular histograma
                n_bins = self.n_points + 2
                hist, _ = np.histogram(
                    cell.ravel(),
                    bins=n_bins,
                    range=(0, n_bins),
                    density=True
                )
                histograms.append(hist)

        # Concatenar todos los histogramas
        feature_vector = np.concatenate(histograms)
        print(f"✅ LBP features generados: shape={feature_vector.shape}")

        return feature_vector

    def train(self, images: List[np.ndarray], labels: List[int]) -> None:
        """
        Entrena el modelo LBP con un conjunto de imágenes

        Args:
            images: Lista de imágenes
            labels: Lista de etiquetas correspondientes
        """
        print(f"🎓 Iniciando entrenamiento LBP con {len(images)} imágenes...")

        self.trained_histograms = []
        self.trained_labels = []

        for i, (img, label) in enumerate(zip(images, labels)):
            # Preprocesar imagen
            processed_img = self.preprocess_image(img)

            # Extraer características LBP
            lbp_features = self.extract_lbp_features(processed_img)

            # Almacenar
            self.trained_histograms.append(lbp_features)
            self.trained_labels.append(label)

            if (i + 1) % 10 == 0:
                print(f"Procesadas {i + 1}/{len(images)} imágenes...")

        self.is_trained = True
        print(f"✅ Entrenamiento LBP completado. Características extraídas: {len(self.trained_histograms)}")

    def add_new_person(self, images: List[np.ndarray], person_id: int) -> None:
        """
        Añade una nueva persona al modelo (entrenamiento incremental)

        Args:
            images: Lista de imágenes de la nueva persona
            person_id: ID de la persona
        """
        print(f"➕ Añadiendo nueva persona ID: {person_id} con {len(images)} imágenes")

        for img in images:
            # Preprocesar imagen
            processed_img = self.preprocess_image(img)

            # Extraer características LBP
            lbp_features = self.extract_lbp_features(processed_img)

            # Añadir a los datos de entrenamiento
            self.trained_histograms.append(lbp_features)
            self.trained_labels.append(person_id)

        print(f"✅ Persona añadida. Total características: {len(self.trained_histograms)}")

    def recognize_face(self, image: np.ndarray) -> Tuple[int, float, dict]:
        """
        Reconoce una cara usando LBP

        Args:
            image: Imagen a reconocer

        Returns:
            Tupla con (person_id, confidence, details)
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        # Extraer características de la imagen consultada
        processed_img = self.preprocess_image(image)
        query_features = self.extract_lbp_features(processed_img)

        # Calcular similitudes con todas las características almacenadas
        similarities = []
        distances = []

        for stored_features in self.trained_histograms:
            # Similitud coseno
            cos_sim = cosine_similarity([query_features], [stored_features])[0][0]
            similarities.append(cos_sim)

            # Distancia euclidiana
            eucl_dist = np.linalg.norm(query_features - stored_features)
            distances.append(eucl_dist)

        # Encontrar la mejor coincidencia
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        best_distance = distances[best_match_idx]
        predicted_person_id = self.trained_labels[best_match_idx]

        # Calcular confianza
        confidence = best_similarity * 100

        # Determinar si es una coincidencia válida
        is_match = best_similarity >= self.threshold_similarity

        details = {
            "similarity": float(best_similarity),
            "distance": float(best_distance),
            "threshold": self.threshold_similarity,
            "is_match": is_match,
            "confidence_score": confidence,
            "algorithm": "lbp",
            "grid_size": self.grid_size,
            "lbp_params": {
                "radius": self.radius,
                "n_points": self.n_points,
                "method": self.method
            },
            "timestamp": datetime.now().isoformat()
        }

        return predicted_person_id if is_match else -1, confidence, details

    def save_model(self, path: str = None) -> None:
        """
        Guarda el modelo entrenado
        """
        if path is None:
            path = self.model_path

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'trained_histograms': self.trained_histograms,
            'trained_labels': self.trained_labels,
            'radius': self.radius,
            'n_points': self.n_points,
            'grid_size': self.grid_size,
            'method': self.method,
            'threshold_similarity': self.threshold_similarity,
            'image_size': self.image_size,
            'is_trained': self.is_trained
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 Modelo LBP guardado en: {path}")

    def load_model(self, path: str = None) -> None:
        """
        Carga un modelo previamente entrenado
        """
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            print(f"⚠️ No se encontró modelo LBP en: {path}")
            return

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.trained_histograms = model_data['trained_histograms']
        self.trained_labels = model_data['trained_labels']
        self.radius = model_data['radius']
        self.n_points = model_data['n_points']
        self.grid_size = model_data['grid_size']
        self.method = model_data['method']
        self.threshold_similarity = model_data['threshold_similarity']
        self.image_size = model_data['image_size']
        self.is_trained = model_data['is_trained']

        print(f"📂 Modelo LBP cargado desde: {path}")
        print(f"📊 Características cargadas: {len(self.trained_histograms)}")

    def get_model_info(self) -> dict:
        """
        Obtiene información del modelo actual
        """
        return {
            "algorithm": "lbp",
            "is_trained": self.is_trained,
            "radius": self.radius,
            "n_points": self.n_points,
            "grid_size": self.grid_size,
            "method": self.method,
            "image_size": self.image_size,
            "total_features": len(self.trained_histograms) if self.is_trained else 0,
            "unique_persons": len(set(self.trained_labels)) if self.is_trained else 0,
            "threshold_similarity": self.threshold_similarity,
            "feature_vector_size": len(self.trained_histograms[0]) if self.trained_histograms else 0
        }

    def calculate_lbp_histogram_comparison(self, hist1: np.ndarray, hist2: np.ndarray) -> dict:
        """
        Compara dos histogramas LBP usando diferentes métricas
        """
        # Correlación
        correlation = np.corrcoef(hist1, hist2)[0, 1]

        # Chi-cuadrado
        chi_squared = cv2.compareHist(
            hist1.astype(np.float32),
            hist2.astype(np.float32),
            cv2.HISTCMP_CHISQR
        )

        # Intersección
        intersection = cv2.compareHist(
            hist1.astype(np.float32),
            hist2.astype(np.float32),
            cv2.HISTCMP_INTERSECT
        )

        # Bhattacharyya
        bhattacharyya = cv2.compareHist(
            hist1.astype(np.float32),
            hist2.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )

        return {
            "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "chi_squared": float(chi_squared),
            "intersection": float(intersection),
            "bhattacharyya": float(bhattacharyya)
        }