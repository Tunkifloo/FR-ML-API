import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import os
from typing import List, Tuple, Optional
from datetime import datetime
import json


class EigenfacesService:
    """
    Implementación del algoritmo Eigenfaces para reconocimiento facial
    Sin usar modelos pre-entrenados, implementado desde cero
    """

    def __init__(self, n_components: int = 150, image_size: Tuple[int, int] = (100, 100)):
        """
        Inicializa el servicio Eigenfaces

        Args:
            n_components: Número de componentes principales a mantener
            image_size: Tamaño estándar de las imágenes (ancho, alto)
        """
        self.n_components = n_components
        self.image_size = image_size
        self.pca = None
        self.scaler = StandardScaler()

        # Datos de entrenamiento
        self.trained_embeddings = []
        self.trained_labels = []
        self.mean_face = None
        self.eigenfaces = None
        self.is_trained = False

        # Configuración
        self.threshold_distance = 2500  # Umbral para considerar una coincidencia
        self.model_path = "storage/models/eigenfaces_model.pkl"

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        CORREGIDO: Preprocesa una imagen para el algoritmo Eigenfaces
        """
        print(f"🔧 Eigenfaces preprocess input: {image.shape}, dtype: {image.dtype}")

        # Si ya viene procesada (float64 en [0,1]), solo verificar dimensiones
        if image.dtype == np.float64 and image.max() <= 1.0 and len(image.shape) == 2:
            if image.shape == self.image_size:
                print(f"✅ Imagen ya preprocesada correctamente")
                return image

        # Procesar imagen raw
        processed = image.copy()

        # Convertir a escala de grises si es necesario
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            print(f"🔧 Convertida a escala de grises: {processed.shape}")

        # CRÍTICO: Redimensionar SIEMPRE al tamaño específico
        if processed.shape != self.image_size:
            processed = cv2.resize(processed, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            print(f"🔧 Redimensionada a: {processed.shape}")

        # Aplicar filtro gaussiano para reducir ruido
        processed = cv2.GaussianBlur(processed, (5, 5), 0)

        # Ecualización de histograma para mejorar contraste
        processed = cv2.equalizeHist(processed)

        # Normalizar valores de pixel a [0, 1]
        processed = processed.astype(np.float64) / 255.0

        print(
            f"✅ Eigenfaces preprocessing completado: {processed.shape}, dtype={processed.dtype}, range=[{processed.min():.3f}, {processed.max():.3f}]")

        return processed

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        CORREGIDO: Extrae características con validación robusta
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Ejecute train() primero.")

        print(f"🔍 Eigenfaces extract_features input: {image.shape}, dtype: {image.dtype}")

        # VALIDACIÓN Y NORMALIZACIÓN ROBUSTA
        processed_image = image.copy()

        # Si la imagen viene como float64 en [0,1] (ya procesada), usarla directamente
        if processed_image.dtype == np.float64 and processed_image.max() <= 1.0:
            print(
                f"✅ Imagen ya preprocesada para Eigenfaces: {processed_image.dtype}, range=[{processed_image.min():.3f}, {processed_image.max():.3f}]")
        else:
            # Convertir y normalizar
            if len(processed_image.shape) == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                print(f"🔧 Convertida a escala de grises: {processed_image.shape}")

            # Redimensionar si es necesario
            if processed_image.shape != self.image_size:
                processed_image = cv2.resize(processed_image, self.image_size, interpolation=cv2.INTER_LANCZOS4)
                print(f"🔧 Redimensionada: {processed_image.shape}")

            # Normalizar a [0, 1]
            if processed_image.max() > 1.0:
                processed_image = processed_image.astype(np.float64) / 255.0
                print(
                    f"🔧 Normalizada: {processed_image.dtype}, range=[{processed_image.min():.3f}, {processed_image.max():.3f}]")

        # Aplanar la imagen
        image_vector = processed_image.flatten()
        print(f"🔍 Vector shape: {image_vector.shape}")
        print(f"🔍 Mean face shape: {self.mean_face.shape}")

        # VALIDACIÓN CRÍTICA DE DIMENSIONES
        if image_vector.shape[0] != self.mean_face.shape[0]:
            raise ValueError(
                f"ERROR DIMENSIONES: imagen_vector={image_vector.shape[0]}, "
                f"mean_face={self.mean_face.shape[0]}. "
                f"La imagen debe ser exactamente {self.image_size}. "
                f"Imagen recibida: {processed_image.shape}"
            )

        # Centrar la imagen restando la cara promedio
        centered_image = image_vector - self.mean_face
        print(f"🔧 Imagen centrada: shape={centered_image.shape}")

        # Proyectar en el espacio de eigenfaces
        embedding = self.pca.transform(centered_image.reshape(1, -1))
        print(f"✅ Embedding generado: shape={embedding.shape}")

        return embedding.flatten()

    def train(self, images: List[np.ndarray], labels: List[int]) -> None:
        """
        Entrena el modelo Eigenfaces con un conjunto de imágenes

        Args:
            images: Lista de imágenes preprocesadas
            labels: Lista de etiquetas correspondientes a cada imagen
        """
        print(f"🎓 Iniciando entrenamiento Eigenfaces con {len(images)} imágenes...")

        # PASO 1: Preprocesar todas las imágenes PRIMERO
        processed_images = []
        for img in images:
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img.flatten())

        # PASO 2: Convertir a matriz numpy
        X = np.array(processed_images)

        # PASO 3: AHORA ajustar n_components (después de crear X)
        max_components = min(len(images), X.shape[1]) - 1
        actual_components = min(self.n_components, max_components)

        print(f"📊 Ajustando componentes: {self.n_components} → {actual_components}")
        print(f"📈 Datos disponibles: {len(images)} imágenes, {X.shape[1]} características")

        # PASO 4: Inicializar PCA con componentes ajustados
        self.pca = PCA(n_components=actual_components, whiten=True)

        # PASO 5: Calcular la cara promedio
        self.mean_face = np.mean(X, axis=0)

        # PASO 6: Centrar los datos
        X_centered = X - self.mean_face

        # PASO 7: Aplicar PCA
        self.pca.fit(X_centered)

        # PASO 8: Guardar eigenfaces
        self.eigenfaces = self.pca.components_

        # PASO 9: Generar embeddings para todas las imágenes de entrenamiento
        self.trained_embeddings = []
        self.trained_labels = []

        for i, img_vector in enumerate(X_centered):
            embedding = self.pca.transform(img_vector.reshape(1, -1))
            self.trained_embeddings.append(embedding.flatten())
            self.trained_labels.append(labels[i])

        self.is_trained = True

        print(f"✅ Entrenamiento completado. Eigenfaces generados: {len(self.eigenfaces)}")
        print(f"📊 Varianza explicada: {sum(self.pca.explained_variance_ratio_):.2%}")
        print(f"🔢 Embeddings creados: {len(self.trained_embeddings)}")

    def add_new_person(self, images: List[np.ndarray], person_id: int) -> None:
        """
        Añade una nueva persona al modelo sin reentrenar completamente
        (Entrenamiento incremental)

        Args:
            images: Lista de imágenes de la nueva persona
            person_id: ID de la persona
        """
        if not self.is_trained:
            raise ValueError("El modelo debe estar entrenado antes de añadir nuevas personas")

        print(f"➕ Añadiendo nueva persona ID: {person_id} con {len(images)} imágenes")

        # Procesar nuevas imágenes
        for img in images:
            processed_img = self.preprocess_image(img)
            img_vector = processed_img.flatten()

            # Centrar la imagen
            centered_img = img_vector - self.mean_face

            # Generar embedding
            embedding = self.pca.transform(centered_img.reshape(1, -1))

            # Añadir a los datos de entrenamiento
            self.trained_embeddings.append(embedding.flatten())
            self.trained_labels.append(person_id)

        print(f"✅ Persona añadida. Total embeddings: {len(self.trained_embeddings)}")

    def recognize_face(self, image: np.ndarray) -> Tuple[int, float, dict]:
        """
        Reconoce una cara en una imagen

        Args:
            image: Imagen a reconocer

        Returns:
            Tupla con (person_id, confidence, details)
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        # Extraer características de la imagen
        query_embedding = self.extract_features(image)

        # Calcular distancias a todos los embeddings conocidos
        distances = []
        for embedding in self.trained_embeddings:
            dist = np.linalg.norm(query_embedding - embedding)
            distances.append(dist)

        # Encontrar la menor distancia
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        predicted_person_id = self.trained_labels[min_distance_idx]

        # Calcular confianza
        confidence = max(0, 100 - (min_distance / self.threshold_distance * 100))

        # Determinar si es una coincidencia válida
        is_match = min_distance < self.threshold_distance

        details = {
            "distance": float(min_distance),
            "threshold": self.threshold_distance,
            "is_match": is_match,
            "confidence_score": confidence,
            "algorithm": "eigenfaces",
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
            'pca': self.pca,
            'mean_face': self.mean_face,
            'eigenfaces': self.eigenfaces,
            'trained_embeddings': self.trained_embeddings,
            'trained_labels': self.trained_labels,
            'n_components': self.n_components,
            'image_size': self.image_size,
            'threshold_distance': self.threshold_distance,
            'is_trained': self.is_trained
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 Modelo guardado en: {path}")

    def load_model(self, path: str = None) -> None:
        """
        Carga un modelo previamente entrenado
        """
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            print(f"⚠️ No se encontró modelo en: {path}")
            return

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.pca = model_data['pca']
        self.mean_face = model_data['mean_face']
        self.eigenfaces = model_data['eigenfaces']
        self.trained_embeddings = model_data['trained_embeddings']
        self.trained_labels = model_data['trained_labels']
        self.n_components = model_data['n_components']
        self.image_size = model_data['image_size']
        self.threshold_distance = model_data['threshold_distance']
        self.is_trained = model_data['is_trained']

        print(f"📂 Modelo cargado desde: {path}")
        print(f"📊 Embeddings cargados: {len(self.trained_embeddings)}")

    def get_model_info(self) -> dict:
        """
        Obtiene información del modelo actual
        """
        return {
            "algorithm": "eigenfaces",
            "is_trained": self.is_trained,
            "n_components": self.n_components,
            "image_size": self.image_size,
            "total_embeddings": len(self.trained_embeddings) if self.is_trained else 0,
            "unique_persons": len(set(self.trained_labels)) if self.is_trained else 0,
            "threshold_distance": self.threshold_distance,
            "variance_explained": sum(self.pca.explained_variance_ratio_) if self.is_trained else 0
        }

    def visualize_eigenfaces(self, n_faces: int = 20) -> List[np.ndarray]:
        """
        Retorna las primeras n eigenfaces para visualización
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        eigenfaces_images = []
        for i in range(min(n_faces, len(self.eigenfaces))):
            # Reshape eigenface to image format
            eigenface = self.eigenfaces[i].reshape(self.image_size)

            # Normalize for visualization
            eigenface = ((eigenface - eigenface.min()) / (eigenface.max() - eigenface.min()) * 255).astype(np.uint8)

            eigenfaces_images.append(eigenface)

        return eigenfaces_images