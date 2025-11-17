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
    ✅ CORREGIDO: Implementación del algoritmo Eigenfaces con manejo de valores Infinity
    """

    def __init__(self, n_components: int = 150, image_size: Tuple[int, int] = (100, 100)):
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
        self.threshold_distance = 2500
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

    def _clean_features_for_storage(self, features: np.ndarray) -> np.ndarray:
        """
        ✅ NUEVO: Limpia características eliminando valores Infinity y NaN
        """
        cleaned = features.copy()

        # Reemplazar Infinity con valores máximos/mínimos válidos
        cleaned[np.isposinf(cleaned)] = 1e6  # Infinity positivo → valor grande
        cleaned[np.isneginf(cleaned)] = -1e6  # Infinity negativo → valor grande negativo
        cleaned[np.isnan(cleaned)] = 0.0  # NaN → 0

        # Clipear a rango razonable
        cleaned = np.clip(cleaned, -1e6, 1e6)

        print(f"🧹 Características limpiadas: shape={cleaned.shape}, range=[{cleaned.min():.3f}, {cleaned.max():.3f}]")

        return cleaned

    def transform_image_vector(self, image_vector: np.ndarray) -> np.ndarray:
        """
        Toma un vector de imagen aplanado (ej. 10000) y lo proyecta
        al espacio de Eigenfaces (ej. 150) usando el PCA entrenado.

        Args:
            image_vector: Vector plano de la imagen (1D array).

        Returns:
            El vector de características reducido (embedding).
        """
        if not self.is_trained or self.pca is None:
            raise ValueError("Error de PCA: El modelo Eigenfaces no ha sido entrenado. No se puede transformar.")

        if image_vector.shape[0] != self.mean_face.shape[0]:
            raise ValueError(
                f"ERROR DIMENSIONES: El vector de entrada ({image_vector.shape[0]}) no coincide "
                f"con la dimensión del modelo ({self.mean_face.shape[0]})."
            )

        # 1. Centrar la imagen restando la cara promedio
        centered_vector = image_vector - self.mean_face

        # 2. Proyectar en el espacio PCA (transform)
        # Reshape (1, -1) es necesario porque transform espera un lote de imágenes
        embedding = self.pca.transform(centered_vector.reshape(1, -1))

        # 3. Retornar el vector 1D
        return embedding.flatten()

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        ✅ CORREGIDO: Extrae características con limpieza de valores infinitos
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

        # ✅ CRÍTICO: Limpiar valores infinitos
        clean_embedding = self._clean_features_for_storage(embedding.flatten())

        return clean_embedding

    def train(self, images: List[np.ndarray], labels: List[int]) -> None:
        """
        ✅ CORREGIDO: Entrena el modelo Eigenfaces con validación de estabilidad numérica
        """
        print(f"🎓 Iniciando entrenamiento Eigenfaces con {len(images)} imágenes...")

        # PASO 1: Preprocesar todas las imágenes PRIMERO
        processed_images = []
        for img in images:
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img.flatten())

        # PASO 2: Convertir a matriz numpy
        X = np.array(processed_images)

        # PASO 3: Validar datos de entrada
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("⚠️ Datos de entrada contienen NaN o Infinity, limpiando...")
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        # PASO 4: Ajustar n_components para evitar singularidad
        max_components = min(len(images) - 1, X.shape[1] - 1)
        actual_components = min(self.n_components, max_components)

        # ✅ AÑADIR REGULARIZACIÓN PARA ESTABILIDAD
        if actual_components <= 1:
            actual_components = 1
            print("⚠️ Muy pocas imágenes, usando 1 componente")

        print(f"📊 Ajustando componentes: {self.n_components} → {actual_components}")
        print(f"📈 Datos disponibles: {len(images)} imágenes, {X.shape[1]} características")

        # PASO 5: Inicializar PCA con componentes ajustados y estabilidad numérica
        self.pca = PCA(n_components=actual_components, whiten=True, svd_solver='auto')

        # PASO 6: Calcular la cara promedio
        self.mean_face = np.mean(X, axis=0)

        # PASO 7: Centrar los datos
        X_centered = X - self.mean_face

        # PASO 8: Aplicar PCA con validación
        try:
            self.pca.fit(X_centered)
        except Exception as e:
            print(f"⚠️ Error en PCA, aplicando regularización: {e}")
            # Añadir pequeña regularización en caso de singularidad
            regularization = np.eye(X_centered.shape[1]) * 1e-6
            X_reg = X_centered + regularization[:X_centered.shape[0], :]
            self.pca.fit(X_reg)

        # PASO 9: Guardar eigenfaces
        self.eigenfaces = self.pca.components_

        # PASO 10: Generar embeddings con limpieza
        self.trained_embeddings = []
        self.trained_labels = []

        for i, img_vector in enumerate(X_centered):
            try:
                embedding = self.pca.transform(img_vector.reshape(1, -1))
                # ✅ LIMPIAR VALORES INFINITOS
                clean_embedding = self._clean_features_for_storage(embedding.flatten())
                self.trained_embeddings.append(clean_embedding)
                self.trained_labels.append(labels[i])
            except Exception as e:
                print(f"⚠️ Error generando embedding para imagen {i}: {e}")
                # Usar embedding por defecto en caso de error
                default_embedding = np.zeros(actual_components)
                self.trained_embeddings.append(default_embedding)
                self.trained_labels.append(labels[i])

        self.is_trained = True

        print(f"✅ Entrenamiento completado. Eigenfaces generados: {len(self.eigenfaces)}")
        print(f"📊 Varianza explicada: {sum(self.pca.explained_variance_ratio_):.2%}")
        print(f"🔢 Embeddings creados: {len(self.trained_embeddings)}")

    def add_new_person(self, images: List[np.ndarray], person_id: int) -> None:
        """
        ✅ CORREGIDO: Añade una nueva persona con limpieza de características
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

            # ✅ LIMPIAR VALORES INFINITOS
            clean_embedding = self._clean_features_for_storage(embedding.flatten())

            # Añadir a los datos de entrenamiento
            self.trained_embeddings.append(clean_embedding)
            self.trained_labels.append(person_id)

        print(f"✅ Persona añadida. Total embeddings: {len(self.trained_embeddings)}")

    def recognize_face(self, image: np.ndarray) -> Tuple[int, float, dict]:
        """
        ✅ CORREGIDO: Reconoce una cara con manejo robusto de valores infinitos
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        # Extraer características de la imagen
        query_embedding = self.extract_features(image)

        # Calcular distancias a todos los embeddings conocidos
        distances = []
        for embedding in self.trained_embeddings:
            # ✅ VALIDAR QUE AMBOS VECTORES SEAN FINITOS
            if np.any(np.isinf(embedding)) or np.any(np.isnan(embedding)):
                print("⚠️ Embedding almacenado contiene valores infinitos, usando distancia máxima")
                distances.append(float('inf'))
            elif np.any(np.isinf(query_embedding)) or np.any(np.isnan(query_embedding)):
                print("⚠️ Query embedding contiene valores infinitos, usando distancia máxima")
                distances.append(float('inf'))
            else:
                dist = np.linalg.norm(query_embedding - embedding)
                distances.append(dist)

        # Encontrar la menor distancia válida
        finite_distances = [d for d in distances if np.isfinite(d)]

        if not finite_distances:
            print("⚠️ Todas las distancias son infinitas, no se puede reconocer")
            return -1, 0.0, {
                "distance": float('inf'),
                "threshold": self.threshold_distance,
                "is_match": False,
                "confidence_score": 0.0,
                "algorithm": "eigenfaces",
                "error": "All distances are infinite",
                "timestamp": datetime.now().isoformat()
            }

        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        predicted_person_id = self.trained_labels[min_distance_idx]

        # Calcular confianza solo si la distancia es finita
        if np.isfinite(min_distance):
            confidence = max(0, 100 - (min_distance / self.threshold_distance * 100))
        else:
            confidence = 0.0

        # Determinar si es una coincidencia válida
        is_match = np.isfinite(min_distance) and min_distance < self.threshold_distance

        details = {
            "distance": float(min_distance) if np.isfinite(min_distance) else float('inf'),
            "threshold": self.threshold_distance,
            "is_match": is_match,
            "confidence_score": confidence,
            "algorithm": "eigenfaces",
            "finite_distances": len(finite_distances),
            "total_distances": len(distances),
            "timestamp": datetime.now().isoformat()
        }

        return predicted_person_id if is_match else -1, confidence, details

    def save_model(self, path: str = None) -> None:
        """
        ✅ CORREGIDO: Guarda el modelo con validación de valores infinitos
        """
        if path is None:
            path = self.model_path

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # ✅ LIMPIAR EMBEDDINGS ANTES DE GUARDAR
        clean_embeddings = []
        for embedding in self.trained_embeddings:
            clean_embedding = self._clean_features_for_storage(embedding)
            clean_embeddings.append(clean_embedding)

        model_data = {
            'pca': self.pca,
            'mean_face': self.mean_face,
            'eigenfaces': self.eigenfaces,
            'trained_embeddings': clean_embeddings,  # ✅ USAR EMBEDDINGS LIMPIOS
            'trained_labels': self.trained_labels,
            'n_components': self.n_components,
            'image_size': self.image_size,
            'threshold_distance': self.threshold_distance,
            'is_trained': self.is_trained,
            'model_version': '2.0'
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 Modelo guardado en: {path}")

    def load_model(self, path: str = None) -> None:
        """
        ✅ CORREGIDO: Carga un modelo con limpieza automática de valores infinitos
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

        # ✅ LIMPIAR EMBEDDINGS AL CARGAR
        loaded_embeddings = model_data['trained_embeddings']
        self.trained_embeddings = []
        for embedding in loaded_embeddings:
            clean_embedding = self._clean_features_for_storage(np.array(embedding))
            self.trained_embeddings.append(clean_embedding)

        self.trained_labels = model_data['trained_labels']
        self.n_components = model_data['n_components']
        self.image_size = model_data['image_size']
        self.threshold_distance = model_data['threshold_distance']
        self.is_trained = model_data['is_trained']

        print(f"📂 Modelo cargado desde: {path}")
        print(f"📊 Embeddings cargados: {len(self.trained_embeddings)}")

        # Verificar limpieza
        infinite_count = sum(1 for emb in self.trained_embeddings if np.any(np.isinf(emb)))
        if infinite_count > 0:
            print(f"⚠️ {infinite_count} embeddings tenían valores infinitos (limpiados automáticamente)")

    def get_model_info(self) -> dict:
        """
        ✅ CORREGIDO: Obtiene información del modelo con diagnósticos de estabilidad
        """
        info = {
            "algorithm": "eigenfaces",
            "is_trained": self.is_trained,
            "n_components": self.n_components,
            "image_size": self.image_size,
            "total_embeddings": len(self.trained_embeddings) if self.is_trained else 0,
            "unique_persons": len(set(self.trained_labels)) if self.is_trained else 0,
            "threshold_distance": self.threshold_distance,
            "variance_explained": sum(self.pca.explained_variance_ratio_) if self.is_trained else 0,
            "model_version": "2.0"
        }

        # ✅ AÑADIR DIAGNÓSTICOS DE ESTABILIDAD
        if self.is_trained and self.trained_embeddings:
            infinite_embeddings = sum(1 for emb in self.trained_embeddings if np.any(np.isinf(emb)))
            nan_embeddings = sum(1 for emb in self.trained_embeddings if np.any(np.isnan(emb)))

            info.update({
                "stability_diagnostics": {
                    "infinite_embeddings": infinite_embeddings,
                    "nan_embeddings": nan_embeddings,
                    "stable_embeddings": len(self.trained_embeddings) - infinite_embeddings - nan_embeddings,
                    "stability_ratio": (len(self.trained_embeddings) - infinite_embeddings - nan_embeddings) / len(
                        self.trained_embeddings)
                }
            })

        return info

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