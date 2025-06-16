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
    ✅ CORREGIDO: Implementación del algoritmo Local Binary Patterns (LBP) para reconocimiento facial
    Con manejo adecuado de tipos de datos uint8 para CLAHE
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
        self.method = 'uniform'

        # Datos de entrenamiento
        self.trained_histograms = []
        self.trained_labels = []
        self.is_trained = False

        # Configuración
        self.threshold_similarity = 0.7
        self.model_path = "storage/models/lbp_model.pkl"
        self.image_size = (100, 100)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        ✅ CORREGIDO: Preprocesa imagen para LBP asegurando uint8 para CLAHE
        """
        print(f"🔧 LBP preprocess input: {image.shape}, dtype: {image.dtype}")

        processed = image.copy()

        # PASO 1: Manejar diferentes tipos de entrada
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

        # PASO 2: Asegurar que sea 2D
        if len(processed.shape) != 2:
            raise ValueError(f"Error: imagen debe ser 2D, recibida: {processed.shape}")

        # PASO 3: Redimensionar a tamaño estándar
        if processed.shape != self.image_size:
            processed = cv2.resize(processed, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            print(f"🔧 Redimensionada: {processed.shape}")

        # ✅ CRÍTICO: Convertir a uint8 ANTES de cualquier operación OpenCV
        if processed.dtype != np.uint8:
            if processed.max() <= 1.0:
                # Si está normalizada [0,1], escalar a [0,255]
                processed = (processed * 255).astype(np.uint8)
                print(f"🔧 Convertida de float [0,1] a uint8: {processed.dtype}")
            else:
                # Si está en otro rango, convertir directamente
                processed = np.clip(processed, 0, 255).astype(np.uint8)
                print(f"🔧 Convertida a uint8: {processed.dtype}")

        # PASO 4: Aplicar filtro gaussiano (ahora en uint8)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # PASO 5: Ecualización adaptiva CLAHE (solo funciona con uint8)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
            print(f"✅ CLAHE aplicado exitosamente: dtype={processed.dtype}")
        except Exception as e:
            print(f"❌ Error en CLAHE: {e}")
            # Fallback a ecualización básica
            processed = cv2.equalizeHist(processed)
            print(f"🔄 Fallback a equalizeHist: dtype={processed.dtype}")

        print(f"✅ LBP preprocessing completado: {processed.shape}, dtype={processed.dtype}")
        return processed

    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """
        ✅ CORREGIDO: Extrae características LBP con manejo robusto de tipos
        """
        print(f"🔍 LBP extract_lbp_features input: {image.shape}, dtype: {image.dtype}")

        # VALIDAR QUE LA IMAGEN SEA ADECUADA PARA LBP
        processed_image = image.copy()

        # MANEJAR DIFERENTES TIPOS DE ENTRADA
        if len(processed_image.shape) == 1:
            # Vector 1D - intentar reformatear
            target_pixels = self.image_size[0] * self.image_size[1]

            if processed_image.shape[0] == target_pixels:
                processed_image = processed_image.reshape(self.image_size)
                print(f"🔧 Vector 1D reformateado a image_size: {processed_image.shape}")
            else:
                size = int(np.sqrt(processed_image.shape[0]))
                if size * size == processed_image.shape[0]:
                    processed_image = processed_image.reshape(size, size)
                    print(f"🔧 Vector 1D reformateado a cuadrado: {processed_image.shape}")
                else:
                    raise ValueError(f"Vector 1D {processed_image.shape} no compatible con LBP")

        elif len(processed_image.shape) == 3:
            # Imagen color - convertir a escala de grises
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            print(f"🔧 Convertida a escala de grises: {processed_image.shape}")

        # VERIFICAR QUE SEA 2D
        if len(processed_image.shape) != 2:
            raise ValueError(f"LBP requiere imagen 2D, recibida: {processed_image.shape}")

        print(f"✅ Imagen 2D confirmada: {processed_image.shape}")

        # ✅ ASEGURAR QUE ESTÉ EN EL FORMATO CORRECTO PARA LBP
        if not (processed_image.dtype == np.uint8 and processed_image.shape == self.image_size):
            print(f"🔧 Aplicando preprocesamiento LBP...")
            processed_image = self.preprocess_image(processed_image)

        # ✅ VERIFICAR TIPO FINAL ANTES DE LBP
        if processed_image.dtype != np.uint8:
            raise ValueError(f"LBP requiere uint8, recibido: {processed_image.dtype}")

        # CALCULAR LBP
        try:
            print(
                f"🔍 Calculando LBP con parámetros: radius={self.radius}, n_points={self.n_points}, method={self.method}")
            lbp_image = local_binary_pattern(
                processed_image,
                self.n_points,
                self.radius,
                method=self.method
            )
            print(f"✅ LBP calculado: shape={lbp_image.shape}, dtype={lbp_image.dtype}")
        except Exception as e:
            print(f"❌ Error calculando LBP: {e}")
            print(f"   Imagen: shape={processed_image.shape}, dtype={processed_image.dtype}")
            print(f"   Range: [{processed_image.min()}, {processed_image.max()}]")
            raise

        # DIVIDIR EN GRILLA Y CALCULAR HISTOGRAMAS
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
        ✅ CORREGIDO: Entrena el modelo LBP con validación de tipos
        """
        print(f"🎓 Iniciando entrenamiento LBP con {len(images)} imágenes...")

        self.trained_histograms = []
        self.trained_labels = []

        for i, (img, label) in enumerate(zip(images, labels)):
            try:
                print(f"🔧 Procesando imagen {i + 1}/{len(images)}: {img.shape}, dtype={img.dtype}")

                # ✅ VALIDAR TIPO DE ENTRADA
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        # Convertir de float [0,1] a uint8 [0,255]
                        img_uint8 = (img * 255).astype(np.uint8)
                        print(f"   🔧 Convertida de float a uint8: {img_uint8.dtype}")
                    else:
                        # Convertir directamente a uint8
                        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                        print(f"   🔧 Convertida a uint8: {img_uint8.dtype}")
                else:
                    img_uint8 = img.copy()

                # Preprocesar imagen
                processed_img = self.preprocess_image(img_uint8)

                # Extraer características LBP
                lbp_features = self.extract_lbp_features(processed_img)

                # Almacenar
                self.trained_histograms.append(lbp_features)
                self.trained_labels.append(label)

                if (i + 1) % 5 == 0:
                    print(f"   ✅ Procesadas {i + 1}/{len(images)} imágenes...")

            except Exception as e:
                print(f"   ❌ Error procesando imagen {i + 1}: {e}")
                continue

        self.is_trained = True
        print(f"✅ Entrenamiento LBP completado. Características extraídas: {len(self.trained_histograms)}")

    def add_new_person(self, images: List[np.ndarray], person_id: int) -> None:
        """
        ✅ CORREGIDO: Añade una nueva persona al modelo (entrenamiento incremental)
        """
        print(f"➕ Añadiendo nueva persona ID: {person_id} con {len(images)} imágenes")

        for i, img in enumerate(images):
            try:
                print(
                    f"🔧 Procesando imagen {i + 1}/{len(images)} para persona {person_id}: {img.shape}, dtype={img.dtype}")

                # ✅ VALIDAR Y CONVERTIR TIPO SI ES NECESARIO
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img_uint8 = (img * 255).astype(np.uint8)
                        print(f"   🔧 Convertida de float a uint8")
                    else:
                        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                        print(f"   🔧 Convertida a uint8")
                else:
                    img_uint8 = img.copy()

                # Preprocesar imagen
                processed_img = self.preprocess_image(img_uint8)

                # Extraer características LBP
                lbp_features = self.extract_lbp_features(processed_img)

                # Añadir a los datos de entrenamiento
                self.trained_histograms.append(lbp_features)
                self.trained_labels.append(person_id)

                print(f"   ✅ Imagen {i + 1} procesada exitosamente")

            except Exception as e:
                print(f"   ❌ Error procesando imagen {i + 1}: {e}")
                continue

        print(f"✅ Persona {person_id} añadida. Total características: {len(self.trained_histograms)}")

    def recognize_face(self, image: np.ndarray) -> Tuple[int, float, dict]:
        """
        ✅ CORREGIDO: Reconoce una cara usando LBP con validación de tipos
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        print(f"🔍 LBP recognize_face input: {image.shape}, dtype: {image.dtype}")

        # ✅ VALIDAR Y CONVERTIR TIPO SI ES NECESARIO
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
                print(f"🔧 Convertida de float a uint8 para reconocimiento")
            else:
                image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
                print(f"🔧 Convertida a uint8 para reconocimiento")
        else:
            image_uint8 = image.copy()

        # Extraer características de la imagen consultada
        processed_img = self.preprocess_image(image_uint8)
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
            "data_type_used": "uint8",
            "timestamp": datetime.now().isoformat()
        }

        return predicted_person_id if is_match else -1, confidence, details

    def save_model(self, path: str = None) -> None:
        """
        ✅ CORREGIDO: Guarda el modelo entrenado
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
            'is_trained': self.is_trained,
            'model_version': '2.0_FIXED',
            'data_type_requirement': 'uint8 [0,255] for CLAHE compatibility'
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 Modelo LBP guardado en: {path}")

    def load_model(self, path: str = None) -> None:
        """
        ✅ CORREGIDO: Carga un modelo previamente entrenado
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

        # Mostrar información de versión si está disponible
        if 'model_version' in model_data:
            print(f"📋 Versión del modelo: {model_data['model_version']}")

    def get_model_info(self) -> dict:
        """
        ✅ CORREGIDO: Obtiene información del modelo actual
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
            "feature_vector_size": len(self.trained_histograms[0]) if self.trained_histograms else 0,
            "data_type_requirement": "uint8 [0,255]",
            "preprocessing_steps": [
                "Conversion to grayscale",
                "Resize to target size",
                "Convert to uint8 [0,255]",
                "Gaussian blur",
                "CLAHE adaptive histogram equalization"
            ],
            "model_version": "2.0_FIXED"
        }

    def calculate_lbp_histogram_comparison(self, hist1: np.ndarray, hist2: np.ndarray) -> dict:
        """
        ✅ CORREGIDO: Compara dos histogramas LBP usando diferentes métricas
        """
        # Asegurar que los histogramas sean float32 para OpenCV
        hist1_f32 = hist1.astype(np.float32)
        hist2_f32 = hist2.astype(np.float32)

        # Correlación
        correlation = np.corrcoef(hist1, hist2)[0, 1] if len(hist1) > 1 and len(hist2) > 1 else 0.0

        # Chi-cuadrado
        try:
            chi_squared = cv2.compareHist(hist1_f32, hist2_f32, cv2.HISTCMP_CHISQR)
        except:
            chi_squared = float('inf')

        # Intersección
        try:
            intersection = cv2.compareHist(hist1_f32, hist2_f32, cv2.HISTCMP_INTERSECT)
        except:
            intersection = 0.0

        # Bhattacharyya
        try:
            bhattacharyya = cv2.compareHist(hist1_f32, hist2_f32, cv2.HISTCMP_BHATTACHARYYA)
        except:
            bhattacharyya = 1.0

        return {
            "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
            "chi_squared": float(chi_squared) if not np.isinf(chi_squared) else 999999.0,
            "intersection": float(intersection),
            "bhattacharyya": float(bhattacharyya)
        }

    def validate_input_image(self, image: np.ndarray) -> dict:
        """
        ✅ NUEVO: Valida una imagen de entrada para LBP
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "input_info": {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "min_value": float(image.min()),
                "max_value": float(image.max())
            }
        }

        # Validar dimensiones
        if len(image.shape) not in [1, 2, 3]:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Dimensiones inválidas: {image.shape}")

        # Validar tamaño mínimo
        if len(image.shape) >= 2:
            if image.shape[0] < 50 or image.shape[1] < 50:
                validation_result["warnings"].append(f"Imagen muy pequeña: {image.shape[:2]}")

        # Validar tipo de datos
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            validation_result["warnings"].append(f"Tipo de datos poco común: {image.dtype}")

        # Validar rango de valores
        if image.dtype == np.uint8:
            if image.min() < 0 or image.max() > 255:
                validation_result["errors"].append(f"Valores fuera de rango uint8: [{image.min()}, {image.max()}]")
        elif image.max() <= 1.0:
            # Probablemente normalizada
            validation_result["input_info"]["likely_normalized"] = True

        # Validar que no esté vacía
        if image.size == 0:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Imagen vacía")

        return validation_result

    def debug_preprocessing_pipeline(self, image: np.ndarray) -> dict:
        """
        ✅ NUEVO: Función de debugging para el pipeline de preprocesamiento
        """
        debug_info = {
            "step_by_step": [],
            "final_success": False,
            "error_occurred": None
        }

        try:
            # Paso 1: Imagen original
            debug_info["step_by_step"].append({
                "step": "input",
                "shape": image.shape,
                "dtype": str(image.dtype),
                "range": [float(image.min()), float(image.max())]
            })

            # Paso 2: Conversión de tipo
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    converted = (image * 255).astype(np.uint8)
                else:
                    converted = np.clip(image, 0, 255).astype(np.uint8)

                debug_info["step_by_step"].append({
                    "step": "type_conversion",
                    "shape": converted.shape,
                    "dtype": str(converted.dtype),
                    "range": [float(converted.min()), float(converted.max())]
                })
            else:
                converted = image.copy()

            # Paso 3: Conversión a escala de grises
            if len(converted.shape) == 3:
                gray = cv2.cvtColor(converted, cv2.COLOR_BGR2GRAY)
                debug_info["step_by_step"].append({
                    "step": "grayscale_conversion",
                    "shape": gray.shape,
                    "dtype": str(gray.dtype),
                    "range": [float(gray.min()), float(gray.max())]
                })
            else:
                gray = converted

            # Paso 4: Redimensionar
            if gray.shape != self.image_size:
                resized = cv2.resize(gray, self.image_size)
                debug_info["step_by_step"].append({
                    "step": "resize",
                    "shape": resized.shape,
                    "dtype": str(resized.dtype),
                    "range": [float(resized.min()), float(resized.max())]
                })
            else:
                resized = gray

            # Paso 5: CLAHE
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_result = clahe.apply(resized)
                debug_info["step_by_step"].append({
                    "step": "clahe_success",
                    "shape": clahe_result.shape,
                    "dtype": str(clahe_result.dtype),
                    "range": [float(clahe_result.min()), float(clahe_result.max())]
                })
            except Exception as e:
                debug_info["step_by_step"].append({
                    "step": "clahe_failed",
                    "error": str(e),
                    "fallback": "equalizeHist"
                })
                clahe_result = cv2.equalizeHist(resized)

            debug_info["final_success"] = True

        except Exception as e:
            debug_info["error_occurred"] = str(e)
            debug_info["final_success"] = False

        return debug_info

    def test_with_sample_images(self, sample_images: List[np.ndarray]) -> dict:
        """
        ✅ NUEVO: Prueba el servicio con imágenes de muestra
        """
        test_results = {
            "total_tested": len(sample_images),
            "successful_preprocessing": 0,
            "successful_feature_extraction": 0,
            "failed_images": [],
            "processing_times": [],
            "feature_vector_sizes": []
        }

        for i, img in enumerate(sample_images):
            start_time = datetime.now()

            try:
                # Validar entrada
                validation = self.validate_input_image(img)
                if not validation["is_valid"]:
                    test_results["failed_images"].append({
                        "image_index": i,
                        "stage": "validation",
                        "errors": validation["errors"]
                    })
                    continue

                # Preprocesar
                processed = self.preprocess_image(img)
                test_results["successful_preprocessing"] += 1

                # Extraer características (solo si está entrenado)
                if self.is_trained:
                    features = self.extract_lbp_features(processed)
                    test_results["successful_feature_extraction"] += 1
                    test_results["feature_vector_sizes"].append(len(features))

                # Calcular tiempo
                processing_time = (datetime.now() - start_time).total_seconds()
                test_results["processing_times"].append(processing_time)

            except Exception as e:
                test_results["failed_images"].append({
                    "image_index": i,
                    "stage": "processing",
                    "error": str(e)
                })

        # Estadísticas
        if test_results["processing_times"]:
            test_results["avg_processing_time"] = np.mean(test_results["processing_times"])
            test_results["max_processing_time"] = np.max(test_results["processing_times"])

        test_results["success_rate"] = test_results["successful_preprocessing"] / test_results["total_tested"] * 100

        return test_results