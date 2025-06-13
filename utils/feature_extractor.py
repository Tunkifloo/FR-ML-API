import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime


class FeatureExtractor:
    """
    Extractor de características faciales adicionales
    (complementa a Eigenfaces y LBP)
    """

    def __init__(self):
        """
        Inicializa el extractor de características
        """
        self.feature_cache = {}
        self.cache_file = "storage/embeddings/feature_cache.json"

        # Configuración de características
        self.enable_geometric_features = True
        self.enable_texture_features = True
        self.enable_statistical_features = True

        # Cargar cache si existe
        self.load_cache()

    def extract_comprehensive_features(self, image: np.ndarray, person_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Extrae un conjunto completo de características faciales

        Args:
            image: Imagen del rostro preprocesada
            person_id: ID de la persona (para cache)

        Returns:
            Diccionario con todas las características extraídas
        """
        features = {
            "extraction_timestamp": datetime.now().isoformat(),
            "image_properties": self._extract_image_properties(image),
            "geometric_features": {},
            "texture_features": {},
            "statistical_features": {}
        }

        # Características geométricas
        if self.enable_geometric_features:
            features["geometric_features"] = self._extract_geometric_features(image)

        # Características de textura
        if self.enable_texture_features:
            features["texture_features"] = self._extract_texture_features(image)

        # Características estadísticas
        if self.enable_statistical_features:
            features["statistical_features"] = self._extract_statistical_features(image)

        # Guardar en cache si se proporciona person_id
        if person_id:
            self._cache_features(person_id, features)

        return features

    def _extract_image_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extrae propiedades básicas de la imagen
        """
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1

        return {
            "height": height,
            "width": width,
            "channels": channels,
            "total_pixels": height * width,
            "aspect_ratio": width / height
        }

    def _extract_geometric_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extrae características geométricas usando detección de landmarks faciales
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        geometric_features = {}

        try:
            # Detectar ojos usando Haar Cascades
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

            if len(eyes) >= 2:
                # Ordenar ojos por posición x
                eyes = sorted(eyes, key=lambda x: x[0])
                left_eye = eyes[0]
                right_eye = eyes[1]

                # Calcular distancia entre ojos
                eye_distance = abs(right_eye[0] - left_eye[0])

                # Calcular centro de cada ojo
                left_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
                right_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

                geometric_features.update({
                    "eye_distance": eye_distance,
                    "left_eye_center": left_center,
                    "right_eye_center": right_center,
                    "eye_distance_ratio": eye_distance / gray.shape[1],  # Normalizado por ancho
                    "eyes_detected": True
                })
            else:
                geometric_features["eyes_detected"] = False

            # Detectar nariz
            nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_nose.xml')
            noses = nose_cascade.detectMultiScale(gray, 1.1, 5)

            if len(noses) > 0:
                nose = noses[0]  # Tomar la primera detección
                nose_center = (nose[0] + nose[2] // 2, nose[1] + nose[3] // 2)
                geometric_features.update({
                    "nose_center": nose_center,
                    "nose_width": nose[2],
                    "nose_height": nose[3],
                    "nose_detected": True
                })
            else:
                geometric_features["nose_detected"] = False

            # Detectar boca
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mouth.xml')
            mouths = mouth_cascade.detectMultiScale(gray, 1.1, 5)

            if len(mouths) > 0:
                mouth = mouths[0]
                mouth_center = (mouth[0] + mouth[2] // 2, mouth[1] + mouth[3] // 2)
                geometric_features.update({
                    "mouth_center": mouth_center,
                    "mouth_width": mouth[2],
                    "mouth_height": mouth[3],
                    "mouth_detected": True
                })
            else:
                geometric_features["mouth_detected"] = False

        except Exception as e:
            geometric_features["error"] = str(e)

        return geometric_features

    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extrae características de textura
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        texture_features = {}

        # Gradientes (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitud del gradiente
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        texture_features.update({
            "gradient_mean": float(np.mean(gradient_magnitude)),
            "gradient_std": float(np.std(gradient_magnitude)),
            "gradient_max": float(np.max(gradient_magnitude)),
            "gradient_min": float(np.min(gradient_magnitude))
        })

        # Laplaciano (detección de bordes)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_features.update({
            "laplacian_variance": float(np.var(laplacian)),
            "laplacian_mean": float(np.mean(np.abs(laplacian)))
        })

        # Filtros de Gabor (análisis de frecuencia y orientación)
        gabor_responses = []
        for theta in range(0, 180, 30):  # 6 orientaciones
            for frequency in [0.1, 0.3, 0.5]:  # 3 frecuencias
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta),
                                            2 * np.pi * frequency, 0.5, 0, ktype=cv2.CV_32F)
                gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                gabor_responses.append(np.mean(gabor_response))

        texture_features["gabor_responses"] = gabor_responses
        texture_features["gabor_mean"] = float(np.mean(gabor_responses))
        texture_features["gabor_std"] = float(np.std(gabor_responses))

        return texture_features

    def _extract_statistical_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extrae características estadísticas de la imagen
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Estadísticas básicas
        statistical_features = {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "min": float(np.min(gray)),
            "max": float(np.max(gray)),
            "median": float(np.median(gray)),
            "skewness": float(self._calculate_skewness(gray)),
            "kurtosis": float(self._calculate_kurtosis(gray))
        }

        # Histograma
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / hist.sum()

        statistical_features.update({
            "hist_entropy": float(self._calculate_entropy(hist_normalized)),
            "hist_uniformity": float(np.sum(hist_normalized ** 2)),
            "hist_peak": int(np.argmax(hist)),
            "hist_valley": int(np.argmin(hist[hist > 0])) if len(hist[hist > 0]) > 0 else 0
        })

        # Momentos de Hu (invariantes geométricos)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Log de momentos de Hu para estabilidad numérica
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        statistical_features["hu_moments"] = hu_moments_log.tolist()

        # Análisis de contraste local
        contrast_map = self._calculate_local_contrast(gray)
        statistical_features.update({
            "local_contrast_mean": float(np.mean(contrast_map)),
            "local_contrast_std": float(np.std(contrast_map)),
            "high_contrast_ratio": float(np.sum(contrast_map > np.mean(contrast_map)) / contrast_map.size)
        })

        return statistical_features

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calcula la asimetría de los datos"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calcula la curtosis de los datos"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_entropy(self, prob_dist: np.ndarray) -> float:
        """Calcula la entropía de una distribución de probabilidad"""
        # Evitar log(0)
        prob_dist = prob_dist[prob_dist > 0]
        return -np.sum(prob_dist * np.log2(prob_dist))

    def _calculate_local_contrast(self, image: np.ndarray, window_size: int = 9) -> np.ndarray:
        """Calcula el contraste local usando una ventana deslizante"""
        kernel = np.ones((window_size, window_size), np.float32) / (window_size ** 2)
        mean_filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)

        # Diferencia cuadrática con la media local
        contrast = (image.astype(np.float32) - mean_filtered) ** 2
        return contrast

    def compare_features(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> Dict[str, float]:
        """
        Compara dos conjuntos de características y calcula similitudes

        Args:
            features1: Primer conjunto de características
            features2: Segundo conjunto de características

        Returns:
            Diccionario con métricas de similitud
        """
        similarities = {}

        # Comparar características estadísticas
        if "statistical_features" in features1 and "statistical_features" in features2:
            stat1 = features1["statistical_features"]
            stat2 = features2["statistical_features"]

            # Similitud en estadísticas básicas
            basic_stats = ["mean", "std", "skewness", "kurtosis"]
            basic_diffs = []

            for stat in basic_stats:
                if stat in stat1 and stat in stat2:
                    diff = abs(stat1[stat] - stat2[stat])
                    basic_diffs.append(diff)

            if basic_diffs:
                similarities["statistical_similarity"] = 1.0 / (1.0 + np.mean(basic_diffs))

            # Similitud en momentos de Hu
            if "hu_moments" in stat1 and "hu_moments" in stat2:
                hu1 = np.array(stat1["hu_moments"])
                hu2 = np.array(stat2["hu_moments"])
                hu_distance = np.linalg.norm(hu1 - hu2)
                similarities["geometric_similarity"] = 1.0 / (1.0 + hu_distance)

        # Comparar características de textura
        if "texture_features" in features1 and "texture_features" in features2:
            tex1 = features1["texture_features"]
            tex2 = features2["texture_features"]

            # Similitud en respuestas de Gabor
            if "gabor_responses" in tex1 and "gabor_responses" in tex2:
                gabor1 = np.array(tex1["gabor_responses"])
                gabor2 = np.array(tex2["gabor_responses"])

                # Correlación de Pearson
                correlation = np.corrcoef(gabor1, gabor2)[0, 1]
                similarities["texture_similarity"] = correlation if not np.isnan(correlation) else 0.0

            # Similitud en gradientes
            gradient_features = ["gradient_mean", "gradient_std"]
            gradient_diffs = []

            for feat in gradient_features:
                if feat in tex1 and feat in tex2:
                    diff = abs(tex1[feat] - tex2[feat]) / max(tex1[feat], tex2[feat], 1e-6)
                    gradient_diffs.append(diff)

            if gradient_diffs:
                similarities["gradient_similarity"] = 1.0 - np.mean(gradient_diffs)

        # Similitud general (promedio ponderado)
        if similarities:
            weights = {
                "statistical_similarity": 0.3,
                "geometric_similarity": 0.4,
                "texture_similarity": 0.2,
                "gradient_similarity": 0.1
            }

            weighted_sum = 0
            total_weight = 0

            for sim_name, sim_value in similarities.items():
                if sim_name in weights:
                    weighted_sum += sim_value * weights[sim_name]
                    total_weight += weights[sim_name]

            if total_weight > 0:
                similarities["overall_similarity"] = weighted_sum / total_weight

        return similarities

    def _cache_features(self, person_id: int, features: Dict[str, Any]) -> None:
        """
        Guarda características en cache
        """
        try:
            self.feature_cache[str(person_id)] = features
            self.save_cache()
        except Exception as e:
            print(f"⚠️ Error al guardar features en cache: {e}")

    def get_cached_features(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtiene características desde cache
        """
        return self.feature_cache.get(str(person_id))

    def load_cache(self) -> None:
        """
        Carga cache desde archivo
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.feature_cache = json.load(f)
                print(f"📂 Cache de características cargado: {len(self.feature_cache)} entradas")
        except Exception as e:
            print(f"⚠️ Error al cargar cache: {e}")
            self.feature_cache = {}

    def save_cache(self) -> None:
        """
        Guarda cache en archivo
        """
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.feature_cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error al guardar cache: {e}")

    def clear_cache(self) -> None:
        """
        Limpia el cache
        """
        self.feature_cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("🗑️ Cache de características limpiado")

    def get_feature_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtiene un resumen de las características principales
        """
        summary = {
            "extraction_time": features.get("extraction_timestamp"),
            "image_info": features.get("image_properties", {}),
            "quality_metrics": {}
        }

        # Métricas de calidad basadas en características
        if "statistical_features" in features:
            stats = features["statistical_features"]
            summary["quality_metrics"].update({
                "contrast": stats.get("std", 0),
                "brightness": stats.get("mean", 0),
                "sharpness": stats.get("laplacian_variance", 0) if "texture_features" in features else 0
            })

        if "geometric_features" in features:
            geom = features["geometric_features"]
            summary["quality_metrics"]["facial_features_detected"] = sum([
                geom.get("eyes_detected", False),
                geom.get("nose_detected", False),
                geom.get("mouth_detected", False)
            ])

        # Puntuación de calidad general (0-100)
        quality_score = 0

        # Contraste (25 puntos máximo)
        contrast = summary["quality_metrics"].get("contrast", 0)
        quality_score += min(25, contrast / 2)

        # Nitidez (25 puntos máximo)
        sharpness = summary["quality_metrics"].get("sharpness", 0)
        quality_score += min(25, sharpness / 100)

        # Características faciales detectadas (30 puntos máximo)
        features_detected = summary["quality_metrics"].get("facial_features_detected", 0)
        quality_score += features_detected * 10

        # Brillo adecuado (20 puntos máximo)
        brightness = summary["quality_metrics"].get("brightness", 0)
        brightness_score = 20 - abs(brightness - 128) / 6.4  # Óptimo en 128
        quality_score += max(0, brightness_score)

        summary["quality_score"] = min(100, max(0, quality_score))

        return summary

    def export_features_dataset(self, output_file: str = None) -> str:
        """
        Exporta todas las características como dataset
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"storage/embeddings/features_dataset_{timestamp}.json"

        try:
            # Preparar dataset
            dataset = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_persons": len(self.feature_cache),
                    "feature_types": [
                        "geometric_features",
                        "texture_features",
                        "statistical_features"
                    ]
                },
                "persons": {}
            }

            # Añadir características por persona
            for person_id, features in self.feature_cache.items():
                dataset["persons"][person_id] = {
                    "features": features,
                    "summary": self.get_feature_summary(features)
                }

            # Guardar dataset
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)

            print(f"📊 Dataset exportado: {output_file}")
            return output_file

        except Exception as e:
            raise Exception(f"Error al exportar dataset: {e}")

    def analyze_feature_distribution(self) -> Dict[str, Any]:
        """
        Analiza la distribución de características en el cache
        """
        if not self.feature_cache:
            return {"error": "No hay características en cache"}

        analysis = {
            "total_persons": len(self.feature_cache),
            "feature_statistics": {},
            "quality_distribution": [],
            "common_characteristics": {}
        }

        # Recopilar todas las características numéricas
        all_stats = []
        all_geometric = []
        quality_scores = []

        for features in self.feature_cache.values():
            # Características estadísticas
            if "statistical_features" in features:
                stats = features["statistical_features"]
                numeric_stats = {k: v for k, v in stats.items()
                                 if isinstance(v, (int, float)) and k != "hist_peak"}
                all_stats.append(numeric_stats)

            # Puntuación de calidad
            summary = self.get_feature_summary(features)
            quality_scores.append(summary["quality_score"])

        # Estadísticas de calidad
        if quality_scores:
            analysis["quality_distribution"] = {
                "mean": float(np.mean(quality_scores)),
                "std": float(np.std(quality_scores)),
                "min": float(np.min(quality_scores)),
                "max": float(np.max(quality_scores)),
                "excellent": sum(1 for q in quality_scores if q >= 80),
                "good": sum(1 for q in quality_scores if 60 <= q < 80),
                "fair": sum(1 for q in quality_scores if 40 <= q < 60),
                "poor": sum(1 for q in quality_scores if q < 40)
            }

        # Estadísticas de características
        if all_stats:
            # Combinar todas las características estadísticas
            combined_stats = {}
            for stat_dict in all_stats:
                for key, value in stat_dict.items():
                    if key not in combined_stats:
                        combined_stats[key] = []
                    combined_stats[key].append(value)

            # Calcular estadísticas por característica
            for key, values in combined_stats.items():
                analysis["feature_statistics"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }

        return analysis