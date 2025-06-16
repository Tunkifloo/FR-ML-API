import cv2
import numpy as np
from typing import Tuple, Optional
import os


class ImagePreprocessor:
    """
    ✅ CORREGIDO: Preprocesador unificado de imágenes con manejo robusto de tipos de datos
    """

    def __init__(self, target_size: Tuple[int, int] = (100, 100)):
        self.target_size = target_size
        self.target_pixels = target_size[0] * target_size[1]

    def preprocess_for_ml(self, image: np.ndarray, algorithm: str = "both") -> np.ndarray:
        """
        ✅ CORREGIDO: Preprocesa imagen con manejo específico por algoritmo

        Args:
            image: Imagen original (cualquier formato)
            algorithm: "eigenfaces", "lbp", o "both"
        """
        print(f"🔧 Preprocessing para {algorithm}: input shape {image.shape}, dtype {image.dtype}")

        # PASO 1: Normalizar dimensiones
        processed = self._normalize_dimensions(image)

        # PASO 2: Convertir a escala de grises si es necesario
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            print(f"🔧 Convertida a escala de grises: {processed.shape}, dtype: {processed.dtype}")

        # PASO 3: Redimensionar SIEMPRE al tamaño target
        if processed.shape != self.target_size:
            processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            print(f"🔧 Redimensionada: {processed.shape}")

        # ✅ PASO 4: PROCESAMIENTO ESPECÍFICO POR ALGORITMO
        if algorithm == "eigenfaces":
            return self._preprocess_for_eigenfaces(processed)
        elif algorithm == "lbp":
            return self._preprocess_for_lbp(processed)
        elif algorithm == "both":
            return self._preprocess_for_both(processed)
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")

    def _preprocess_for_eigenfaces(self, image: np.ndarray) -> np.ndarray:
        """
        ✅ NUEVO: Preprocesamiento específico para Eigenfaces (salida: float64 [0,1])
        """
        print(f"🔧 Preprocesando para Eigenfaces...")

        # Asegurar tipo uint8 ANTES de operaciones OpenCV
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                processed = (image * 255).astype(np.uint8)
            else:
                processed = np.clip(image, 0, 255).astype(np.uint8)
        else:
            processed = image.copy()

        # Aplicar filtros básicos (en uint8)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # Ecualización básica para Eigenfaces
        processed = cv2.equalizeHist(processed)

        # CONVERTIR A FLOAT64 [0,1] PARA PCA
        processed = processed.astype(np.float64) / 255.0

        print(f"✅ Eigenfaces listo: dtype={processed.dtype}, range=[{processed.min():.3f}, {processed.max():.3f}]")
        return processed

    def _preprocess_for_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        ✅ NUEVO: Preprocesamiento específico para LBP (salida: uint8 [0,255])
        """
        print(f"🔧 Preprocesando para LBP...")

        # Asegurar tipo uint8 ANTES de cualquier operación
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                processed = (image * 255).astype(np.uint8)
            else:
                processed = np.clip(image, 0, 255).astype(np.uint8)
        else:
            processed = image.copy()

        # Aplicar filtros básicos (en uint8)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # Ecualización adaptiva CLAHE (solo funciona con uint8)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
            print(f"✅ CLAHE aplicado exitosamente")
        except Exception as e:
            print(f"⚠️ Error en CLAHE: {e}, usando ecualización básica")
            processed = cv2.equalizeHist(processed)

        # MANTENER COMO UINT8 PARA LBP
        print(f"✅ LBP listo: dtype={processed.dtype}, range=[{processed.min()}, {processed.max()}]")
        return processed

    def _preprocess_for_both(self, image: np.ndarray) -> np.ndarray:
        """
        ✅ NUEVO: Preprocesamiento para ambos algoritmos (salida: float64 [0,1])
        """
        print(f"🔧 Preprocesando para ambos algoritmos...")

        # Asegurar tipo uint8 ANTES de operaciones OpenCV
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                processed = (image * 255).astype(np.uint8)
            else:
                processed = np.clip(image, 0, 255).astype(np.uint8)
        else:
            processed = image.copy()

        # Aplicar filtros básicos
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # Ecualización básica (compatible con ambos)
        processed = cv2.equalizeHist(processed)

        # CONVERTIR A FLOAT64 [0,1] COMO BASE
        # (LBP puede convertir de vuelta a uint8 cuando sea necesario)
        processed = processed.astype(np.float64) / 255.0

        print(
            f"✅ Ambos algoritmos listo: dtype={processed.dtype}, range=[{processed.min():.3f}, {processed.max():.3f}]")
        return processed

    def _normalize_dimensions(self, image: np.ndarray) -> np.ndarray:
        """
        ✅ CORREGIDO: Normaliza dimensiones manteniendo tipos correctos
        """
        print(f"🔍 Normalizando dimensiones: {image.shape}, dtype: {image.dtype}")

        if len(image.shape) == 1:
            # Vector 1D - reformatear
            if image.shape[0] == self.target_pixels:
                reshaped = image.reshape(self.target_size)
                print(f"🔧 Vector 1D reformateado a target_size: {reshaped.shape}")
                return reshaped
            else:
                # Intentar formar cuadrado perfecto
                size = int(np.sqrt(image.shape[0]))
                if size * size == image.shape[0]:
                    reshaped = image.reshape(size, size)
                    print(f"🔧 Vector 1D reformateado a cuadrado: {reshaped.shape}")
                    return reshaped
                else:
                    raise ValueError(f"Vector 1D {image.shape} no puede ser reformateado a imagen válida")

        elif len(image.shape) in [2, 3]:
            return image

        else:
            raise ValueError(f"Dimensiones no soportadas: {image.shape}")

    def validate_image(self, image: np.ndarray) -> bool:
        """
        ✅ CORREGIDO: Valida que una imagen sea procesable
        """
        try:
            if image is None:
                return False

            if len(image.shape) not in [1, 2, 3]:
                return False

            if image.size == 0:
                return False

            # Intentar procesar
            self._normalize_dimensions(image)
            return True

        except Exception:
            return False

    def get_image_info(self, image: np.ndarray) -> dict:
        """
        ✅ CORREGIDO: Obtiene información detallada de una imagen
        """
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "size": image.size,
            "min_value": float(image.min()),
            "max_value": float(image.max()),
            "mean_value": float(image.mean()),
            "is_valid": self.validate_image(image),
            "target_size": self.target_size,
            "processing_recommendations": self._get_processing_recommendations(image)
        }

    def _get_processing_recommendations(self, image: np.ndarray) -> dict:
        """
        ✅ NUEVO: Obtiene recomendaciones de procesamiento para una imagen
        """
        recommendations = {
            "eigenfaces": [],
            "lbp": [],
            "general": []
        }

        # Analizar tipo de datos
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            recommendations["general"].append(f"Convertir tipo de datos desde {image.dtype}")

        # Analizar rango de valores
        if image.dtype == np.uint8:
            if image.min() < 0 or image.max() > 255:
                recommendations["general"].append("Valores fuera del rango uint8 válido")
        elif image.max() <= 1.0:
            recommendations["eigenfaces"].append("Imagen ya normalizada para Eigenfaces")
            recommendations["lbp"].append("Necesita conversión a uint8 para LBP")
        else:
            recommendations["general"].append("Normalizar valores al rango apropiado")

        # Analizar dimensiones
        if len(image.shape) > 2:
            recommendations["general"].append("Convertir a escala de grises")

        if image.shape[:2] != self.target_size:
            recommendations["general"].append(f"Redimensionar a {self.target_size}")

        # Analizar calidad de imagen
        if len(image.shape) >= 2:
            # Calcular métricas de calidad básicas
            if image.std() < 20:  # Bajo contraste
                recommendations["general"].append("Aplicar mejora de contraste")
                recommendations["eigenfaces"].append("Ecualización de histograma recomendada")
                recommendations["lbp"].append("CLAHE recomendado")

        return recommendations

    def convert_for_algorithm(self, image: np.ndarray, target_algorithm: str) -> np.ndarray:
        """
        ✅ NUEVO: Convierte imagen procesada para algoritmo específico
        """
        if target_algorithm == "eigenfaces":
            # Si ya está en float64 [0,1], mantener
            if image.dtype == np.float64 and image.max() <= 1.0:
                return image
            # Si está en uint8, convertir a float64 [0,1]
            elif image.dtype == np.uint8:
                return image.astype(np.float64) / 255.0
            else:
                # Normalizar y convertir
                if image.max() > 1.0:
                    normalized = image / 255.0
                else:
                    normalized = image
                return normalized.astype(np.float64)

        elif target_algorithm == "lbp":
            # Si ya está en uint8, mantener
            if image.dtype == np.uint8:
                return image
            # Si está en float [0,1], convertir a uint8 [0,255]
            elif image.max() <= 1.0:
                return (image * 255).astype(np.uint8)
            else:
                # Clip y convertir
                return np.clip(image, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Algoritmo no soportado: {target_algorithm}")

    def batch_preprocess(self, images: list, algorithm: str = "both") -> list:
        """
        ✅ NUEVO: Preprocesa un lote de imágenes
        """
        processed_images = []

        for i, image in enumerate(images):
            try:
                processed = self.preprocess_for_ml(image, algorithm)
                processed_images.append(processed)

                if (i + 1) % 10 == 0:
                    print(f"📸 Procesadas {i + 1}/{len(images)} imágenes...")

            except Exception as e:
                print(f"⚠️ Error procesando imagen {i + 1}: {e}")
                continue

        print(f"✅ Batch procesado: {len(processed_images)}/{len(images)} exitosas")
        return processed_images

    def debug_preprocessing_step_by_step(self, image: np.ndarray, algorithm: str) -> dict:
        """
        ✅ NUEVO: Debug paso a paso del preprocesamiento
        """
        debug_info = {
            "original": {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "range": [float(image.min()), float(image.max())]
            },
            "steps": [],
            "final": {},
            "success": False
        }

        try:
            current = image.copy()

            # Paso 1: Normalizar dimensiones
            if len(current.shape) != 2 or current.shape != self.target_size:
                current = self._normalize_dimensions(current)
                if len(current.shape) == 3:
                    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                if current.shape != self.target_size:
                    current = cv2.resize(current, self.target_size)

                debug_info["steps"].append({
                    "step": "normalize_and_resize",
                    "shape": current.shape,
                    "dtype": str(current.dtype),
                    "range": [float(current.min()), float(current.max())]
                })

            # Paso 2: Procesamiento específico por algoritmo
            if algorithm == "eigenfaces":
                current = self._preprocess_for_eigenfaces(current)
            elif algorithm == "lbp":
                current = self._preprocess_for_lbp(current)
            else:
                current = self._preprocess_for_both(current)

            debug_info["final"] = {
                "shape": current.shape,
                "dtype": str(current.dtype),
                "range": [float(current.min()), float(current.max())]
            }
            debug_info["success"] = True

        except Exception as e:
            debug_info["error"] = str(e)
            debug_info["success"] = False

        return debug_info

    def compare_algorithm_preprocessing(self, image: np.ndarray) -> dict:
        """
        ✅ NUEVO: Compara el preprocesamiento para diferentes algoritmos
        """
        comparison = {
            "original": self.get_image_info(image),
            "algorithms": {}
        }

        for algorithm in ["eigenfaces", "lbp", "both"]:
            try:
                processed = self.preprocess_for_ml(image.copy(), algorithm)
                comparison["algorithms"][algorithm] = {
                    "success": True,
                    "result": self.get_image_info(processed),
                    "suitable_for": [algorithm] if algorithm != "both" else ["eigenfaces", "lbp"]
                }
            except Exception as e:
                comparison["algorithms"][algorithm] = {
                    "success": False,
                    "error": str(e)
                }

        return comparison