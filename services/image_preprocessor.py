import cv2
import numpy as np
from typing import Tuple, Optional
import os


class ImagePreprocessor:
    """
    Preprocesador unificado de imágenes CORREGIDO para manejar errores CLAHE
    """

    def __init__(self, target_size: Tuple[int, int] = (100, 100)):
        self.target_size = target_size
        self.target_pixels = target_size[0] * target_size[1]

    def preprocess_for_ml(self, image: np.ndarray, algorithm: str = "both") -> np.ndarray:
        """
        CORREGIDO: Preprocesa imagen con manejo robusto de tipos

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

        # CRÍTICO: Asegurar tipo uint8 ANTES de cualquier operación
        if processed.dtype != np.uint8:
            if processed.max() <= 1.0:
                # Si está normalizada [0,1], escalar a [0,255]
                processed = (processed * 255).astype(np.uint8)
            else:
                # Si está en otro rango, convertir directamente
                processed = processed.astype(np.uint8)
            print(f"🔧 Convertida a uint8: dtype={processed.dtype}")

        # PASO 3: Redimensionar SIEMPRE al tamaño target
        if processed.shape != self.target_size:
            processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            print(f"🔧 Redimensionada: {processed.shape}")

        # PASO 4: Aplicar filtros básicos (en uint8)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # PASO 5: Normalización específica por algoritmo
        if algorithm == "eigenfaces":
            # Para Eigenfaces: ecualización básica + normalización [0,1]
            processed = cv2.equalizeHist(processed)
            processed = processed.astype(np.float64) / 255.0
            print(
                f"🔧 Normalizada para Eigenfaces: dtype={processed.dtype}, range=[{processed.min():.3f}, {processed.max():.3f}]")

        elif algorithm == "lbp":
            # Para LBP: CLAHE SEGURO (ya está en uint8)
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
                print(f"🔧 CLAHE aplicado para LBP: dtype={processed.dtype}")
            except Exception as e:
                print(f"⚠️ Error en CLAHE, usando ecualización básica: {e}")
                processed = cv2.equalizeHist(processed)

        elif algorithm == "both":
            # Para entrenamiento general - ecualización básica + normalización
            processed = cv2.equalizeHist(processed)
            processed = processed.astype(np.float64) / 255.0
            print(f"🔧 Normalizada para ambos algoritmos: dtype={processed.dtype}")

        return processed

    def _normalize_dimensions(self, image: np.ndarray) -> np.ndarray:
        """
        CORREGIDO: Normaliza dimensiones manteniendo tipos correctos
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
        Valida que una imagen sea procesable
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
        Obtiene información detallada de una imagen
        """
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "size": image.size,
            "min_value": float(image.min()),
            "max_value": float(image.max()),
            "mean_value": float(image.mean()),
            "is_valid": self.validate_image(image)
        }