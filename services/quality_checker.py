import cv2
import numpy as np
from typing import Dict


class ImageQualityChecker:
    """
    Evalúa calidad de imágenes antes del procesamiento
    """

    @staticmethod
    def check_image_quality(image: np.ndarray) -> Dict[str, any]:
        """
        Evalúa múltiples aspectos de calidad de imagen
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        quality_metrics = {
            "resolution_score": ImageQualityChecker._check_resolution(image),
            "brightness_score": ImageQualityChecker._check_brightness(gray),
            "contrast_score": ImageQualityChecker._check_contrast(gray),
            "sharpness_score": ImageQualityChecker._check_sharpness(gray),
            "noise_level": ImageQualityChecker._estimate_noise(gray)
        }

        # Calcular score general
        weights = {
            "resolution_score": 0.2,
            "brightness_score": 0.2,
            "contrast_score": 0.2,
            "sharpness_score": 0.3,
            "noise_level": 0.1
        }

        overall_score = sum(
            quality_metrics[key] * weights[key]
            for key in weights.keys()
        )

        quality_metrics["overall_score"] = overall_score
        quality_metrics["quality_level"] = ImageQualityChecker._get_quality_level(overall_score)
        quality_metrics["is_acceptable"] = overall_score >= 40

        return quality_metrics

    @staticmethod
    def _check_resolution(image: np.ndarray) -> float:
        """Verifica si la resolución es adecuada"""
        height, width = image.shape[:2]
        pixels = height * width

        if pixels >= 640 * 480:
            return 100
        elif pixels >= 320 * 240:
            return 75
        elif pixels >= 160 * 120:
            return 50
        else:
            return 25

    @staticmethod
    def _check_brightness(gray: np.ndarray) -> float:
        """Verifica si el brillo es adecuado"""
        mean_brightness = np.mean(gray)

        if 100 <= mean_brightness <= 150:
            return 100
        elif 80 <= mean_brightness <= 180:
            return 80
        elif 60 <= mean_brightness <= 200:
            return 60
        else:
            return 40

    @staticmethod
    def _check_contrast(gray: np.ndarray) -> float:
        """Verifica contraste de la imagen"""
        std_dev = np.std(gray)

        if std_dev >= 50:
            return 100
        elif std_dev >= 35:
            return 80
        elif std_dev >= 20:
            return 60
        else:
            return 40

    @staticmethod
    def _check_sharpness(gray: np.ndarray) -> float:
        """Verifica nitidez usando varianza de Laplaciano"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        if variance >= 100:
            return 100
        elif variance >= 50:
            return 75
        elif variance >= 25:
            return 50
        else:
            return 25

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        """Estima nivel de ruido (menor es mejor)"""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.std(gray.astype(np.float32) - blurred.astype(np.float32))

        if noise <= 5:
            return 100
        elif noise <= 10:
            return 80
        elif noise <= 15:
            return 60
        else:
            return 40

    @staticmethod
    def _get_quality_level(score: float) -> str:
        """Clasifica nivel de calidad"""
        if score >= 80:
            return "Excelente"
        elif score >= 60:
            return "Buena"
        elif score >= 40:
            return "Aceptable"
        else:
            return "Pobre"