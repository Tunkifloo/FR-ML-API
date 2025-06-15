# services/__init__.py
"""
Servicios de Machine Learning para reconocimiento facial
"""

from .ml_service import MLService
from .eigenfaces_service import EigenfacesService
from .lbp_service import LBPService
from .face_detection_service import FaceDetectionService
from .image_preprocessor import ImagePreprocessor  # ⚡ AÑADIR

__all__ = [
    "MLService",
    "EigenfacesService",
    "LBPService",
    "FaceDetectionService",
    "ImagePreprocessor"  # ⚡ AÑADIR
]