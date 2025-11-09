"""
Services package - Servicios del sistema de reconocimiento facial
"""

# Importar servicios base primero (sin dependencias circulares)
from .image_preprocessor import ImagePreprocessor
from .face_detection_service import FaceDetectionService

# Luego servicios de algoritmos
from .eigenfaces_service import EigenfacesService
from .lbp_service import LBPService

# Servicios de mejoras (nuevos)
from .quality_checker import ImageQualityChecker
from .face_alignment import FaceAlignmentService

# Finalmente el servicio principal que depende de los anteriores
from .ml_service import MLService

__all__ = [
    'ImagePreprocessor',
    'FaceDetectionService',
    'EigenfacesService',
    'LBPService',
    'ImageQualityChecker',
    'FaceAlignmentService',
    'MLService'
]