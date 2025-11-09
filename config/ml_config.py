class MLConfig:
    """
    Configuración de parámetros del sistema de reconocimiento facial
    """

    # ============ EIGENFACES ============
    EIGENFACES_COMPONENTS = 150  # Componentes PCA
    EIGENFACES_WHITEN = True  # Usar whitening en PCA
    EIGENFACES_THRESHOLD = 0.65  # Umbral de confianza

    # ============ LBP ============
    LBP_RADIUS = 2  # Radio para LBP
    LBP_POINTS = 16  # Puntos de muestreo
    LBP_GRID_SIZE = (8, 8)  # Grilla de regiones
    LBP_THRESHOLD = 0.70  # Umbral de confianza

    # ============ SISTEMA HÍBRIDO ============
    FUSION_METHOD = 'adaptive'  # adaptive, weighted, voting
    EIGENFACES_WEIGHT = 0.6  # Peso para Eigenfaces
    LBP_WEIGHT = 0.4  # Peso para LBP

    # ============ CALIDAD DE IMAGEN ============
    MIN_QUALITY_SCORE = 40  # Score mínimo aceptable
    USE_QUALITY_CHECK = True  # Activar verificación de calidad

    # ============ ALINEACIÓN FACIAL ============
    USE_FACE_ALIGNMENT = True  # Activar alineación facial
    SAVE_ALIGNED_IMAGES = True  # Guardar versiones alineadas

    # ============ PREPROCESAMIENTO ============
    USE_ADVANCED_ILLUMINATION = True  # Normalización avanzada de iluminación
    TARGET_IMAGE_SIZE = (100, 100)  # Tamaño objetivo

    # ============ UMBRALES GENERALES ============
    CONFIDENCE_THRESHOLD = 70.0  # Umbral global de confianza
    CONSENSUS_BONUS = 1.1  # Bonus cuando algoritmos coinciden
    CONFLICT_PENALTY = 0.85  # Penalización cuando no coinciden

    # ============ DATA AUGMENTATION ============
    USE_AUGMENTATION = True  # Activar augmentación
    AUGMENTATION_ROTATIONS = [-5, 5]  # Rotaciones en grados
    AUGMENTATION_SCALES = [0.95, 1.05]  # Escalas

    @classmethod
    def get_config_summary(cls) -> dict:
        """Retorna resumen de configuración actual"""
        return {
            "eigenfaces": {
                "components": cls.EIGENFACES_COMPONENTS,
                "whiten": cls.EIGENFACES_WHITEN,
                "threshold": cls.EIGENFACES_THRESHOLD
            },
            "lbp": {
                "radius": cls.LBP_RADIUS,
                "points": cls.LBP_POINTS,
                "grid_size": cls.LBP_GRID_SIZE,
                "threshold": cls.LBP_THRESHOLD
            },
            "hybrid": {
                "fusion_method": cls.FUSION_METHOD,
                "eigenfaces_weight": cls.EIGENFACES_WEIGHT,
                "lbp_weight": cls.LBP_WEIGHT
            },
            "quality": {
                "min_score": cls.MIN_QUALITY_SCORE,
                "use_quality_check": cls.USE_QUALITY_CHECK
            },
            "alignment": {
                "use_face_alignment": cls.USE_FACE_ALIGNMENT,
                "save_aligned": cls.SAVE_ALIGNED_IMAGES
            }
        }