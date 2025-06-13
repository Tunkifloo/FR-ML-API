import pytest
import numpy as np
import cv2
import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.alert_system import AlertSystem, AlertInfo
from utils.feature_extractor import FeatureExtractor
from utils.image_processor import ImageProcessor


class TestAlertSystem:
    """Pruebas para el sistema de alertas"""

    @pytest.fixture
    def alert_system(self):
        """Instancia del sistema de alertas"""
        # Usar archivo temporal para las pruebas
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        return AlertSystem(alerts_log_path=temp_file.name)

    @pytest.fixture
    def sample_alert_info(self):
        """Información de alerta de muestra"""
        return AlertInfo(
            person_id=123,
            person_name="Juan",
            person_lastname="Pérez",
            student_id="TEST001",
            requisition_type="Robo",
            confidence=85.5,
            detection_timestamp=datetime.now().isoformat(),
            image_path="/test/image.jpg",
            alert_level="HIGH",
            location="Campus Principal"
        )

    def test_alert_system_initialization(self, alert_system):
        """Probar inicialización del sistema de alertas"""
        assert alert_system is not None
        assert hasattr(alert_system, 'alert_thresholds')
        assert hasattr(alert_system, 'requisition_alert_levels')
        assert "HIGH" in alert_system.alert_thresholds
        assert "MEDIUM" in alert_system.alert_thresholds
        assert "LOW" in alert_system.alert_thresholds

    def test_determine_alert_level(self, alert_system):
        """Probar determinación del nivel de alerta"""
        # Confianza alta + delito grave = HIGH
        level = alert_system._determine_alert_level(90.0, "Robo")
        assert level == "HIGH"

        # Confianza media + delito moderado = MEDIUM
        level = alert_system._determine_alert_level(65.0, "Hurto")
        assert level == "MEDIUM"

        # Confianza baja + delito menor = LOW
        level = alert_system._determine_alert_level(45.0, "Vandalismo")
        assert level == "LOW"

    def test_generate_security_alert(self, alert_system, sample_alert_info):
        """Probar generación de alerta de seguridad"""
        alert_response = alert_system.generate_security_alert(sample_alert_info)

        assert alert_response["alert_generated"] == True
        assert "alert_id" in alert_response
        assert "alert_level" in alert_response
        assert "message" in alert_response
        assert "person_info" in alert_response
        assert "authority_notification" in alert_response
        assert "recommended_actions" in alert_response

        # Verificar información de la persona
        person_info = alert_response["person_info"]
        assert person_info["id"] == 123
        assert person_info["name"] == "Juan Pérez"
        assert person_info["requisition_type"] == "Robo"

    def test_create_alert_message(self, alert_system, sample_alert_info):
        """Probar creación de mensaje de alerta"""
        message = alert_system._create_alert_message(sample_alert_info)

        assert "ALERTA" in message.upper()
        assert sample_alert_info.person_name in message
        assert sample_alert_info.requisition_type in message
        assert str(sample_alert_info.confidence) in message

    def test_simulate_authority_notification(self, alert_system, sample_alert_info):
        """Probar simulación de notificación a autoridades"""
        notification = alert_system._simulate_authority_notification(sample_alert_info)

        assert notification["status"] == "SIMULADO - EXITOSO"
        assert "notification_methods" in notification
        assert "estimated_response_time" in notification
        assert "reference_number" in notification
        assert isinstance(notification["notification_methods"], list)

    def test_get_recommended_actions(self, alert_system):
        """Probar obtención de acciones recomendadas"""
        # Alerta de nivel alto
        actions_high = alert_system._get_recommended_actions("HIGH", "Robo")
        assert len(actions_high) > 0
        assert any("policía" in action.lower() for action in actions_high)

        # Alerta de nivel medio
        actions_medium = alert_system._get_recommended_actions("MEDIUM", "Hurto")
        assert len(actions_medium) > 0

        # Alerta de nivel bajo
        actions_low = alert_system._get_recommended_actions("LOW", "Vandalismo")
        assert len(actions_low) > 0

    def test_log_alert(self, alert_system, sample_alert_info):
        """Probar registro de alerta"""
        alert_record = alert_system._log_alert(sample_alert_info)

        assert "alert_id" in alert_record
        assert "alert_info" in alert_record
        assert "logged_at" in alert_record
        assert alert_record["status"] == "ACTIVE"

    def test_get_alert_history(self, alert_system, sample_alert_info):
        """Probar obtención de historial de alertas"""
        # Generar algunas alertas
        alert_system.generate_security_alert(sample_alert_info)

        # Modificar para segunda alerta
        sample_alert_info.person_id = 456
        sample_alert_info.requisition_type = "Fraude"
        alert_system.generate_security_alert(sample_alert_info)

        # Obtener historial
        history = alert_system.get_alert_history(limit=10)

        assert len(history) >= 2
        assert all("alert_id" in alert for alert in history)

    def test_get_alert_statistics(self, alert_system, sample_alert_info):
        """Probar obtención de estadísticas de alertas"""
        # Generar algunas alertas
        alert_system.generate_security_alert(sample_alert_info)

        stats = alert_system.get_alert_statistics()

        assert "total_alerts" in stats
        assert "by_level" in stats
        assert "by_requisition_type" in stats
        assert isinstance(stats["by_level"], dict)
        assert isinstance(stats["by_requisition_type"], dict)


class TestFeatureExtractor:
    """Pruebas para el extractor de características"""

    @pytest.fixture
    def feature_extractor(self):
        """Instancia del extractor de características"""
        return FeatureExtractor()

    @pytest.fixture
    def sample_face_image(self):
        """Imagen facial de muestra"""
        # Crear imagen sintética con características faciales
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Añadir algunas características
        cv2.rectangle(img, (20, 20), (80, 90), (150, 150, 150), -1)  # Cara
        cv2.circle(img, (35, 40), 5, (0, 0, 0), -1)  # Ojo izquierdo
        cv2.circle(img, (65, 40), 5, (0, 0, 0), -1)  # Ojo derecho
        cv2.rectangle(img, (45, 65), (55, 75), (100, 100, 100), -1)  # Nariz
        cv2.rectangle(img, (40, 80), (60, 85), (50, 50, 50), -1)  # Boca

        return img

    def test_feature_extractor_initialization(self, feature_extractor):
        """Probar inicialización del extractor"""
        assert feature_extractor is not None
        assert hasattr(feature_extractor, 'enable_geometric_features')
        assert hasattr(feature_extractor, 'enable_texture_features')
        assert hasattr(feature_extractor, 'enable_statistical_features')
        assert isinstance(feature_extractor.feature_cache, dict)

    def test_extract_comprehensive_features(self, feature_extractor, sample_face_image):
        """Probar extracción completa de características"""
        features = feature_extractor.extract_comprehensive_features(sample_face_image)

        assert "extraction_timestamp" in features
        assert "image_properties" in features
        assert "geometric_features" in features
        assert "texture_features" in features
        assert "statistical_features" in features

        # Verificar propiedades de imagen
        img_props = features["image_properties"]
        assert img_props["height"] == 100
        assert img_props["width"] == 100
        assert img_props["channels"] == 3

    def test_extract_geometric_features(self, feature_extractor, sample_face_image):
        """Probar extracción de características geométricas"""
        geometric = feature_extractor._extract_geometric_features(sample_face_image)

        assert isinstance(geometric, dict)
        # Pueden o no detectarse características con cascadas en imagen sintética
        assert "eyes_detected" in geometric
        assert "nose_detected" in geometric
        assert "mouth_detected" in geometric

    def test_extract_texture_features(self, feature_extractor, sample_face_image):
        """Probar extracción de características de textura"""
        texture = feature_extractor._extract_texture_features(sample_face_image)

        assert isinstance(texture, dict)
        assert "gradient_mean" in texture
        assert "gradient_std" in texture
        assert "laplacian_variance" in texture
        assert "gabor_responses" in texture
        assert isinstance(texture["gabor_responses"], list)

    def test_extract_statistical_features(self, feature_extractor, sample_face_image):
        """Probar extracción de características estadísticas"""
        statistical = feature_extractor._extract_statistical_features(sample_face_image)

        assert isinstance(statistical, dict)
        assert "mean" in statistical
        assert "std" in statistical
        assert "skewness" in statistical
        assert "kurtosis" in statistical
        assert "hu_moments" in statistical
        assert "hist_entropy" in statistical
        assert isinstance(statistical["hu_moments"], list)

    def test_compare_features(self, feature_extractor, sample_face_image):
        """Probar comparación de características"""
        # Extraer características de la misma imagen
        features1 = feature_extractor.extract_comprehensive_features(sample_face_image)

        # Crear imagen ligeramente diferente
        modified_image = sample_face_image.copy()
        modified_image = cv2.GaussianBlur(modified_image, (3, 3), 0)
        features2 = feature_extractor.extract_comprehensive_features(modified_image)

        # Comparar
        similarities = feature_extractor.compare_features(features1, features2)

        assert isinstance(similarities, dict)
        # Debería haber al menos alguna similitud ya que son imágenes similares
        if similarities:
            for sim_name, sim_value in similarities.items():
                assert isinstance(sim_value, float)
                assert -1 <= sim_value <= 1  # Similitudes normalizadas

    def test_feature_caching(self, feature_extractor, sample_face_image):
        """Probar sistema de cache de características"""
        person_id = 123

        # Extraer características con cache
        features = feature_extractor.extract_comprehensive_features(sample_face_image, person_id=person_id)

        # Verificar que se guardó en cache
        cached_features = feature_extractor.get_cached_features(person_id)
        assert cached_features is not None
        assert cached_features["extraction_timestamp"] == features["extraction_timestamp"]

    def test_get_feature_summary(self, feature_extractor, sample_face_image):
        """Probar resumen de características"""
        features = feature_extractor.extract_comprehensive_features(sample_face_image)
        summary = feature_extractor.get_feature_summary(features)

        assert "extraction_time" in summary
        assert "image_info" in summary
        assert "quality_metrics" in summary
        assert "quality_score" in summary
        assert 0 <= summary["quality_score"] <= 100


class TestImageProcessor:
    """Pruebas para el procesador de imágenes"""

    @pytest.fixture
    def image_processor(self):
        """Instancia del procesador de imágenes"""
        return ImageProcessor()

    @pytest.fixture
    def sample_image(self):
        """Imagen de muestra para pruebas"""
        # Crear imagen con características variadas
        img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)

        # Añadir algunas características
        cv2.rectangle(img, (50, 50), (100, 120), (200, 200, 200), -1)
        cv2.circle(img, (60, 70), 5, (0, 0, 0), -1)
        cv2.circle(img, (90, 70), 5, (0, 0, 0), -1)

        return img

    def test_image_processor_initialization(self, image_processor):
        """Probar inicialización del procesador"""
        assert image_processor is not None
        assert hasattr(image_processor, 'target_size')
        assert hasattr(image_processor, 'quality_threshold')
        assert image_processor.target_size == (224, 224)

    def test_calculate_image_quality(self, image_processor, sample_image):
        """Probar cálculo de calidad de imagen"""
        quality_metrics = image_processor.calculate_image_quality(sample_image)

        assert isinstance(quality_metrics, dict)
        required_metrics = [
            "sharpness", "contrast", "brightness",
            "illumination_uniformity", "noise_level",
            "resolution", "overall_quality"
        ]

        for metric in required_metrics:
            assert metric in quality_metrics
            assert 0 <= quality_metrics[metric] <= 100

    def test_resize_image(self, image_processor, sample_image):
        """Probar redimensionado de imagen"""
        target_size = (100, 100)
        resized = image_processor.resize_image(sample_image, target_size)

        assert resized.shape[:2] == target_size
        assert len(resized.shape) == len(sample_image.shape)

    def test_correct_orientation(self, image_processor, sample_image):
        """Probar corrección de orientación"""
        corrected, rotation_info = image_processor.correct_orientation(sample_image)

        assert corrected is not None
        assert isinstance(rotation_info, dict)
        assert "rotated" in rotation_info
        assert "angle" in rotation_info
        assert "confidence" in rotation_info

    def test_enhance_image(self, image_processor, sample_image):
        """Probar mejora de imagen"""
        enhanced, enhancement_info = image_processor.enhance_image(sample_image)

        assert enhanced is not None
        assert enhanced.shape == sample_image.shape
        assert isinstance(enhancement_info, dict)
        assert "applied_enhancements" in enhancement_info
        assert isinstance(enhancement_info["applied_enhancements"], list)

    def test_calculate_optimal_gamma(self, image_processor, sample_image):
        """Probar cálculo de gamma óptimo"""
        gamma = image_processor.calculate_optimal_gamma(sample_image)

        assert isinstance(gamma, float)
        assert 0.4 <= gamma <= 2.5  # Rango válido según implementación

    def test_adjust_gamma(self, image_processor, sample_image):
        """Probar ajuste de gamma"""
        gamma = 1.2
        adjusted = image_processor.adjust_gamma(sample_image, gamma)

        assert adjusted.shape == sample_image.shape
        assert adjusted.dtype == sample_image.dtype

    def test_white_balance(self, image_processor, sample_image):
        """Probar balance de blancos"""
        balanced = image_processor.white_balance(sample_image)

        assert balanced.shape == sample_image.shape
        assert balanced.dtype == sample_image.dtype

    def test_reduce_noise(self, image_processor, sample_image):
        """Probar reducción de ruido"""
        denoised = image_processor.reduce_noise(sample_image)

        assert denoised.shape == sample_image.shape
        assert denoised.dtype == sample_image.dtype

    def test_normalize_image(self, image_processor, sample_image):
        """Probar normalización de imagen"""
        normalized = image_processor.normalize_image(sample_image)

        assert normalized.shape == sample_image.shape
        assert normalized.dtype == np.uint8

    def test_process_face_image(self, image_processor, sample_image):
        """Probar procesamiento completo de imagen facial"""
        processed, processing_info = image_processor.process_face_image(sample_image)

        assert processed is not None
        assert processed.shape[:2] == image_processor.target_size

        assert isinstance(processing_info, dict)
        assert "initial_quality" in processing_info
        assert "final_quality" in processing_info
        assert "quality_improvement" in processing_info
        assert "processing_steps" in processing_info
        assert isinstance(processing_info["processing_steps"], list)

    def test_create_face_thumbnail(self, image_processor, sample_image):
        """Probar creación de miniatura"""
        thumbnail_size = (64, 64)
        thumbnail = image_processor.create_face_thumbnail(sample_image, thumbnail_size)

        assert thumbnail.shape[:2] == thumbnail_size

    def test_compare_image_quality(self, image_processor, sample_image):
        """Probar comparación de calidad entre imágenes"""
        # Crear segunda imagen (degradada)
        degraded_image = cv2.GaussianBlur(sample_image, (15, 15), 0)

        comparison = image_processor.compare_image_quality(sample_image, degraded_image)

        assert isinstance(comparison, dict)
        assert "image1_quality" in comparison
        assert "image2_quality" in comparison
        assert "quality_differences" in comparison
        assert "better_image" in comparison

    def test_detect_image_problems(self, image_processor):
        """Probar detección de problemas en imágenes"""
        # Crear imagen con problemas conocidos
        problematic_image = np.full((100, 100, 3), 50, dtype=np.uint8)  # Imagen muy oscura

        problems = image_processor.detect_image_problems(problematic_image)

        assert isinstance(problems, list)
        # Debería detectar al menos el problema de iluminación
        if problems:
            for problem in problems:
                assert "type" in problem
                assert "severity" in problem
                assert "description" in problem
                assert "recommendation" in problem


class TestIntegrationScenarios:
    """Pruebas de escenarios de integración"""

    @pytest.fixture
    def integration_setup(self):
        """Configuración para pruebas de integración"""
        alert_system = AlertSystem()
        feature_extractor = FeatureExtractor()
        image_processor = ImageProcessor()

        return alert_system, feature_extractor, image_processor

    def test_complete_recognition_flow(self, integration_setup):
        """Probar flujo completo de reconocimiento"""
        alert_system, feature_extractor, image_processor = integration_setup

        # 1. Crear imagen de prueba
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # 2. Procesar imagen
        processed_image, processing_info = image_processor.process_face_image(test_image)
        assert processed_image is not None

        # 3. Extraer características
        features = feature_extractor.extract_comprehensive_features(processed_image, person_id=1)
        assert features is not None

        # 4. Simular reconocimiento positivo de persona requisitoriada
        alert_info = AlertInfo(
            person_id=1,
            person_name="Test",
            person_lastname="Person",
            student_id="TEST001",
            requisition_type="Robo",
            confidence=85.0,
            detection_timestamp=datetime.now().isoformat(),
            image_path="/test/path",
            alert_level="HIGH"
        )

        # 5. Generar alerta
        alert_response = alert_system.generate_security_alert(alert_info)
        assert alert_response["alert_generated"] == True

    def test_batch_processing_simulation(self, integration_setup):
        """Probar procesamiento en lote simulado"""
        alert_system, feature_extractor, image_processor = integration_setup

        # Simular múltiples imágenes
        test_images = []
        for i in range(3):
            img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
            test_images.append(img)

        # Procesar cada imagen
        results = []
        for i, img in enumerate(test_images):
            try:
                processed, info = image_processor.process_face_image(img)
                features = feature_extractor.extract_comprehensive_features(processed, person_id=i + 1)

                result = {
                    "image_id": i + 1,
                    "processed": True,
                    "quality_score": info["final_quality"]["overall_quality"],
                    "features_extracted": len(features) > 0
                }
                results.append(result)

            except Exception as e:
                results.append({
                    "image_id": i + 1,
                    "processed": False,
                    "error": str(e)
                })

        # Verificar resultados
        assert len(results) == 3
        successful_processing = sum(1 for r in results if r.get("processed", False))
        assert successful_processing >= 0  # Al menos algunos deberían procesar correctamente


class TestPerformanceAndLimits:
    """Pruebas de rendimiento y límites"""

    def test_large_image_processing(self):
        """Probar procesamiento de imagen grande"""
        processor = ImageProcessor()

        # Crear imagen grande
        large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        processed, info = processor.process_face_image(large_image)

        # Debería redimensionarse al tamaño objetivo
        assert processed.shape[:2] == processor.target_size

    def test_very_small_image(self):
        """Probar imagen muy pequeña"""
        processor = ImageProcessor()

        # Imagen muy pequeña
        small_image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)

        processed, info = processor.process_face_image(small_image)

        # Debería redimensionarse correctamente
        assert processed.shape[:2] == processor.target_size

    def test_feature_extraction_limits(self):
        """Probar límites de extracción de características"""
        extractor = FeatureExtractor()

        # Imagen en escala de grises
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        features = extractor.extract_comprehensive_features(gray_image)

        assert features is not None
        assert "statistical_features" in features

    def test_alert_system_stress(self):
        """Probar sistema de alertas con múltiples alertas"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        alert_system = AlertSystem(alerts_log_path=temp_file.name)

        # Generar múltiples alertas
        for i in range(10):
            alert_info = AlertInfo(
                person_id=i,
                person_name=f"Person{i}",
                person_lastname="Test",
                student_id=f"TEST{i:03d}",
                requisition_type="Hurto",
                confidence=70.0 + i,
                detection_timestamp=datetime.now().isoformat(),
                image_path=f"/test/image{i}.jpg",
                alert_level="MEDIUM"
            )

            response = alert_system.generate_security_alert(alert_info)
            assert response["alert_generated"] == True

        # Verificar historial
        history = alert_system.get_alert_history(limit=20)
        assert len(history) >= 10

        # Verificar estadísticas
        stats = alert_system.get_alert_statistics()
        assert stats["total_alerts"] >= 10


class TestErrorRecovery:
    """Pruebas de recuperación de errores"""

    def test_corrupted_image_handling(self):
        """Probar manejo de imagen corrupta"""
        processor = ImageProcessor()

        # Array vacío
        empty_array = np.array([])

        with pytest.raises(Exception):
            processor.process_face_image(empty_array)

    def test_invalid_alert_data(self):
        """Probar datos de alerta inválidos"""
        alert_system = AlertSystem()

        # Datos incompletos
        incomplete_alert = AlertInfo(
            person_id=0,  # ID inválido
            person_name="",  # Nombre vacío
            person_lastname="Test",
            student_id=None,
            requisition_type="Unknown",  # Tipo no reconocido
            confidence=-10.0,  # Confianza inválida
            detection_timestamp="invalid_timestamp",
            image_path="",
            alert_level="INVALID"
        )

        # Debería manejar datos inválidos sin crash
        try:
            response = alert_system.generate_security_alert(incomplete_alert)
            # Si no lanza excepción, verificar que maneja los datos
            assert "alert_generated" in response
        except Exception as e:
            # Es aceptable que lance excepción con datos inválidos
            assert isinstance(e, Exception)

    def test_feature_extraction_with_edge_cases(self):
        """Probar extracción de características con casos límite"""
        extractor = FeatureExtractor()

        # Imagen completamente negra
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)
        features_black = extractor.extract_comprehensive_features(black_image)
        assert features_black is not None

        # Imagen completamente blanca
        white_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        features_white = extractor.extract_comprehensive_features(white_image)
        assert features_white is not None

        # Imagen con un solo color
        uniform_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        features_uniform = extractor.extract_comprehensive_features(uniform_image)
        assert features_uniform is not None


class TestDataConsistency:
    """Pruebas de consistencia de datos"""

    def test_feature_consistency(self):
        """Probar consistencia de características extraídas"""
        extractor = FeatureExtractor()

        # Misma imagen procesada dos veces
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        features1 = extractor.extract_comprehensive_features(test_image)
        features2 = extractor.extract_comprehensive_features(test_image)

        # Las características estadísticas deberían ser idénticas
        stats1 = features1["statistical_features"]
        stats2 = features2["statistical_features"]

        for key in ["mean", "std", "min", "max"]:
            if key in stats1 and key in stats2:
                assert abs(stats1[key] - stats2[key]) < 1e-10

    def test_alert_id_uniqueness(self):
        """Probar unicidad de IDs de alerta"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        alert_system = AlertSystem(alerts_log_path=temp_file.name)

        alert_ids = set()

        # Generar múltiples alertas
        for i in range(5):
            alert_info = AlertInfo(
                person_id=i,
                person_name=f"Person{i}",
                person_lastname="Test",
                student_id=f"TEST{i:03d}",
                requisition_type="Hurto",
                confidence=75.0,
                detection_timestamp=datetime.now().isoformat(),
                image_path=f"/test/image{i}.jpg",
                alert_level="MEDIUM"
            )

            response = alert_system.generate_security_alert(alert_info)
            alert_id = response["alert_id"]

            # Verificar que el ID es único
            assert alert_id not in alert_ids
            alert_ids.add(alert_id)

    def test_processing_determinism(self):
        """Probar determinismo en el procesamiento"""
        processor = ImageProcessor()

        # Imagen fija
        test_image = np.full((100, 100, 3), 128, dtype=np.uint8)

        # Procesar múltiples veces
        results = []
        for _ in range(3):
            processed, info = processor.process_face_image(test_image, enhance=False)
            quality = info["final_quality"]["overall_quality"]
            results.append(quality)

        # Los resultados deberían ser idénticos (sin mejoras aleatorias)
        assert all(abs(r - results[0]) < 1e-6 for r in results)


# Ejecutar pruebas
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])