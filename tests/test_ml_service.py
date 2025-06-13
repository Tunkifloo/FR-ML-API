import pytest
import numpy as np
import cv2
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ml_service import MLService
from services.eigenfaces_service import EigenfacesService
from services.lbp_service import LBPService
from services.face_detection_service import FaceDetectionService


class TestMLService:
    """Pruebas para el servicio principal de ML"""

    @pytest.fixture
    def ml_service(self):
        """Instancia del servicio ML para pruebas"""
        return MLService()

    @pytest.fixture
    def sample_images(self):
        """Generar imágenes de prueba"""
        images = {}

        # Crear imágenes sintéticas para 3 personas
        for person_id in [1, 2, 3]:
            person_images = []
            for i in range(3):  # 3 imágenes por persona
                # Crear imagen con patrón único por persona
                img = np.random.randint(50 + person_id * 30, 100 + person_id * 30,
                                        (100, 100, 3), dtype=np.uint8)

                # Añadir algo de estructura facial
                cv2.rectangle(img, (30, 30), (70, 80), (255, 255, 255), -1)  # Cara
                cv2.circle(img, (40, 45), 5, (0, 0, 0), -1)  # Ojo izquierdo
                cv2.circle(img, (60, 45), 5, (0, 0, 0), -1)  # Ojo derecho
                cv2.rectangle(img, (45, 65), (55, 70), (0, 0, 0), -1)  # Boca

                person_images.append(img)

            images[person_id] = person_images

        return images

    def test_ml_service_initialization(self, ml_service):
        """Probar inicialización del servicio ML"""
        assert ml_service is not None
        assert hasattr(ml_service, 'eigenfaces_service')
        assert hasattr(ml_service, 'lbp_service')
        assert hasattr(ml_service, 'face_detector')
        assert not ml_service.is_trained

    def test_preprocess_image_for_training(self, ml_service, sample_images):
        """Probar preprocesamiento de imagen"""
        test_image = sample_images[1][0]

        # Mock del detector de rostros para que siempre encuentre un rostro
        with patch.object(ml_service.face_detector, 'detect_faces') as mock_detect:
            mock_detect.return_value = [(10, 10, 80, 80)]  # (x, y, w, h)

            with patch.object(ml_service.face_detector, 'get_largest_face') as mock_largest:
                mock_largest.return_value = (10, 10, 80, 80)

                with patch.object(ml_service.face_detector, 'extract_face_roi') as mock_roi:
                    mock_roi.return_value = test_image[10:90, 10:90]

                    with patch.object(ml_service.face_detector, 'enhance_face_image') as mock_enhance:
                        mock_enhance.return_value = test_image[10:90, 10:90]

                        result = ml_service.preprocess_image_for_training(test_image)

                        assert result is not None
                        assert isinstance(result, np.ndarray)

    def test_train_models(self, ml_service, sample_images):
        """Probar entrenamiento de modelos"""
        # Mock de los métodos de detección facial
        with patch.object(ml_service, 'preprocess_image_for_training') as mock_preprocess:
            # Simular que el preprocesamiento siempre devuelve la imagen
            mock_preprocess.side_effect = lambda img: img

            with patch.object(ml_service, '_generate_and_save_embeddings'):
                # Entrenar con imágenes de muestra
                stats = ml_service.train_models(sample_images)

                assert stats is not None
                assert "total_images" in stats
                assert "unique_persons" in stats
                assert stats["unique_persons"] == 3
                assert ml_service.is_trained

    def test_add_new_person(self, ml_service, sample_images):
        """Probar añadir nueva persona al modelo"""
        # Primero entrenar el modelo
        with patch.object(ml_service, 'preprocess_image_for_training') as mock_preprocess:
            mock_preprocess.side_effect = lambda img: img

            with patch.object(ml_service, '_generate_and_save_embeddings'):
                ml_service.train_models(sample_images)

                # Añadir nueva persona
                new_person_images = sample_images[1]  # Usar imágenes existentes
                stats = ml_service.add_new_person(99, new_person_images)

                assert stats is not None
                assert stats["person_id"] == 99
                assert stats["images_processed"] == len(new_person_images)

    def test_get_system_info(self, ml_service):
        """Probar obtención de información del sistema"""
        info = ml_service.get_system_info()

        assert info is not None
        assert "system_info" in info
        assert "eigenfaces_info" in info
        assert "lbp_info" in info
        assert "weights" in info


class TestEigenfacesService:
    """Pruebas para el servicio Eigenfaces"""

    @pytest.fixture
    def eigenfaces_service(self):
        """Instancia del servicio Eigenfaces"""
        return EigenfacesService(n_components=10)  # Pocos componentes para pruebas

    @pytest.fixture
    def training_data(self):
        """Datos de entrenamiento sintéticos"""
        images = []
        labels = []

        # Generar imágenes sintéticas
        for person_id in [1, 2, 3]:
            for i in range(5):
                # Imagen con patrón único por persona
                img = np.random.randint(person_id * 50, person_id * 50 + 100,
                                        (50, 50), dtype=np.uint8)
                images.append(img)
                labels.append(person_id)

        return images, labels

    def test_eigenfaces_initialization(self, eigenfaces_service):
        """Probar inicialización de Eigenfaces"""
        assert eigenfaces_service.n_components == 10
        assert eigenfaces_service.image_size == (100, 100)
        assert not eigenfaces_service.is_trained

    def test_preprocess_image(self, eigenfaces_service):
        """Probar preprocesamiento de imagen"""
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)

        processed = eigenfaces_service.preprocess_image(test_image)

        assert processed.shape == eigenfaces_service.image_size
        assert processed.dtype == np.float64
        assert 0 <= processed.max() <= 1

    def test_eigenfaces_training(self, eigenfaces_service, training_data):
        """Probar entrenamiento de Eigenfaces"""
        images, labels = training_data

        eigenfaces_service.train(images, labels)

        assert eigenfaces_service.is_trained
        assert eigenfaces_service.mean_face is not None
        assert eigenfaces_service.eigenfaces is not None
        assert len(eigenfaces_service.trained_embeddings) == len(images)

    def test_extract_features(self, eigenfaces_service, training_data):
        """Probar extracción de características"""
        images, labels = training_data

        # Entrenar primero
        eigenfaces_service.train(images, labels)

        # Extraer características de una imagen nueva
        test_image = images[0]
        features = eigenfaces_service.extract_features(test_image)

        assert features is not None
        assert len(features) == eigenfaces_service.n_components
        assert isinstance(features, np.ndarray)

    def test_recognize_face(self, eigenfaces_service, training_data):
        """Probar reconocimiento de rostro"""
        images, labels = training_data

        # Entrenar
        eigenfaces_service.train(images, labels)

        # Reconocer una imagen de entrenamiento
        test_image = images[0]
        expected_label = labels[0]

        person_id, confidence, details = eigenfaces_service.recognize_face(test_image)

        assert person_id == expected_label or person_id == -1  # Puede no reconocer con datos sintéticos
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100
        assert "distance" in details
        assert "algorithm" in details


class TestLBPService:
    """Pruebas para el servicio LBP"""

    @pytest.fixture
    def lbp_service(self):
        """Instancia del servicio LBP"""
        return LBPService(radius=1, n_points=8, grid_size=(4, 4))  # Parámetros pequeños para pruebas

    @pytest.fixture
    def training_data(self):
        """Datos de entrenamiento para LBP"""
        images = []
        labels = []

        for person_id in [1, 2, 3]:
            for i in range(3):
                # Crear imagen con texturas diferentes por persona
                img = np.zeros((60, 60), dtype=np.uint8)

                # Añadir patrón único por persona
                if person_id == 1:
                    img[10:50, 10:50] = 100
                    img[20:40, 20:40] = 200
                elif person_id == 2:
                    img[::2, :] = 150  # Líneas horizontales
                else:
                    img[:, ::2] = 180  # Líneas verticales

                images.append(img)
                labels.append(person_id)

        return images, labels

    def test_lbp_initialization(self, lbp_service):
        """Probar inicialización de LBP"""
        assert lbp_service.radius == 1
        assert lbp_service.n_points == 8
        assert lbp_service.grid_size == (4, 4)
        assert not lbp_service.is_trained

    def test_extract_lbp_features(self, lbp_service):
        """Probar extracción de características LBP"""
        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (60, 60), dtype=np.uint8)

        features = lbp_service.extract_lbp_features(test_image)

        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) > 0

        # El tamaño debe ser grid_size[0] * grid_size[1] * (n_points + 2)
        expected_size = 4 * 4 * (8 + 2)  # 4x4 grid, 8 points + 2 bins
        assert len(features) == expected_size

    def test_lbp_training(self, lbp_service, training_data):
        """Probar entrenamiento de LBP"""
        images, labels = training_data

        lbp_service.train(images, labels)

        assert lbp_service.is_trained
        assert len(lbp_service.trained_histograms) == len(images)
        assert len(lbp_service.trained_labels) == len(labels)

    def test_lbp_recognition(self, lbp_service, training_data):
        """Probar reconocimiento con LBP"""
        images, labels = training_data

        # Entrenar
        lbp_service.train(images, labels)

        # Reconocer imagen de entrenamiento
        test_image = images[0]
        expected_label = labels[0]

        person_id, confidence, details = lbp_service.recognize_face(test_image)

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100
        assert "similarity" in details
        assert "algorithm" in details
        assert details["algorithm"] == "lbp"


class TestFaceDetectionService:
    """Pruebas para el servicio de detección facial"""

    @pytest.fixture
    def face_detector(self):
        """Instancia del detector de rostros"""
        return FaceDetectionService()

    @pytest.fixture
    def face_image(self):
        """Imagen sintética con rostro"""
        # Crear imagen con forma de rostro básica
        img = np.zeros((200, 200, 3), dtype=np.uint8)

        # Cara (rectángulo)
        cv2.rectangle(img, (50, 50), (150, 180), (200, 200, 200), -1)

        # Ojos
        cv2.circle(img, (80, 90), 8, (0, 0, 0), -1)
        cv2.circle(img, (120, 90), 8, (0, 0, 0), -1)

        # Nariz
        cv2.rectangle(img, (95, 110), (105, 130), (150, 150, 150), -1)

        # Boca
        cv2.rectangle(img, (85, 150), (115, 160), (0, 0, 0), -1)

        return img

    def test_face_detection_initialization(self, face_detector):
        """Probar inicialización del detector"""
        assert face_detector.face_cascade is not None
        assert face_detector.scale_factor == 1.1
        assert face_detector.min_neighbors == 5

    def test_detect_faces(self, face_detector, face_image):
        """Probar detección de rostros"""
        faces = face_detector.detect_faces(face_image)

        # Puede o no detectar el rostro sintético con Haar Cascades
        assert isinstance(faces, list)
        # No aseguramos detección porque Haar Cascades puede no reconocer rostros sintéticos

    def test_extract_face_roi(self, face_detector, face_image):
        """Probar extracción de ROI"""
        # Simular coordenadas de rostro detectado
        face_coords = (50, 50, 100, 130)  # x, y, w, h

        roi = face_detector.extract_face_roi(face_image, face_coords)

        assert roi is not None
        assert isinstance(roi, np.ndarray)
        assert len(roi.shape) == 3  # Imagen en color

    def test_get_largest_face(self, face_detector):
        """Probar obtención del rostro más grande"""
        faces = [(10, 10, 50, 50), (100, 100, 80, 80), (200, 200, 30, 30)]

        largest = face_detector.get_largest_face(faces)

        assert largest == (100, 100, 80, 80)  # 80*80 = 6400 es el más grande

    def test_get_largest_face_empty(self, face_detector):
        """Probar obtención del rostro más grande con lista vacía"""
        largest = face_detector.get_largest_face([])
        assert largest is None

    def test_calculate_face_quality(self, face_detector, face_image):
        """Probar cálculo de calidad facial"""
        # Extraer ROI simulado
        face_roi = face_image[50:180, 50:150]

        quality = face_detector._calculate_face_quality(face_roi)

        assert isinstance(quality, float)
        assert 0 <= quality <= 100


class TestMLIntegration:
    """Pruebas de integración entre servicios ML"""

    @pytest.fixture
    def integration_setup(self):
        """Configuración para pruebas de integración"""
        ml_service = MLService()

        # Datos de prueba
        images_by_person = {}
        for person_id in [1, 2]:
            person_images = []
            for i in range(2):
                # Crear imagen sintética
                img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
                person_images.append(img)
            images_by_person[person_id] = person_images

        return ml_service, images_by_person

    def test_full_training_pipeline(self, integration_setup):
        """Probar pipeline completo de entrenamiento"""
        ml_service, images_by_person = integration_setup

        # Mock de preprocesamiento para evitar dependencias externas
        with patch.object(ml_service, 'preprocess_image_for_training') as mock_preprocess:
            mock_preprocess.side_effect = lambda img: cv2.resize(img, (100, 100))

            with patch.object(ml_service, '_generate_and_save_embeddings'):
                stats = ml_service.train_models(images_by_person)

                assert stats is not None
                assert ml_service.is_trained
                assert ml_service.eigenfaces_service.is_trained
                assert ml_service.lbp_service.is_trained

    def test_recognition_pipeline(self, integration_setup):
        """Probar pipeline de reconocimiento"""
        ml_service, images_by_person = integration_setup

        # Entrenar primero
        with patch.object(ml_service, 'preprocess_image_for_training') as mock_preprocess:
            mock_preprocess.side_effect = lambda img: cv2.resize(img, (100, 100))

            with patch.object(ml_service, '_generate_and_save_embeddings'):
                ml_service.train_models(images_by_person)

        # Probar reconocimiento
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with patch.object(ml_service, 'preprocess_image_for_training') as mock_preprocess:
            mock_preprocess.return_value = cv2.resize(test_image, (100, 100))

            result = ml_service.recognize_face(test_image, method="hybrid")

            assert result is not None
            assert "recognized" in result
            assert "confidence" in result
            assert "method" in result


class TestErrorHandling:
    """Pruebas de manejo de errores"""

    def test_eigenfaces_without_training(self):
        """Probar Eigenfaces sin entrenar"""
        service = EigenfacesService()

        with pytest.raises(ValueError):
            service.extract_features(np.zeros((100, 100)))

    def test_lbp_without_training(self):
        """Probar LBP sin entrenar"""
        service = LBPService()

        with pytest.raises(ValueError):
            service.recognize_face(np.zeros((100, 100)))

    def test_ml_service_without_training(self):
        """Probar MLService sin entrenar"""
        service = MLService()

        with pytest.raises(ValueError):
            service.recognize_face(np.zeros((100, 100, 3)))

    def test_invalid_image_dimensions(self):
        """Probar con dimensiones de imagen inválidas"""
        service = EigenfacesService()

        # Imagen vacía
        with pytest.raises(Exception):
            service.preprocess_image(np.array([]))

    def test_empty_training_data(self):
        """Probar entrenamiento con datos vacíos"""
        service = EigenfacesService()

        with pytest.raises(Exception):
            service.train([], [])


class TestPerformanceMetrics:
    """Pruebas de métricas de rendimiento"""

    def test_model_info_structure(self):
        """Probar estructura de información del modelo"""
        eigenfaces_service = EigenfacesService()
        info = eigenfaces_service.get_model_info()

        required_keys = [
            "algorithm", "is_trained", "n_components",
            "image_size", "total_embeddings", "unique_persons"
        ]

        for key in required_keys:
            assert key in info

    def test_benchmark_structure(self):
        """Probar estructura de benchmark"""
        ml_service = MLService()

        # Mock de datos de prueba
        test_data = [
            (np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), 1),
            (np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), 2)
        ]

        # Mock de servicios entrenados
        with patch.object(ml_service, 'is_trained', True):
            with patch.object(ml_service, '_recognize_eigenfaces_only') as mock_eigen:
                mock_eigen.return_value = {"person_id": 1, "confidence": 85}

                with patch.object(ml_service, '_recognize_lbp_only') as mock_lbp:
                    mock_lbp.return_value = {"person_id": 1, "confidence": 80}

                    with patch.object(ml_service, '_recognize_hybrid') as mock_hybrid:
                        mock_hybrid.return_value = {"person_id": 1, "confidence": 90}

                        results = ml_service.benchmark_algorithms(test_data)

                        assert "eigenfaces" in results
                        assert "lbp" in results
                        assert "hybrid" in results

                        for method in results:
                            assert "accuracy" in results[method]
                            assert "average_confidence" in results[method]


# Ejecutar pruebas
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])