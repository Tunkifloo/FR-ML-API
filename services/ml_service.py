import numpy as np
import cv2
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from .eigenfaces_service import EigenfacesService
from .lbp_service import LBPService
from .face_detection_service import FaceDetectionService


class MLService:
    """
    Servicio principal de Machine Learning que combina Eigenfaces y LBP
    para reconocimiento facial robusto
    """

    def __init__(self):
        """
        Inicializa el servicio de ML con algoritmos híbridos
        """
        # Inicializar servicios
        self.eigenfaces_service = EigenfacesService(n_components=150)
        self.lbp_service = LBPService(radius=2, n_points=16, grid_size=(8, 8))
        self.face_detector = FaceDetectionService()

        # Estado del modelo
        self.is_trained = False
        self.model_version = "1.0"
        self.training_history = []

        # Configuración de combinación
        self.combination_method = "weighted_average"  # weighted_average, voting, cascade
        self.eigenfaces_weight = 0.6
        self.lbp_weight = 0.4

        # Umbrales
        self.confidence_threshold = 70.0
        self.consensus_threshold = 0.7  # Umbral para consenso entre algoritmos

        # Almacenamiento
        self.storage_path = "storage/embeddings/"
        os.makedirs(self.storage_path, exist_ok=True)

    def preprocess_image_for_training(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocesa una imagen para entrenamiento

        Args:
            image: Imagen original

        Returns:
            Imagen preprocesada del rostro o None si no se detecta rostro
        """
        # Detectar rostros
        faces = self.face_detector.detect_faces(image)

        if not faces:
            return None

        # Obtener el rostro más grande
        largest_face = self.face_detector.get_largest_face(faces)

        # Extraer ROI del rostro
        face_roi = self.face_detector.extract_face_roi(image, largest_face)

        # Mejorar la imagen
        enhanced_face = self.face_detector.enhance_face_image(face_roi)

        return enhanced_face

    def train_models(self, images_by_person: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
        """
        Entrena ambos modelos (Eigenfaces y LBP) con las imágenes proporcionadas

        Args:
            images_by_person: Diccionario {person_id: [list_of_images]}

        Returns:
            Diccionario con estadísticas del entrenamiento
        """
        print("🚀 Iniciando entrenamiento del modelo híbrido...")

        # Preparar datos de entrenamiento
        all_images = []
        all_labels = []

        for person_id, images in images_by_person.items():
            for image in images:
                # Preprocesar imagen
                processed_face = self.preprocess_image_for_training(image)

                if processed_face is not None:
                    all_images.append(processed_face)
                    all_labels.append(person_id)

        if not all_images:
            raise ValueError("No se pudieron procesar imágenes para entrenamiento")

        print(f"📊 Datos de entrenamiento: {len(all_images)} imágenes de {len(set(all_labels))} personas")

        # Entrenar Eigenfaces
        print("🎭 Entrenando modelo Eigenfaces...")
        self.eigenfaces_service.train(all_images, all_labels)

        # Entrenar LBP
        print("🔍 Entrenando modelo LBP...")
        self.lbp_service.train(all_images, all_labels)

        # Guardar modelos
        self.eigenfaces_service.save_model()
        self.lbp_service.save_model()

        # Generar y guardar embeddings
        self._generate_and_save_embeddings(all_images, all_labels)

        self.is_trained = True

        # Estadísticas del entrenamiento
        training_stats = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(all_images),
            "unique_persons": len(set(all_labels)),
            "eigenfaces_info": self.eigenfaces_service.get_model_info(),
            "lbp_info": self.lbp_service.get_model_info(),
            "model_version": self.model_version
        }

        # Guardar historial
        self.training_history.append(training_stats)

        print("✅ Entrenamiento híbrido completado exitosamente!")
        return training_stats

    def add_new_person(self, person_id: int, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Añade una nueva persona al sistema (entrenamiento incremental)

        Args:
            person_id: ID de la nueva persona
            images: Lista de imágenes de la persona

        Returns:
            Estadísticas del proceso
        """
        print(f"➕ Añadiendo nueva persona ID: {person_id}")

        # Preprocesar imágenes
        processed_images = []
        for image in images:
            processed_face = self.preprocess_image_for_training(image)
            if processed_face is not None:
                processed_images.append(processed_face)

        if not processed_images:
            raise ValueError("No se pudieron procesar las imágenes proporcionadas")

        # Añadir a ambos modelos
        self.eigenfaces_service.add_new_person(processed_images, person_id)
        self.lbp_service.add_new_person(processed_images, person_id)

        # Generar embeddings para la nueva persona
        self._generate_and_save_embeddings(processed_images, [person_id] * len(processed_images))

        # Guardar modelos actualizados
        self.eigenfaces_service.save_model()
        self.lbp_service.save_model()

        stats = {
            "person_id": person_id,
            "images_processed": len(processed_images),
            "timestamp": datetime.now().isoformat()
        }

        print(f"✅ Persona {person_id} añadida con {len(processed_images)} imágenes")
        return stats

    def recognize_face(self, image: np.ndarray, method: str = "hybrid") -> Dict[str, Any]:
        """
        Reconoce un rostro usando el método especificado

        Args:
            image: Imagen a procesar
            method: Método a usar ("hybrid", "eigenfaces", "lbp", "voting")

        Returns:
            Diccionario con resultado del reconocimiento
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        # Preprocesar imagen
        processed_face = self.preprocess_image_for_training(image)

        if processed_face is None:
            return {
                "recognized": False,
                "person_id": None,
                "confidence": 0.0,
                "error": "No se detectó rostro en la imagen",
                "timestamp": datetime.now().isoformat()
            }

        # Realizar reconocimiento según el método
        if method == "eigenfaces":
            return self._recognize_eigenfaces_only(processed_face)
        elif method == "lbp":
            return self._recognize_lbp_only(processed_face)
        elif method == "voting":
            return self._recognize_voting(processed_face)
        else:  # hybrid (default)
            return self._recognize_hybrid(processed_face)

    def _recognize_hybrid(self, processed_face: np.ndarray) -> Dict[str, Any]:
        """
        Reconocimiento híbrido combinando Eigenfaces y LBP
        """
        # Reconocimiento con Eigenfaces
        eigen_person_id, eigen_confidence, eigen_details = self.eigenfaces_service.recognize_face(processed_face)

        # Reconocimiento con LBP
        lbp_person_id, lbp_confidence, lbp_details = self.lbp_service.recognize_face(processed_face)

        # Combinar resultados
        if self.combination_method == "weighted_average":
            result = self._combine_weighted_average(
                (eigen_person_id, eigen_confidence, eigen_details),
                (lbp_person_id, lbp_confidence, lbp_details)
            )
        elif self.combination_method == "voting":
            result = self._combine_voting(
                (eigen_person_id, eigen_confidence, eigen_details),
                (lbp_person_id, lbp_confidence, lbp_details)
            )
        else:  # cascade
            result = self._combine_cascade(
                (eigen_person_id, eigen_confidence, eigen_details),
                (lbp_person_id, lbp_confidence, lbp_details)
            )

        return result

    def _recognize_eigenfaces_only(self, processed_face: np.ndarray) -> Dict[str, Any]:
        """
        Reconocimiento solo con Eigenfaces
        """
        person_id, confidence, details = self.eigenfaces_service.recognize_face(processed_face)

        return {
            "recognized": person_id != -1,
            "person_id": person_id if person_id != -1 else None,
            "confidence": confidence,
            "method": "eigenfaces",
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

    def _recognize_lbp_only(self, processed_face: np.ndarray) -> Dict[str, Any]:
        """
        Reconocimiento solo con LBP
        """
        person_id, confidence, details = self.lbp_service.recognize_face(processed_face)

        return {
            "recognized": person_id != -1,
            "person_id": person_id if person_id != -1 else None,
            "confidence": confidence,
            "method": "lbp",
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

    def _recognize_voting(self, processed_face: np.ndarray) -> Dict[str, Any]:
        """
        Reconocimiento por votación simple
        """
        eigen_person_id, eigen_confidence, eigen_details = self.eigenfaces_service.recognize_face(processed_face)
        lbp_person_id, lbp_confidence, lbp_details = self.lbp_service.recognize_face(processed_face)

        # Votación simple
        if eigen_person_id == lbp_person_id and eigen_person_id != -1:
            # Ambos concuerdan
            final_confidence = (eigen_confidence + lbp_confidence) / 2
            final_person_id = eigen_person_id
            consensus = True
        elif eigen_confidence > lbp_confidence:
            # Eigenfaces tiene más confianza
            final_person_id = eigen_person_id
            final_confidence = eigen_confidence * 0.8  # Penalizar por falta de consenso
            consensus = False
        else:
            # LBP tiene más confianza
            final_person_id = lbp_person_id
            final_confidence = lbp_confidence * 0.8  # Penalizar por falta de consenso
            consensus = False

        return {
            "recognized": final_person_id != -1 and final_confidence >= self.confidence_threshold,
            "person_id": final_person_id if final_person_id != -1 else None,
            "confidence": final_confidence,
            "method": "voting",
            "consensus": consensus,
            "details": {
                "eigenfaces": eigen_details,
                "lbp": lbp_details,
                "final_decision": "consensus" if consensus else "confidence_based"
            },
            "timestamp": datetime.now().isoformat()
        }

    def _combine_weighted_average(self, eigen_result: Tuple, lbp_result: Tuple) -> Dict[str, Any]:
        """
        Combina resultados usando promedio ponderado
        """
        eigen_person_id, eigen_confidence, eigen_details = eigen_result
        lbp_person_id, lbp_confidence, lbp_details = lbp_result

        # Si ambos algoritmos identifican la misma persona
        if eigen_person_id == lbp_person_id and eigen_person_id != -1:
            final_person_id = eigen_person_id
            final_confidence = (eigen_confidence * self.eigenfaces_weight +
                                lbp_confidence * self.lbp_weight)
            consensus = True
        else:
            # Usar el algoritmo con mayor confianza ponderada
            eigen_weighted = eigen_confidence * self.eigenfaces_weight
            lbp_weighted = lbp_confidence * self.lbp_weight

            if eigen_weighted > lbp_weighted:
                final_person_id = eigen_person_id
                final_confidence = eigen_weighted
            else:
                final_person_id = lbp_person_id
                final_confidence = lbp_weighted

            consensus = False

        return {
            "recognized": final_person_id != -1 and final_confidence >= self.confidence_threshold,
            "person_id": final_person_id if final_person_id != -1 else None,
            "confidence": final_confidence,
            "method": "weighted_average",
            "consensus": consensus,
            "weights": {
                "eigenfaces": self.eigenfaces_weight,
                "lbp": self.lbp_weight
            },
            "details": {
                "eigenfaces": eigen_details,
                "lbp": lbp_details
            },
            "timestamp": datetime.now().isoformat()
        }

    def _combine_voting(self, eigen_result: Tuple, lbp_result: Tuple) -> Dict[str, Any]:
        """
        Combina resultados usando votación
        """
        return self._recognize_voting(None)  # Ya implementado arriba

    def _combine_cascade(self, eigen_result: Tuple, lbp_result: Tuple) -> Dict[str, Any]:
        """
        Combina resultados usando cascada (Eigenfaces primero, LBP como verificación)
        """
        eigen_person_id, eigen_confidence, eigen_details = eigen_result
        lbp_person_id, lbp_confidence, lbp_details = lbp_result

        # Paso 1: Eigenfaces
        if eigen_person_id != -1 and eigen_confidence >= self.confidence_threshold:
            # Paso 2: Verificar con LBP
            if lbp_person_id == eigen_person_id:
                # LBP confirma
                final_confidence = min(100, eigen_confidence * 1.1)  # Bonus por confirmación
                verification = "confirmed"
            else:
                # LBP no confirma, reducir confianza
                final_confidence = eigen_confidence * 0.7
                verification = "not_confirmed"

            return {
                "recognized": final_confidence >= self.confidence_threshold,
                "person_id": eigen_person_id,
                "confidence": final_confidence,
                "method": "cascade",
                "verification": verification,
                "details": {
                    "primary": eigen_details,
                    "verification": lbp_details
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Eigenfaces no reconoce, usar LBP como respaldo
            return {
                "recognized": lbp_person_id != -1 and lbp_confidence >= self.confidence_threshold,
                "person_id": lbp_person_id if lbp_person_id != -1 else None,
                "confidence": lbp_confidence * 0.9,  # Penalizar por ser respaldo
                "method": "cascade_fallback",
                "details": {
                    "primary": eigen_details,
                    "fallback": lbp_details
                },
                "timestamp": datetime.now().isoformat()
            }

    def _generate_and_save_embeddings(self, images: List[np.ndarray], labels: List[int]) -> None:
        embeddings_data = []

        for image, label in zip(images, labels):
            try:
                # Extraer características con ambos algoritmos
                eigen_features = self.eigenfaces_service.extract_features(image)
                lbp_features = self.lbp_service.extract_lbp_features(image)

                embedding = {
                    "person_id": label,
                    "eigenfaces_embedding": eigen_features.tolist(),
                    "lbp_embedding": lbp_features.tolist(),
                    "timestamp": datetime.now().isoformat(),
                    "algorithm_version": self.model_version
                }

                embeddings_data.append(embedding)

            except Exception as e:
                print(f"⚠️ Error generando embedding para persona {label}: {e}")
                continue  # Saltar este embedding pero continuar

        # Guardar solo los embeddings exitosos
        if embeddings_data:
            embeddings_file = os.path.join(self.storage_path,
                                           f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(embeddings_file, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            print(f"💾 {len(embeddings_data)} embeddings guardados en: {embeddings_file}")

    def get_system_info(self) -> Dict[str, Any]:
        """
        Obtiene información completa del sistema
        """
        return {
            "system_info": {
                "is_trained": self.is_trained,
                "model_version": self.model_version,
                "combination_method": self.combination_method,
                "confidence_threshold": self.confidence_threshold,
                "training_sessions": len(self.training_history)
            },
            "eigenfaces_info": self.eigenfaces_service.get_model_info(),
            "lbp_info": self.lbp_service.get_model_info(),
            "weights": {
                "eigenfaces": self.eigenfaces_weight,
                "lbp": self.lbp_weight
            },
            "last_training": self.training_history[-1] if self.training_history else None
        }

    def load_models(self) -> bool:
        """
        Carga modelos previamente entrenados
        """
        try:
            self.eigenfaces_service.load_model()
            self.lbp_service.load_model()

            self.is_trained = (self.eigenfaces_service.is_trained and
                               self.lbp_service.is_trained)

            if self.is_trained:
                print("✅ Modelos cargados exitosamente")
            else:
                print("⚠️ Los modelos no están completamente entrenados")

            return self.is_trained

        except Exception as e:
            print(f"❌ Error al cargar modelos: {e}")
            return False

    def update_configuration(self, config: Dict[str, Any]) -> None:
        """
        Actualiza la configuración del sistema
        """
        if "eigenfaces_weight" in config:
            self.eigenfaces_weight = config["eigenfaces_weight"]
            self.lbp_weight = 1.0 - self.eigenfaces_weight

        if "confidence_threshold" in config:
            self.confidence_threshold = config["confidence_threshold"]

        if "combination_method" in config:
            self.combination_method = config["combination_method"]

        print("⚙️ Configuración actualizada")

    def benchmark_algorithms(self, test_images: List[Tuple[np.ndarray, int]]) -> Dict[str, Any]:
        """
        Evalúa el rendimiento de los diferentes algoritmos
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        results = {
            "eigenfaces": {"correct": 0, "total": 0, "confidences": []},
            "lbp": {"correct": 0, "total": 0, "confidences": []},
            "hybrid": {"correct": 0, "total": 0, "confidences": []}
        }

        for image, true_label in test_images:
            # Test Eigenfaces
            eigen_result = self._recognize_eigenfaces_only(image)
            results["eigenfaces"]["total"] += 1
            results["eigenfaces"]["confidences"].append(eigen_result["confidence"])
            if eigen_result["person_id"] == true_label:
                results["eigenfaces"]["correct"] += 1

            # Test LBP
            lbp_result = self._recognize_lbp_only(image)
            results["lbp"]["total"] += 1
            results["lbp"]["confidences"].append(lbp_result["confidence"])
            if lbp_result["person_id"] == true_label:
                results["lbp"]["correct"] += 1

            # Test Hybrid
            hybrid_result = self._recognize_hybrid(image)
            results["hybrid"]["total"] += 1
            results["hybrid"]["confidences"].append(hybrid_result["confidence"])
            if hybrid_result["person_id"] == true_label:
                results["hybrid"]["correct"] += 1

        # Calcular métricas
        for method in results:
            accuracy = results[method]["correct"] / results[method]["total"]
            avg_confidence = np.mean(results[method]["confidences"])

            results[method]["accuracy"] = accuracy
            results[method]["average_confidence"] = avg_confidence

        return results