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

        # ⚡ AÑADIR ESTAS LÍNEAS NUEVAS ⚡
        self.auto_training_enabled = True
        self.min_persons_for_training = 2  # Mínimo 2 personas para entrenar
        self.pending_persons = {}  # Personas pendientes de entrenamiento

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
        Entrena modelos conservando imágenes originales para características
        """
        print("🚀 Iniciando entrenamiento del modelo híbrido...")

        # Preparar datos de entrenamiento
        all_images = []
        all_labels = []
        original_images_by_person = {}  # ⚡ MANTENER ORIGINALES

        for person_id, images in images_by_person.items():
            person_processed = []
            person_originals = []

            for image in images:
                # Guardar imagen original
                person_originals.append(image.copy())

                # Procesar para entrenamiento
                processed_face = self.preprocess_image_for_training(image)
                if processed_face is not None:
                    all_images.append(processed_face)
                    all_labels.append(person_id)
                    person_processed.append(processed_face)

            if person_originals:
                original_images_by_person[person_id] = person_originals

        # Entrenar con imágenes procesadas
        self.eigenfaces_service.train(all_images, all_labels)
        self.lbp_service.train(all_images, all_labels)

        # Guardar modelos
        self.eigenfaces_service.save_model()
        self.lbp_service.save_model()
        self.is_trained = True

        # Estadísticas
        training_stats = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(all_images),
            "unique_persons": len(set(all_labels)),
            "eigenfaces_info": self.eigenfaces_service.get_model_info(),
            "lbp_info": self.lbp_service.get_model_info(),
            "model_version": self.model_version
        }

        # ⚡ USAR IMÁGENES ORIGINALES para características
        try:
            self._save_characteristics_to_db(original_images_by_person)
            self._save_training_record(training_stats)
        except Exception as e:
            print(f"⚠️ Error en BD: {e}")

        return training_stats

    def _save_characteristics_to_db(self, images_by_person: Dict[int, List[np.ndarray]]):
        """
        Guarda características faciales usando LAS IMÁGENES ORIGINALES
        """
        from config.database import SessionLocal
        from models.database_models import CaracteristicasFaciales, ImagenFacial

        print("💾 INICIANDO GUARDADO DE CARACTERÍSTICAS EN BD")

        db = SessionLocal()
        try:
            characteristics_saved = 0
            errors = []

            for person_id, images in images_by_person.items():
                print(f"\n👤 Procesando persona ID: {person_id}")

                # Obtener imágenes de la BD
                db_images = db.query(ImagenFacial).filter(
                    ImagenFacial.usuario_id == person_id,
                    ImagenFacial.activa == True
                ).all()

                if not db_images:
                    continue

                images_to_process = min(len(images), len(db_images))

                for i in range(images_to_process):
                    original_image = images[i]  # Imagen original 3D/2D
                    db_image = db_images[i]

                    try:
                        print(f"   🔍 Imagen original shape: {original_image.shape}")

                        # ⚡ CRÍTICO: Usar la imagen ORIGINAL para ambos algoritmos
                        # Cada servicio debe hacer su propio preprocesamiento

                        # Eigenfaces - usa su propio preprocesamiento
                        print(f"   📐 Extrayendo Eigenfaces...")
                        eigenfaces_features = self.eigenfaces_service.extract_features(original_image)
                        print(f"   ✅ Eigenfaces OK: {eigenfaces_features.shape}")

                        # LBP - usa su propio preprocesamiento desde la imagen ORIGINAL
                        print(f"   🔍 Extrayendo LBP...")
                        lbp_features = self.lbp_service.extract_lbp_features(original_image)
                        print(f"   ✅ LBP OK: {lbp_features.shape}")

                        # Verificar características existentes
                        existing = db.query(CaracteristicasFaciales).filter(
                            CaracteristicasFaciales.imagen_id == db_image.id
                        ).first()

                        if existing:
                            existing.eigenfaces_vector = eigenfaces_features.tolist()
                            existing.lbp_histogram = lbp_features.tolist()
                            existing.fecha_procesamiento = datetime.now()
                        else:
                            caracteristicas = CaracteristicasFaciales(
                                usuario_id=person_id,
                                imagen_id=db_image.id,
                                eigenfaces_vector=eigenfaces_features.tolist(),
                                lbp_histogram=lbp_features.tolist(),
                                algoritmo_version="2.0",
                                calidad_deteccion=85
                            )
                            db.add(caracteristicas)

                        characteristics_saved += 1
                        print(f"   ✅ Características guardadas")

                    except Exception as e:
                        error_msg = f"Error en imagen {db_image.id}: {str(e)}"
                        print(f"   ❌ {error_msg}")
                        errors.append(error_msg)
                        continue

            if characteristics_saved > 0:
                db.commit()
                print(f"✅ {characteristics_saved} características guardadas en BD")

            if errors:
                print(f"\n⚠️ ERRORES ({len(errors)}):")
                for error in errors[:3]:
                    print(f"   • {error}")

        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {str(e)}")
            db.rollback()
        finally:
            db.close()

    def _save_training_record(self, training_stats: Dict[str, Any]):
        """
        Guarda registro del entrenamiento en BD
        """
        from config.database import SessionLocal
        from models.database_models import ModeloEntrenamiento

        db = SessionLocal()
        try:
            # Crear registro de entrenamiento
            training_record = ModeloEntrenamiento(
                version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                algoritmo="hybrid",
                total_usuarios=training_stats.get("unique_persons", 0),
                total_imagenes=training_stats.get("total_images", 0),
                precision_promedio="N/A",  # Se calculará después con validación
                ruta_modelo_eigenfaces="storage/models/eigenfaces_model.pkl",
                ruta_modelo_lbp="storage/models/lbp_model.pkl",
                configuracion={
                    "eigenfaces_components": training_stats.get("eigenfaces_info", {}).get("n_components", 0),
                    "lbp_radius": training_stats.get("lbp_info", {}).get("radius", 0),
                    "lbp_points": training_stats.get("lbp_info", {}).get("n_points", 0),
                    "training_stats": training_stats
                }
            )

            db.add(training_record)
            db.commit()

            print(f"✅ Registro de entrenamiento guardado: {training_record.version}")

        except Exception as e:
            db.rollback()
            print(f"❌ Error guardando registro de entrenamiento: {e}")
        finally:
            db.close()

    def add_new_person(self, person_id: int, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Añade una nueva persona al sistema con entrenamiento automático inteligente

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

        # LÓGICA MEJORADA: Determinar acción según estado del modelo
        if not self.is_trained:
            print("🔄 Modelo no entrenado - Acumulando datos para entrenamiento inicial")

            # Añadir a pendientes
            self.pending_persons[person_id] = processed_images

            # Verificar si tenemos suficientes personas para entrenar
            if len(self.pending_persons) >= self.min_persons_for_training:
                print(f"🚀 Iniciando entrenamiento automático con {len(self.pending_persons)} personas")
                return self._trigger_auto_training()
            else:
                print(
                    f"⏳ Esperando más datos. Personas actuales: {len(self.pending_persons)}/{self.min_persons_for_training}")
                return {
                    "person_id": person_id,
                    "images_processed": len(processed_images),
                    "status": "pending_training",
                    "message": f"Datos guardados. Se necesitan {self.min_persons_for_training - len(self.pending_persons)} personas más para entrenar",
                    "timestamp": datetime.now().isoformat()
                }
        else:
            print("🔄 Modelo ya entrenado - Entrenamiento incremental")

            # Entrenamiento incremental
            self.eigenfaces_service.add_new_person(processed_images, person_id)
            self.lbp_service.add_new_person(processed_images, person_id)

            # Guardar modelos actualizados
            self.eigenfaces_service.save_model()
            self.lbp_service.save_model()

            # Generar embeddings
            self._generate_and_save_embeddings(processed_images, [person_id] * len(processed_images))

            # Guardar características en BD
            try:
                self._save_incremental_characteristics({person_id: images})  # Usar imágenes originales
            except Exception as e:
                print(f"⚠️ Error guardando características en BD: {e}")

            return {
                "person_id": person_id,
                "images_processed": len(processed_images),
                "status": "added_incremental",
                "message": "Persona añadida al modelo existente",
                "timestamp": datetime.now().isoformat()
            }

    def _trigger_auto_training(self) -> Dict[str, Any]:
        """
        Dispara el entrenamiento automático con los datos pendientes
        """
        try:
            print("🎓 Iniciando entrenamiento automático...")

            # Usar datos pendientes para entrenamiento inicial
            training_stats = self.train_models(self.pending_persons)

            # Limpiar pendientes
            self.pending_persons.clear()

            training_stats.update({
                "status": "auto_trained",
                "message": "Modelo entrenado automáticamente",
                "auto_training": True
            })

            return training_stats

        except Exception as e:
            print(f"❌ Error en entrenamiento automático: {e}")
            return {
                "status": "training_failed",
                "error": str(e),
                "message": "Error en entrenamiento automático",
                "timestamp": datetime.now().isoformat()
            }

    def _save_incremental_characteristics(self, images_by_person: Dict[int, List[np.ndarray]]):
        """
        Guarda características para entrenamiento incremental
        """
        from config.database import SessionLocal
        from models.database_models import CaracteristicasFaciales, ImagenFacial

        print("💾 Guardando características incrementales en BD")

        db = SessionLocal()
        try:
            for person_id, images in images_by_person.items():
                # Obtener las imágenes más recientes de la BD
                db_images = db.query(ImagenFacial).filter(
                    ImagenFacial.usuario_id == person_id,
                    ImagenFacial.activa == True
                ).order_by(ImagenFacial.fecha_subida.desc()).limit(len(images)).all()

                if not db_images:
                    continue

                for i, (original_image, db_image) in enumerate(zip(images, db_images)):
                    try:
                        # Extraer características usando imágenes originales
                        eigenfaces_features = self.eigenfaces_service.extract_features(original_image)
                        lbp_features = self.lbp_service.extract_lbp_features(original_image)

                        # Verificar si ya existe
                        existing = db.query(CaracteristicasFaciales).filter(
                            CaracteristicasFaciales.imagen_id == db_image.id
                        ).first()

                        if existing:
                            # Actualizar existente
                            existing.eigenfaces_vector = eigenfaces_features.tolist()
                            existing.lbp_histogram = lbp_features.tolist()
                            existing.fecha_procesamiento = datetime.now()
                            print(f"   🔄 Características actualizadas para imagen {db_image.id}")
                        else:
                            # Crear nuevo registro
                            caracteristicas = CaracteristicasFaciales(
                                usuario_id=person_id,
                                imagen_id=db_image.id,
                                eigenfaces_vector=eigenfaces_features.tolist(),
                                lbp_histogram=lbp_features.tolist(),
                                algoritmo_version="2.0",
                                calidad_deteccion=85
                            )
                            db.add(caracteristicas)
                            print(f"   ✅ Características creadas para imagen {db_image.id}")

                    except Exception as e:
                        print(f"   ❌ Error procesando imagen {db_image.id}: {e}")
                        continue

            db.commit()
            print("✅ Características incrementales guardadas exitosamente")

        except Exception as e:
            print(f"❌ Error crítico guardando características: {e}")
            db.rollback()
        finally:
            db.close()

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

    def check_training_requirements(self) -> Dict[str, Any]:
        """
        Verifica los requisitos para entrenamiento
        """
        from config.database import SessionLocal
        from models.database_models import Usuario, ImagenFacial

        db = SessionLocal()
        try:
            # Contar usuarios con imágenes
            usuarios_con_imagenes = db.query(Usuario).filter(
                Usuario.activo == True
            ).join(ImagenFacial).filter(
                ImagenFacial.activa == True
            ).distinct().count()

            total_imagenes = db.query(ImagenFacial).filter(
                ImagenFacial.activa == True
            ).count()

            requirements = {
                "can_train": usuarios_con_imagenes >= self.min_persons_for_training,
                "users_with_images": usuarios_con_imagenes,
                "total_images": total_imagenes,
                "min_required": self.min_persons_for_training,
                "pending_users": len(self.pending_persons),
                "model_trained": self.is_trained,
                "auto_training_enabled": self.auto_training_enabled
            }

            return requirements

        finally:
            db.close()

    def force_retrain_from_database(self) -> Dict[str, Any]:
        """
        Fuerza un reentrenamiento completo desde la base de datos
        """
        from config.database import SessionLocal
        from models.database_models import Usuario, ImagenFacial
        import cv2

        print("🔄 Forzando reentrenamiento desde base de datos...")

        db = SessionLocal()
        try:
            # Obtener todos los usuarios activos con imágenes
            usuarios = db.query(Usuario).filter(Usuario.activo == True).all()
            images_by_person = {}

            for usuario in usuarios:
                imagenes = db.query(ImagenFacial).filter(
                    ImagenFacial.usuario_id == usuario.id,
                    ImagenFacial.activa == True
                ).all()

                if imagenes:
                    user_images = []
                    for imagen in imagenes:
                        if os.path.exists(imagen.ruta_archivo):
                            img = cv2.imread(imagen.ruta_archivo)
                            if img is not None:
                                user_images.append(img)

                    if user_images:
                        images_by_person[usuario.id] = user_images

            if len(images_by_person) < self.min_persons_for_training:
                return {
                    "success": False,
                    "error": f"Insuficientes usuarios con imágenes. Requeridos: {self.min_persons_for_training}, Disponibles: {len(images_by_person)}"
                }

            # Resetear estado
            self.is_trained = False
            self.pending_persons.clear()

            # Entrenar modelo
            training_stats = self.train_models(images_by_person)
            training_stats["forced_retrain"] = True
            training_stats["success"] = True

            return training_stats

        finally:
            db.close()

    def get_training_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del entrenamiento
        """
        requirements = self.check_training_requirements()

        status = {
            "model_trained": self.is_trained,
            "auto_training_enabled": self.auto_training_enabled,
            "training_requirements": requirements,
            "system_ready": self.is_trained or requirements["can_train"],
            "recommendation": self._get_training_recommendation(requirements)
        }

        return status

    def _get_training_recommendation(self, requirements: Dict[str, Any]) -> str:
        """
        Obtiene recomendación sobre el entrenamiento
        """
        if self.is_trained:
            return "✅ Modelo entrenado y listo para uso"
        elif requirements["can_train"]:
            return "🎓 Datos suficientes - Se puede entrenar automáticamente"
        else:
            needed = requirements["min_required"] - requirements["users_with_images"]
            return f"⏳ Se necesitan {needed} usuarios más con imágenes para entrenar"

