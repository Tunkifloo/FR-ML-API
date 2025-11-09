import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np

from .eigenfaces_service import EigenfacesService
from .face_detection_service import FaceDetectionService
from .lbp_service import LBPService
from config.ml_config import MLConfig
from services.quality_checker import ImageQualityChecker
from services.face_alignment import FaceAlignmentService

class MLService:
    """
    VERSIÓN CORREGIDA: Servicio principal de Machine Learning que combina Eigenfaces y LBP
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
        self.model_version = "2.0_FIXED"
        self.training_history = []

        # ⚡ CORREGIR: Añadir preprocesador SOLO si el archivo existe
        try:
            from .image_preprocessor import ImagePreprocessor
            self.preprocessor = ImagePreprocessor(target_size=(100, 100))
            print(f"✅ ImagePreprocessor inicializado correctamente")
        except ImportError as e:
            print(f"⚠️ ImagePreprocessor no disponible: {e}")
            self.preprocessor = None

        # Configuración de entrenamiento automático
        self.auto_training_enabled = True
        self.min_persons_for_training = 2
        self.pending_persons = {}

        # Configuración de combinación
        self.combination_method = "weighted_average"
        self.eigenfaces_weight = 0.6
        self.lbp_weight = 0.4

        # Umbrales
        self.confidence_threshold = 70.0
        self.consensus_threshold = 0.7

        # Almacenamiento
        self.storage_path = "storage/embeddings/"
        os.makedirs(self.storage_path, exist_ok=True)

        # Servicios adicionales para mejoras
        self.quality_checker = ImageQualityChecker()
        try:
            self.face_aligner = FaceAlignmentService()
            self.alignment_available = True
            print("✅ Alineación facial disponible")
        except Exception as e:
            self.face_aligner = None
            self.alignment_available = False
            print(f"⚠️ Alineación facial no disponible: {e}")

    def preprocess_image_for_training(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        CORREGIDO: Preprocesa imagen de manera ROBUSTA para entrenamiento
        """
        try:
            print(f"🔧 Preprocessing para entrenamiento: {image.shape}, dtype: {image.dtype}")

            # PASO 1: Validar imagen
            if image is None or image.size == 0:
                print(f"❌ Imagen inválida")
                return None

            # PASO 2: Detectar rostros (opcional, con fallback)
            try:
                faces = self.face_detector.detect_faces(image)

                if faces:
                    # Obtener el rostro más grande
                    largest_face = self.face_detector.get_largest_face(faces)
                    face_roi = self.face_detector.extract_face_roi(image, largest_face)
                    print(f"✅ Rostro detectado y extraído: {face_roi.shape}")

                    # Procesar ROI del rostro
                    if self.preprocessor:
                        processed_face = self.preprocessor.preprocess_for_ml(face_roi, "both")
                    else:
                        processed_face = self._basic_preprocess(face_roi)
                    return processed_face
                else:
                    print(f"⚠️ No se detectaron rostros, usando imagen completa")
                    # FALLBACK: Usar imagen completa
                    if self.preprocessor:
                        processed_face = self.preprocessor.preprocess_for_ml(image, "both")
                    else:
                        processed_face = self._basic_preprocess(image)
                    return processed_face

            except Exception as e:
                print(f"⚠️ Error en detección de rostros: {e}")
                # FALLBACK: Usar imagen completa
                if self.preprocessor:
                    processed_face = self.preprocessor.preprocess_for_ml(image, "both")
                else:
                    processed_face = self._basic_preprocess(image)
                return processed_face

        except Exception as e:
            print(f"❌ Error crítico en preprocesamiento: {e}")
            print(f"   Input shape: {image.shape if image is not None else 'None'}")

            # ÚLTIMO FALLBACK: Preprocesamiento básico sin preprocessor
            try:
                if image is not None and image.size > 0:
                    basic_processed = self._basic_preprocess(image)
                    print(f"🔄 Fallback básico exitoso: {basic_processed.shape}")
                    return basic_processed
            except Exception as e2:
                print(f"❌ Fallback también falló: {e2}")

            return None

    def _basic_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        NUEVO: Preprocesamiento básico sin dependencias externas
        """
        try:
            processed = image.copy()

            # Convertir a escala de grises
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # Redimensionar
            processed = cv2.resize(processed, (100, 100))

            # Asegurar uint8
            if processed.dtype != np.uint8:
                if processed.max() <= 1.0:
                    processed = (processed * 255).astype(np.uint8)
                else:
                    processed = processed.astype(np.uint8)

            # Ecualización básica
            processed = cv2.equalizeHist(processed)

            # ✅ CRUCIAL: Para entrenamiento, normalizar a float64 [0,1] PERO mantener información de origen
            processed = processed.astype(np.float64) / 255.0

            print(f"🔧 Preprocesamiento básico: {processed.shape}, dtype: {processed.dtype}")
            return processed

        except Exception as e:
            print(f"❌ Error en preprocesamiento básico: {e}")
            raise

    def _apply_data_augmentation(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Aplica data augmentation a un conjunto de imágenes
        Retorna imágenes en el MISMO formato que recibe (float64 o uint8)
        """
        if not MLConfig.USE_AUGMENTATION:
            return images

        augmented_images = []

        for img in images:
            # Siempre añadir la imagen original
            augmented_images.append(img)

            # Detectar el tipo de dato de entrada
            is_float = img.dtype == np.float64 or img.dtype == np.float32

            # Convertir temporalmente a uint8 para transformaciones
            if is_float:
                img_work = (img * 255).astype(np.uint8)
            else:
                img_work = img.copy()

            h, w = img_work.shape[:2]

            # 1. ROTACIONES
            for angle in MLConfig.AUGMENTATION_ROTATIONS:
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                rotated = cv2.warpAffine(img_work, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

                # Convertir de vuelta al formato original
                if is_float:
                    rotated = rotated.astype(np.float64) / 255.0

                augmented_images.append(rotated)

            # 2. ESCALAS
            for scale in MLConfig.AUGMENTATION_SCALES:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(img_work, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Crop o pad para mantener el tamaño original
                if scale > 1.0:
                    # Crop del centro
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    scaled = scaled[start_h:start_h + h, start_w:start_w + w]
                else:
                    # Pad con replicación de bordes
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    pad_h_extra = h - new_h - pad_h
                    pad_w_extra = w - new_w - pad_w
                    scaled = cv2.copyMakeBorder(
                        scaled, pad_h, pad_h_extra, pad_w, pad_w_extra,
                        cv2.BORDER_REPLICATE
                    )

                # Convertir de vuelta al formato original
                if is_float:
                    scaled = scaled.astype(np.float64) / 255.0

                augmented_images.append(scaled)

        original_count = len(images)
        augmented_count = len(augmented_images)
        factor = augmented_count / original_count if original_count > 0 else 0

        print(f"📊 Data Augmentation: {original_count} → {augmented_count} imágenes (factor: {factor:.1f}x)")

        return augmented_images

    def train_models(self, images_by_person: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
        """
        Entrena modelos con manejo adecuado de tipos de datos y data augmentation opcional
        """
        print("=" * 60)
        print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO HÍBRIDO")
        print("=" * 60)

        if MLConfig.USE_AUGMENTATION:
            print(f"📊 Data Augmentation: ACTIVADO")
            print(f"   • Rotaciones: {MLConfig.AUGMENTATION_ROTATIONS}")
            print(f"   • Escalas: {MLConfig.AUGMENTATION_SCALES}")
        else:
            print(f"📊 Data Augmentation: DESACTIVADO")

        print()

        # Preparar datos de entrenamiento
        all_images_eigenfaces = []  # Para Eigenfaces (float64)
        all_images_lbp = []  # Para LBP (uint8)
        all_labels = []
        original_images_by_person = {}

        # Contador para estadísticas
        total_original_images = 0
        total_augmented_images = 0

        for person_id, images in images_by_person.items():
            print(f"👤 Procesando persona ID {person_id}: {len(images)} imágenes originales")

            person_originals = []
            person_eigenfaces_processed = []
            person_lbp_processed = []

            for image in images:
                total_original_images += 1

                # Guardar imagen original (sin modificar)
                person_originals.append(image.copy())

                # Procesar imagen para entrenamiento
                processed_face = self.preprocess_image_for_training(image)
                if processed_face is None:
                    print(f"   ⚠️ No se pudo preprocesar una imagen")
                    continue

                # Guardar versión procesada para cada algoritmo
                person_eigenfaces_processed.append(processed_face)  # float64 [0,1]
                person_lbp_processed.append((processed_face * 255).astype(np.uint8))  # uint8 [0,255]

            # APLICAR DATA AUGMENTATION por persona
            if person_eigenfaces_processed:
                # Augmentar para Eigenfaces
                augmented_eigenfaces = self._apply_data_augmentation(person_eigenfaces_processed)

                # Augmentar para LBP
                augmented_lbp = self._apply_data_augmentation(person_lbp_processed)

                # Verificar que ambos tengan la misma cantidad
                if len(augmented_eigenfaces) != len(augmented_lbp):
                    print(f"   ⚠️ Advertencia: Desbalance en augmentation")

                # Añadir a los conjuntos globales
                for eigen_img, lbp_img in zip(augmented_eigenfaces, augmented_lbp):
                    all_images_eigenfaces.append(eigen_img)
                    all_images_lbp.append(lbp_img)
                    all_labels.append(person_id)
                    total_augmented_images += 1

                # Guardar originales (sin augmentation) para características
                original_images_by_person[person_id] = person_originals

                print(f"   ✅ {len(person_eigenfaces_processed)} → {len(augmented_eigenfaces)} imágenes")

        print()
        print("=" * 60)
        print("📊 RESUMEN DE DATOS PREPARADOS:")
        print("=" * 60)
        print(f"   • Imágenes originales: {total_original_images}")
        print(f"   • Imágenes para entrenamiento: {total_augmented_images}")
        print(
            f"   • Factor de augmentation: {total_augmented_images / total_original_images:.1f}x" if total_original_images > 0 else "N/A")
        print(f"   • Personas únicas: {len(set(all_labels))}")
        print(f"   • Eigenfaces: {len(all_images_eigenfaces)} imágenes (float64)")
        print(f"   • LBP: {len(all_images_lbp)} imágenes (uint8)")
        print()

        # Verificar que tengamos suficientes datos
        if len(all_labels) < 2:
            raise ValueError("Se necesitan al menos 2 imágenes para entrenar")

        # ✅ ENTRENAR CON DATOS ESPECÍFICOS PARA CADA ALGORITMO
        print("=" * 60)
        print("🎓 ENTRENANDO ALGORITMOS")
        print("=" * 60)

        print("🔹 Entrenando Eigenfaces...")
        self.eigenfaces_service.train(all_images_eigenfaces, all_labels)
        print("   ✅ Eigenfaces entrenado")

        print()
        print("🔹 Entrenando LBP...")
        self.lbp_service.train(all_images_lbp, all_labels)
        print("   ✅ LBP entrenado")

        print()

        # Guardar modelos
        print("💾 Guardando modelos...")
        self.eigenfaces_service.save_model()
        self.lbp_service.save_model()
        self.is_trained = True
        print("   ✅ Modelos guardados")
        print()

        # Estadísticas del entrenamiento
        training_stats = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(all_labels),
            "total_original_images": total_original_images,
            "augmentation_factor": total_augmented_images / total_original_images if total_original_images > 0 else 1.0,
            "augmentation_enabled": MLConfig.USE_AUGMENTATION,
            "unique_persons": len(set(all_labels)),
            "eigenfaces_info": self.eigenfaces_service.get_model_info(),
            "lbp_info": self.lbp_service.get_model_info(),
            "model_version": self.model_version,
            "data_types_used": {
                "eigenfaces": "float64 [0,1]",
                "lbp": "uint8 [0,255]"
            },
            "config_used": {
                "use_quality_check": MLConfig.USE_QUALITY_CHECK,
                "use_face_alignment": MLConfig.USE_FACE_ALIGNMENT,
                "use_advanced_illumination": MLConfig.USE_ADVANCED_ILLUMINATION,
                "use_augmentation": MLConfig.USE_AUGMENTATION
            }
        }

        # USAR IMÁGENES ORIGINALES (sin augmentation) para características en BD
        print("💾 Guardando características en base de datos...")
        try:
            self._save_characteristics_to_db(original_images_by_person)
            print("   ✅ Características guardadas")
        except Exception as e:
            print(f"   ⚠️ Error guardando características: {e}")
            import traceback
            traceback.print_exc()

        print()
        print("💾 Guardando registro de entrenamiento...")
        try:
            self._save_training_record(training_stats)
            print("   ✅ Registro guardado")
        except Exception as e:
            print(f"   ⚠️ Error guardando registro: {e}")

        print()
        print("=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📊 Resumen:")
        print(f"   • {training_stats['unique_persons']} personas")
        print(f"   • {training_stats['total_original_images']} imágenes originales")
        print(f"   • {training_stats['total_images']} imágenes de entrenamiento")
        if MLConfig.USE_AUGMENTATION:
            print(f"   • Factor de augmentation: {training_stats['augmentation_factor']:.1f}x")
        print("=" * 60)
        print()

        return training_stats

    def _clean_for_json_storage(self, features: np.ndarray) -> list:
        """
        ✅ NUEVO: Limpia características para almacenamiento JSON seguro
        """
        if features is None:
            return None

        cleaned = np.array(features, copy=True)

        # Reemplazar valores problemáticos
        cleaned[np.isposinf(cleaned)] = 1e6  # Infinity positivo
        cleaned[np.isneginf(cleaned)] = -1e6  # Infinity negativo
        cleaned[np.isnan(cleaned)] = 0.0  # NaN

        # Clipear a rango seguro
        cleaned = np.clip(cleaned, -1e6, 1e6)

        # Convertir a lista Python con validación
        result = []
        for value in cleaned:
            if isinstance(value, np.ndarray):
                # Si es un array, procesar recursivamente
                result.extend(self._clean_for_json_storage(value))
            else:
                # Convertir a float Python nativo
                clean_value = float(value)
                # Validación final
                if np.isfinite(clean_value):
                    result.append(clean_value)
                else:
                    result.append(0.0)

        return result

    def _save_characteristics_to_db(self, images_by_person: Dict[int, List[np.ndarray]]):
        """
        ✅ CORREGIDO: Guarda características con limpieza de valores infinitos
        """
        from config.database import SessionLocal
        from models.database_models import CaracteristicasFaciales, ImagenFacial

        print("💾 GUARDANDO CARACTERÍSTICAS CON LIMPIEZA DE INFINITY")

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
                    print(f"   ⚠️ No se encontraron imágenes en BD para usuario {person_id}")
                    continue

                images_to_process = min(len(images), len(db_images))

                for i in range(images_to_process):
                    original_image = images[i]
                    db_image = db_images[i]

                    try:
                        print(f"   📷 Procesando imagen {db_image.id}: {original_image.shape}")

                        # ✅ CRÍTICO: Procesar SEPARADAMENTE para cada algoritmo

                        # EIGENFACES: Necesita float64 [0,1]
                        eigenfaces_features = None
                        try:
                            # Preprocesar específicamente para Eigenfaces
                            processed_for_eigen = self.preprocess_image_for_training(original_image.copy())
                            if processed_for_eigen is not None:
                                raw_features = self.eigenfaces_service.extract_features(processed_for_eigen)
                                # ✅ LIMPIAR PARA JSON
                                eigenfaces_features = self._clean_for_json_storage(raw_features)
                                print(f"   ✅ Eigenfaces: {len(eigenfaces_features)} características limpias")
                        except Exception as e:
                            print(f"   ❌ Error Eigenfaces: {e}")

                        # LBP: Necesita uint8 [0,255]
                        lbp_features = None
                        try:
                            # Preprocesar específicamente para LBP
                            processed_for_lbp = self.preprocess_image_for_training(original_image.copy())
                            if processed_for_lbp is not None:
                                # Convertir a uint8 para LBP
                                lbp_input = (processed_for_lbp * 255).astype(np.uint8)
                                raw_lbp = self.lbp_service.extract_lbp_features(lbp_input)
                                # ✅ LIMPIAR PARA JSON
                                lbp_features = self._clean_for_json_storage(raw_lbp)
                                print(f"   ✅ LBP: {len(lbp_features)} características limpias")
                        except Exception as e:
                            print(f"   ❌ Error LBP: {e}")

                        # Solo guardar si al menos uno funcionó
                        if eigenfaces_features is not None or lbp_features is not None:
                            # Verificar si ya existen características
                            existing = db.query(CaracteristicasFaciales).filter(
                                CaracteristicasFaciales.imagen_id == db_image.id
                            ).first()

                            if existing:
                                # Actualizar existente
                                if eigenfaces_features is not None:
                                    existing.eigenfaces_vector = eigenfaces_features
                                if lbp_features is not None:
                                    existing.lbp_histogram = lbp_features
                                existing.fecha_procesamiento = datetime.now()
                                existing.algoritmo_version = "2.0_INFINITY_FIXED"
                                print(f"   🔄 Características actualizadas para imagen {db_image.id}")
                            else:
                                # Crear nuevo registro
                                caracteristicas = CaracteristicasFaciales(
                                    usuario_id=person_id,
                                    imagen_id=db_image.id,
                                    eigenfaces_vector=eigenfaces_features,
                                    lbp_histogram=lbp_features,
                                    algoritmo_version="2.0_INFINITY_FIXED",
                                    calidad_deteccion=90
                                )
                                db.add(caracteristicas)
                                print(f"   ✅ Características creadas para imagen {db_image.id}")

                            characteristics_saved += 1
                        else:
                            error_msg = f"No se pudieron extraer características de imagen {db_image.id}"
                            print(f"   ❌ {error_msg}")
                            errors.append(error_msg)

                    except Exception as e:
                        error_msg = f"Error procesando imagen {db_image.id}: {str(e)}"
                        print(f"   ❌ {error_msg}")
                        errors.append(error_msg)
                        continue

            # Commit solo si hay características guardadas
            if characteristics_saved > 0:
                db.commit()
                print(f"✅ {characteristics_saved} características guardadas exitosamente en BD")

            if errors:
                print(f"\n⚠️ ERRORES ENCONTRADOS ({len(errors)}):")
                for error in errors[:5]:
                    print(f"   • {error}")
                if len(errors) > 5:
                    print(f"   • ... y {len(errors) - 5} errores más")

        except Exception as e:
            print(f"❌ ERROR CRÍTICO guardando características: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()

    def add_new_person(self, person_id: int, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Añade una nueva persona con manejo adecuado de tipos
        """
        try:
            print(f"[ML] Procesando persona ID: {person_id} con {len(images)} imágenes")

            # VALIDACIÓN INICIAL
            if not images:
                return {
                    "status": "error",
                    "message": "No se proporcionaron imágenes",
                    "person_id": person_id,
                    "timestamp": datetime.now().isoformat()
                }

            # PREPROCESAR IMÁGENES SEPARADAMENTE PARA CADA ALGORITMO
            processed_images_eigenfaces = []
            processed_images_lbp = []

            for i, image in enumerate(images):
                try:
                    # Preprocesar imagen base
                    base_processed = self.preprocess_image_for_training(image)
                    if base_processed is not None:
                        # Para Eigenfaces: mantener float64
                        processed_images_eigenfaces.append(base_processed.copy())

                        # Para LBP: convertir a uint8
                        lbp_image = (base_processed * 255).astype(np.uint8)
                        processed_images_lbp.append(lbp_image)
                    else:
                        print(f"[WARNING] No se pudo procesar imagen {i + 1} para persona {person_id}")
                except Exception as e:
                    print(f"[WARNING] Error procesando imagen {i + 1}: {str(e)}")
                    continue

            if not processed_images_eigenfaces or not processed_images_lbp:
                return {
                    "status": "error",
                    "message": "No se pudieron procesar las imágenes",
                    "person_id": person_id,
                    "timestamp": datetime.now().isoformat()
                }

            print(f"[ML] {len(processed_images_eigenfaces)} imágenes procesadas para Eigenfaces")
            print(f"[ML] {len(processed_images_lbp)} imágenes procesadas para LBP")

            # ENTRENAMIENTO INCREMENTAL SEGURO
            try:
                # Verificar si los modelos están cargados
                if not self.is_trained:
                    print("[ML] Modelos no entrenados, intentando cargar...")
                    self.load_models()

                # Si aún no están entrenados, usar entrenamiento desde BD
                if not self.is_trained:
                    print("[ML] Modelos no disponibles, iniciando entrenamiento desde BD...")
                    return self._train_from_database_safely()

                # AÑADIR A CADA MODELO CON SUS DATOS ESPECÍFICOS
                print(f"[ML] Añadiendo persona {person_id} al modelo existente...")

                # Añadir a eigenfaces (con float64)
                self.eigenfaces_service.add_new_person(processed_images_eigenfaces, person_id)
                print(f"[ML] Eigenfaces actualizado para persona {person_id}")

                # Añadir a LBP (con uint8)
                self.lbp_service.add_new_person(processed_images_lbp, person_id)
                print(f"[ML] LBP actualizado para persona {person_id}")

                # GUARDAR MODELOS DE FORMA SEGURA
                try:
                    self.eigenfaces_service.save_model()
                    self.lbp_service.save_model()
                    print(f"[ML] Modelos guardados exitosamente")
                except Exception as e:
                    print(f"[WARNING] Error guardando modelos: {e}")

                return {
                    "status": "added_incremental",
                    "message": f"Persona {person_id} añadida exitosamente al modelo",
                    "person_id": person_id,
                    "images_processed": {
                        "eigenfaces": len(processed_images_eigenfaces),
                        "lbp": len(processed_images_lbp)
                    },
                    "model_version": self.model_version,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                print(f"[ERROR] Error en entrenamiento incremental: {e}")
                # FALLBACK: Intentar reentrenamiento completo
                print(f"[ML] Intentando reentrenamiento completo como fallback...")
                return self._train_from_database_safely()

        except Exception as e:
            print(f"[ERROR] Error crítico en add_new_person: {e}")
            return {
                "status": "error",
                "message": f"Error crítico: {str(e)}",
                "person_id": person_id,
                "timestamp": datetime.now().isoformat()
            }

    def recognize_face(self, image: np.ndarray, method: str = "hybrid") -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Reconoce un rostro con manejo adecuado de tipos de datos
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        print(f"🔍 Iniciando reconocimiento con método: {method}")
        print(f"🔍 Imagen entrada: {image.shape}, dtype: {image.dtype}")

        # PASO 1: Preprocesar imagen de manera robusta
        try:
            # Obtener imagen base procesada (float64)
            base_processed = self.preprocess_image_for_training(image)

            if base_processed is None:
                return {
                    "recognized": False,
                    "person_id": None,
                    "confidence": 0.0,
                    "error": "No se pudo procesar la imagen",
                    "method": method,
                    "timestamp": datetime.now().isoformat()
                }

            print(f"✅ Imagen base procesada: {base_processed.shape}, dtype: {base_processed.dtype}")

        except Exception as e:
            print(f"❌ Error en preprocesamiento: {e}")
            return {
                "recognized": False,
                "person_id": None,
                "confidence": 0.0,
                "error": f"Error en preprocesamiento: {str(e)}",
                "method": method,
                "timestamp": datetime.now().isoformat()
            }

        # PASO 2: Realizar reconocimiento según el método
        try:
            if method == "eigenfaces":
                return self._recognize_eigenfaces_only(base_processed)
            elif method == "lbp":
                return self._recognize_lbp_only(base_processed)
            elif method == "voting":
                return self._recognize_voting(base_processed)
            else:  # hybrid (default)
                return self._recognize_hybrid(base_processed)

        except Exception as e:
            print(f"❌ Error en reconocimiento {method}: {e}")
            return {
                "recognized": False,
                "person_id": None,
                "confidence": 0.0,
                "error": f"Error en algoritmo {method}: {str(e)}",
                "method": method,
                "timestamp": datetime.now().isoformat()
            }

    def _recognize_hybrid(self, base_processed: np.ndarray) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Reconocimiento híbrido con tipos de datos adecuados
        """
        print(f"🔍 Reconocimiento híbrido con imagen: {base_processed.shape}")

        results = {
            "eigenfaces": None,
            "lbp": None,
            "errors": []
        }

        # Intentar Eigenfaces (usar imagen float64 directamente)
        try:
            eigen_person_id, eigen_confidence, eigen_details = self.eigenfaces_service.recognize_face(base_processed)
            results["eigenfaces"] = (eigen_person_id, eigen_confidence, eigen_details)
            print(f"✅ Eigenfaces: ID={eigen_person_id}, conf={eigen_confidence:.2f}")
        except Exception as e:
            error_msg = f"Error en Eigenfaces: {str(e)}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            results["eigenfaces"] = (-1, 0.0, {"error": error_msg})

        # Intentar LBP (convertir a uint8)
        try:
            lbp_input = (base_processed * 255).astype(np.uint8)
            lbp_person_id, lbp_confidence, lbp_details = self.lbp_service.recognize_face(lbp_input)
            results["lbp"] = (lbp_person_id, lbp_confidence, lbp_details)
            print(f"✅ LBP: ID={lbp_person_id}, conf={lbp_confidence:.2f}")
        except Exception as e:
            error_msg = f"Error en LBP: {str(e)}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            results["lbp"] = (-1, 0.0, {"error": error_msg})

        # Combinar resultados
        eigen_result = results["eigenfaces"]
        lbp_result = results["lbp"]

        # Si ambos fallaron
        if eigen_result[0] == -1 and lbp_result[0] == -1:
            return {
                "recognized": False,
                "person_id": None,
                "confidence": 0.0,
                "method": "hybrid",
                "errors": results["errors"],
                "details": {
                    "eigenfaces": eigen_result[2],
                    "lbp": lbp_result[2]
                },
                "timestamp": datetime.now().isoformat()
            }

        # Combinar con promedio ponderado
        return self._combine_weighted_average(eigen_result, lbp_result)

    def _recognize_eigenfaces_only(self, processed_face: np.ndarray) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Reconocimiento solo con Eigenfaces
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
        ✅ CORREGIDO: Reconocimiento solo con LBP
        """
        # Convertir a uint8 para LBP
        lbp_input = (processed_face * 255).astype(np.uint8)
        person_id, confidence, details = self.lbp_service.recognize_face(lbp_input)

        return {
            "recognized": person_id != -1,
            "person_id": person_id if person_id != -1 else None,
            "confidence": confidence,
            "method": "lbp",
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

    def _combine_weighted_average(self, eigen_result: Tuple, lbp_result: Tuple) -> Dict[str, Any]:
        """
        ✅ MANTENIDO: Combina resultados usando promedio ponderado
        """
        eigen_person_id, eigen_confidence, eigen_details = eigen_result
        lbp_person_id, lbp_confidence, lbp_details = lbp_result

        print(
            f"🔄 Combinando: Eigen(ID={eigen_person_id}, conf={eigen_confidence:.2f}), LBP(ID={lbp_person_id}, conf={lbp_confidence:.2f})")

        # Manejar casos donde uno o ambos algoritmos fallaron
        valid_eigen = eigen_person_id != -1
        valid_lbp = lbp_person_id != -1

        if not valid_eigen and not valid_lbp:
            return {
                "recognized": False,
                "person_id": None,
                "confidence": 0.0,
                "method": "weighted_average",
                "consensus": False,
                "details": {
                    "eigenfaces": eigen_details,
                    "lbp": lbp_details,
                    "combination_status": "both_failed"
                },
                "timestamp": datetime.now().isoformat()
            }

        elif not valid_eigen:
            final_person_id = lbp_person_id
            final_confidence = lbp_confidence * 0.8
            combination_status = "lbp_only"
            consensus = False

        elif not valid_lbp:
            final_person_id = eigen_person_id
            final_confidence = eigen_confidence * 0.8
            combination_status = "eigenfaces_only"
            consensus = False

        else:
            # Ambos funcionaron
            if eigen_person_id == lbp_person_id:
                final_person_id = eigen_person_id
                final_confidence = (eigen_confidence * self.eigenfaces_weight +
                                    lbp_confidence * self.lbp_weight)
                consensus = True
                combination_status = "consensus"
            else:
                eigen_weighted = eigen_confidence * self.eigenfaces_weight
                lbp_weighted = lbp_confidence * self.lbp_weight

                if eigen_weighted > lbp_weighted:
                    final_person_id = eigen_person_id
                    final_confidence = eigen_weighted * 0.9
                else:
                    final_person_id = lbp_person_id
                    final_confidence = lbp_weighted * 0.9

                consensus = False
                combination_status = "no_consensus"

        return {
            "recognized": final_person_id != -1 and final_confidence >= self.confidence_threshold,
            "person_id": final_person_id if final_person_id != -1 else None,
            "confidence": round(final_confidence, 2),
            "method": "weighted_average",
            "consensus": consensus,
            "combination_status": combination_status,
            "weights": {
                "eigenfaces": self.eigenfaces_weight,
                "lbp": self.lbp_weight
            },
            "details": {
                "eigenfaces": eigen_details,
                "lbp": lbp_details,
                "valid_eigenfaces": valid_eigen,
                "valid_lbp": valid_lbp
            },
            "timestamp": datetime.now().isoformat()
        }

    def _train_from_database_safely(self) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Entrena desde la base de datos de forma segura
        """
        try:
            print("[ML] Iniciando entrenamiento seguro desde base de datos...")

            from config.database import SessionLocal
            from models.database_models import Usuario, ImagenFacial
            import cv2

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
                            try:
                                if os.path.exists(imagen.ruta_archivo):
                                    img = cv2.imread(imagen.ruta_archivo)
                                    if img is not None:
                                        user_images.append(img)
                            except Exception as e:
                                print(f"[WARNING] Error leyendo imagen {imagen.ruta_archivo}: {e}")
                                continue

                        if user_images:
                            images_by_person[usuario.id] = user_images

                if len(images_by_person) < 2:
                    return {
                        "status": "insufficient_data",
                        "message": f"Insuficientes usuarios para entrenar. Disponibles: {len(images_by_person)}, Requeridos: 2",
                        "timestamp": datetime.now().isoformat()
                    }

                # Entrenar modelo completo
                print(f"[ML] Entrenando con {len(images_by_person)} usuarios...")
                training_stats = self.train_models(images_by_person)

                return {
                    "status": "trained_from_database",
                    "message": "Modelo entrenado exitosamente desde base de datos",
                    "training_stats": training_stats,
                    "timestamp": datetime.now().isoformat()
                }

            finally:
                db.close()

        except Exception as e:
            print(f"[ERROR] Error en entrenamiento desde BD: {e}")
            return {
                "status": "training_failed",
                "message": f"Error en entrenamiento desde BD: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _save_training_record(self, training_stats: Dict[str, Any]):
        """
        Guarda registro del entrenamiento en BD con métricas calculadas
        """
        from config.database import SessionLocal
        from models.database_models import ModeloEntrenamiento

        db = SessionLocal()
        try:
            # Calcular precisión promedio basada en los stats
            precision_promedio = "0.00"

            if "model_performance" in training_stats:
                hybrid_accuracy = training_stats["model_performance"].get("hybrid", {}).get("accuracy", 0)
                eigenfaces_accuracy = training_stats["model_performance"].get("eigenfaces", {}).get("accuracy", 0)
                lbp_accuracy = training_stats["model_performance"].get("lbp", {}).get("accuracy", 0)

                # Calcular promedio de las tres precisiones
                avg_precision = (hybrid_accuracy + eigenfaces_accuracy + lbp_accuracy) / 3
                precision_promedio = f"{avg_precision:.4f}"

            training_record = ModeloEntrenamiento(
                version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                algoritmo="hybrid_fixed",
                total_usuarios=training_stats.get("unique_persons", 0),
                total_imagenes=training_stats.get("total_images", 0),
                precision_promedio=precision_promedio,
                ruta_modelo_eigenfaces="storage/models/eigenfaces_model.pkl",
                ruta_modelo_lbp="storage/models/lbp_model.pkl",
                configuracion={
                    "eigenfaces_components": training_stats.get("eigenfaces_info", {}).get("n_components", 0),
                    "lbp_radius": training_stats.get("lbp_info", {}).get("radius", 0),
                    "lbp_points": training_stats.get("lbp_info", {}).get("n_points", 0),
                    "training_stats": training_stats,
                    "model_version": self.model_version,
                    "data_types_fixed": training_stats.get("data_types_used", {}),
                    "precision_detail": {
                        "hybrid": training_stats.get("model_performance", {}).get("hybrid", {}).get("accuracy", 0),
                        "eigenfaces": training_stats.get("model_performance", {}).get("eigenfaces", {}).get("accuracy",
                                                                                                            0),
                        "lbp": training_stats.get("model_performance", {}).get("lbp", {}).get("accuracy", 0)
                    }
                }
            )

            db.add(training_record)
            db.commit()

            print(f"✅ Registro de entrenamiento guardado: {training_record.version} - Precisión: {precision_promedio}")

        except Exception as e:
            db.rollback()
            print(f"❌ Error guardando registro de entrenamiento: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            db.close()

    def get_system_info(self) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Obtiene información completa del sistema
        """
        return {
            "system_info": {
                "is_trained": self.is_trained,
                "model_version": self.model_version,
                "combination_method": self.combination_method,
                "confidence_threshold": self.confidence_threshold,
                "training_sessions": len(self.training_history),
                "data_type_handling": "Separado por algoritmo (Eigenfaces: float64, LBP: uint8)"
            },
            "eigenfaces_info": self.eigenfaces_service.get_model_info(),
            "lbp_info": self.lbp_service.get_model_info(),
            "weights": {
                "eigenfaces": self.eigenfaces_weight,
                "lbp": self.lbp_weight
            },
            "last_training": self.training_history[-1] if self.training_history else None,
            "fixes_applied": [
                "Separación de tipos de datos por algoritmo",
                "Eigenfaces: float64 [0,1] para PCA",
                "LBP: uint8 [0,255] para CLAHE",
                "Manejo robusto de errores en entrenamiento incremental",
                "Fallback a entrenamiento completo si falla incremental"
            ]
        }

    def load_models(self) -> bool:
        """
        ✅ CORREGIDO: Carga modelos previamente entrenados
        """
        try:
            self.eigenfaces_service.load_model()
            self.lbp_service.load_model()

            self.is_trained = (self.eigenfaces_service.is_trained and
                               self.lbp_service.is_trained)

            if self.is_trained:
                print("✅ Modelos cargados exitosamente")
                print(f"📊 Eigenfaces: {len(self.eigenfaces_service.trained_embeddings)} embeddings")
                print(f"📊 LBP: {len(self.lbp_service.trained_histograms)} histogramas")
            else:
                print("⚠️ Los modelos no están completamente entrenados")

            return self.is_trained

        except Exception as e:
            print(f"❌ Error al cargar modelos: {e}")
            return False

    def force_retrain_from_database(self) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Fuerza un reentrenamiento completo desde la base de datos
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

    def check_training_requirements(self) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Verifica los requisitos para entrenamiento
        """
        from config.database import SessionLocal
        from models.database_models import Usuario, ImagenFacial

        db = SessionLocal()
        try:
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
                "auto_training_enabled": self.auto_training_enabled,
                "model_version": self.model_version
            }

            return requirements

        finally:
            db.close()

    def get_training_status(self) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Obtiene el estado actual del entrenamiento
        """
        requirements = self.check_training_requirements()

        status = {
            "model_trained": self.is_trained,
            "auto_training_enabled": self.auto_training_enabled,
            "training_requirements": requirements,
            "system_ready": self.is_trained or requirements["can_train"],
            "recommendation": self._get_training_recommendation(requirements),
            "model_version": self.model_version,
            "fixes_status": "✅ Tipos de datos corregidos para ambos algoritmos"
        }

        return status

    def _get_training_recommendation(self, requirements: Dict[str, Any]) -> str:
        """
        ✅ CORREGIDO: Obtiene recomendación sobre el entrenamiento
        """
        if self.is_trained:
            return f"✅ Modelo entrenado y listo para uso (versión {self.model_version})"
        elif requirements["can_train"]:
            return "🎓 Datos suficientes - Se puede entrenar automáticamente"
        else:
            needed = requirements["min_required"] - requirements["users_with_images"]
            return f"⏳ Se necesitan {needed} usuarios más con imágenes para entrenar"

    def _adaptive_fusion(self, eigen_result: dict, lbp_result: dict) -> dict:
        """
        Fusión adaptativa que ajusta pesos según confianza individual
        """
        eigen_conf = eigen_result.get("confidence", 0)
        lbp_conf = lbp_result.get("confidence", 0)
        eigen_id = eigen_result.get("person_id", -1)
        lbp_id = lbp_result.get("person_id", -1)

        # CASO 1: Ambos coinciden
        if eigen_id == lbp_id and eigen_id != -1:
            total_conf = eigen_conf + lbp_conf
            weight_eigen = eigen_conf / total_conf if total_conf > 0 else 0.5
            weight_lbp = lbp_conf / total_conf if total_conf > 0 else 0.5

            final_confidence = weight_eigen * eigen_conf + weight_lbp * lbp_conf
            final_confidence = min(final_confidence * MLConfig.CONSENSUS_BONUS, 100)

            return {
                "person_id": eigen_id,
                "confidence": final_confidence,
                "method": "adaptive_consensus",
                "details": {
                    "eigen_weight": round(weight_eigen, 3),
                    "lbp_weight": round(weight_lbp, 3),
                    "consensus": True
                }
            }

        # CASO 2: No coinciden pero ambos reconocieron
        elif eigen_id != -1 and lbp_id != -1:
            if eigen_conf > lbp_conf:
                final_id = eigen_id
                final_conf = eigen_conf * MLConfig.CONFLICT_PENALTY
                winner = "eigenfaces"
            else:
                final_id = lbp_id
                final_conf = lbp_conf * MLConfig.CONFLICT_PENALTY
                winner = "lbp"

            return {
                "person_id": final_id,
                "confidence": final_conf,
                "method": "adaptive_best",
                "details": {
                    "winner": winner,
                    "consensus": False
                }
            }

        # CASO 3: Solo uno reconoció
        elif eigen_id != -1:
            return {
                "person_id": eigen_id,
                "confidence": eigen_conf * 0.9,
                "method": "eigenfaces_only"
            }
        elif lbp_id != -1:
            return {
                "person_id": lbp_id,
                "confidence": lbp_conf * 0.9,
                "method": "lbp_only"
            }

        # CASO 4: Ninguno reconoció
        return {
            "person_id": -1,
            "confidence": 0,
            "method": "no_recognition"
        }

    def preprocess_image_with_quality_check(self, image: np.ndarray) -> tuple:
        """
        Preprocesa imagen con verificación de calidad
        Retorna: (imagen_procesada, quality_metrics)
        """
        # Verificar calidad
        quality_metrics = self.quality_checker.check_image_quality(image)

        # Intentar alinear si está disponible
        aligned_image = image
        if self.alignment_available and MLConfig.USE_FACE_ALIGNMENT:
            aligned = self.face_aligner.align_face(image)
            if aligned is not None:
                aligned_image = aligned
                quality_metrics["face_aligned"] = True
            else:
                quality_metrics["face_aligned"] = False

        # Preprocesar con método avanzado si está habilitado
        if MLConfig.USE_ADVANCED_ILLUMINATION:
            processed = self.image_preprocessor.preprocess_with_advanced_illumination(aligned_image)
        else:
            processed = self.preprocess_image_for_training(aligned_image)

        return processed, quality_metrics


    def benchmark_algorithms(self, test_images: List[Tuple[np.ndarray, int]]) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Evalúa el rendimiento de los diferentes algoritmos
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        results = {
            "eigenfaces": {"correct": 0, "total": 0, "confidences": []},
            "lbp": {"correct": 0, "total": 0, "confidences": []},
            "hybrid": {"correct": 0, "total": 0, "confidences": []}
        }

        for image, true_label in test_images:
            # Preprocesar imagen base
            base_processed = self.preprocess_image_for_training(image)
            if base_processed is None:
                continue

            # Test Eigenfaces
            eigen_result = self._recognize_eigenfaces_only(base_processed)
            results["eigenfaces"]["total"] += 1
            results["eigenfaces"]["confidences"].append(eigen_result["confidence"])
            if eigen_result["person_id"] == true_label:
                results["eigenfaces"]["correct"] += 1

            # Test LBP
            lbp_result = self._recognize_lbp_only(base_processed)
            results["lbp"]["total"] += 1
            results["lbp"]["confidences"].append(lbp_result["confidence"])
            if lbp_result["person_id"] == true_label:
                results["lbp"]["correct"] += 1

            # Test Hybrid
            hybrid_result = self._recognize_hybrid(base_processed)
            results["hybrid"]["total"] += 1
            results["hybrid"]["confidences"].append(hybrid_result["confidence"])
            if hybrid_result["person_id"] == true_label:
                results["hybrid"]["correct"] += 1

        # Calcular métricas
        for method in results:
            if results[method]["total"] > 0:
                accuracy = results[method]["correct"] / results[method]["total"]
                avg_confidence = np.mean(results[method]["confidences"])
                results[method]["accuracy"] = accuracy
                results[method]["average_confidence"] = avg_confidence
            else:
                results[method]["accuracy"] = 0
                results[method]["average_confidence"] = 0

        return results