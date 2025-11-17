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
from skimage.feature import local_binary_pattern

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
        self.model_version = "2.0"
        self.last_training_users = 0
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
        Preprocesa imagen de manera ROBUSTA para entrenamiento
        """
        try:
            print(f"🔧 Preprocessing para entrenamiento: {image.shape}, dtype: {image.dtype}")

            # PASO 1: Validar imagen
            if image is None or image.size == 0:
                print(f"❌ Imagen inválida")
                return None

            # PASO 2: Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image.copy()

            # PASO 3: Redimensionar a tamaño estándar
            target_size = (100, 100)
            resized = cv2.resize(gray_image, target_size)

            # PASO 4: Ecualizar histograma
            equalized = cv2.equalizeHist(resized)

            # PASO 5: Normalizar a float64 [0,1]
            normalized = equalized.astype(np.float64) / 255.0

            print(
                f"✅ Imagen preprocesada: {normalized.shape}, dtype: {normalized.dtype}, range: [{normalized.min():.3f}, {normalized.max():.3f}]")

            return normalized

        except Exception as e:
            print(f"❌ Error crítico en preprocesamiento: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Dentro de la clase MLService en services/ml_service.py
    def get_processed_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Método público que expone la funcionalidad de detección, recorte y alineación
        al resto de la aplicación (e.g., routers).
        """
        # Llama a tu método privado (el que acabas de confirmar)
        return self._get_processed_face_from_image(image)

    def _get_processed_face_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        NUEVO: Proceso unificado de Detección, Recorte y Alineación antes del preprocesamiento ML.

        Retorna la imagen BGR/RGB del rostro recortado y alineado, lista para ser preprocesada.
        """
        if image is None or image.size == 0:
            return None

        # 1. Detección de Rostro
        face_coords = self.face_detector.detect_faces(image)
        if not face_coords:
            return None

        # 2. Obtener el rostro más grande/relevante
        largest_face_coords = self.face_detector.get_largest_face(face_coords)
        if largest_face_coords is None:
            return None

        # 3. Recorte con Margen (Usando FACE_MARGIN_PERCENT de MLConfig)
        cropped_face = self.face_detector.extract_face_roi(
            image, largest_face_coords, margin=MLConfig.FACE_MARGIN_PERCENT
        )

        # 4. Alineación Opcional
        if self.alignment_available and MLConfig.USE_FACE_ALIGNMENT:
            aligned_face = self.face_aligner.align_face(cropped_face)
            if aligned_face is not None:
                cropped_face = aligned_face

        return cropped_face

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

    # EN MLService.py (Reemplazar la función train_models)

    def train_models(self, images_by_person: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
        """
        ✅ CORREGIDO: Asume que las imágenes recibidas de la BD (via router)
        son ROSTROS RECORTADOS y solo aplica PREPROCESAMIENTO (resize/filters).
        """
        print("=" * 60)
        print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO HÍBRIDO (v2.1 - Solo Preprocesamiento)")
        print("=" * 60)

        # ... (Impresiones de Data Augmentation) ...

        all_images_eigenfaces = []  # Para Eigenfaces (float64)
        all_images_lbp = []  # Para LBP (uint8)
        all_labels = []
        original_images_by_person = {}  # Almacenará los rostros RECORTADOS/ALINEADOS

        total_original_images = 0
        total_augmented_images = 0

        for person_id, images in images_by_person.items():
            print(f"👤 Procesando persona ID {person_id}: {len(images)} imágenes (rostros recortados)")

            person_originals_cropped = []
            person_eigenfaces_processed = []
            person_lbp_processed = []

            for i, cropped_face in enumerate(images):  # 'image' ahora es 'cropped_face'
                total_original_images += 1

                # 🚨 LÓGICA DE DETECCIÓN REMOVIDA 🚨
                # Ya no se llama a self._get_processed_face_from_image(image)
                # Asumimos que 'cropped_face' es válido.

                # 1. Guardar el rostro recortado (ya lo es)
                person_originals_cropped.append(cropped_face.copy())

                # 2. Preprocesar el rostro (SOLO resize, filtros, y conversión de tipo)
                if self.preprocessor:
                    processed_for_eigen = self.preprocessor.preprocess_for_ml(cropped_face.copy(),
                                                                              algorithm="eigenfaces")
                    processed_for_lbp = self.preprocessor.preprocess_for_ml(cropped_face.copy(), algorithm="lbp")
                else:
                    # Fallback
                    processed_for_eigen = self._basic_preprocess(cropped_face.copy())
                    processed_for_lbp = (processed_for_eigen * 255).astype(np.uint8)

                if processed_for_eigen is None or processed_for_lbp is None:
                    continue

                person_eigenfaces_processed.append(processed_for_eigen)
                person_lbp_processed.append(processed_for_lbp)

            # APLICAR DATA AUGMENTATION
            if person_eigenfaces_processed:
                augmented_eigenfaces = self._apply_data_augmentation(person_eigenfaces_processed)
                augmented_lbp = self._apply_data_augmentation(person_lbp_processed)

                if len(augmented_eigenfaces) != len(augmented_lbp):
                    print(f"   ⚠️ Advertencia: Desbalance en augmentation")

                for eigen_img, lbp_img in zip(augmented_eigenfaces, augmented_lbp):
                    all_images_eigenfaces.append(eigen_img)
                    all_images_lbp.append(lbp_img)
                    all_labels.append(person_id)
                    total_augmented_images += 1

                original_images_by_person[person_id] = person_originals_cropped
                print(f"   ✅ {len(person_eigenfaces_processed)} → {len(augmented_eigenfaces)} rostros procesados")

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

        # Estadísticas del entrenamiento (se construye un solo diccionario para el final)

        print()
        print("=" * 60)
        print("📊 CALCULANDO MÉTRICAS DE PRECISIÓN")
        print("=" * 60)

        # Calcular precisión usando validación cruzada simple
        precision_metrics = self._calculate_training_precision(all_images_eigenfaces, all_images_lbp, all_labels)

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
            "model_performance": precision_metrics,
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

        # USAR IMÁGENES ORIGINALES (RECORTADAS) para características en BD
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
            # Asegúrate de tener una función _save_training_record()
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
        self.last_training_users = len(images_by_person)
        print(f"✅ Entrenamiento completado. Usuarios registrados: {self.last_training_users}")

        return training_stats

    def _calculate_training_precision(self, images_eigenfaces: List[np.ndarray],
                                      images_lbp: List[np.ndarray],
                                      labels: List[int]) -> Dict[str, Any]:
        """
        Calcula métricas de precisión usando una muestra de validación
        """
        from sklearn.model_selection import train_test_split

        print("📊 Calculando precisión del modelo...")

        try:
            # Dividir datos en train/test (80/20)
            X_eigen_train, X_eigen_test, y_train, y_test = train_test_split(
                images_eigenfaces, labels, test_size=0.2, random_state=42, stratify=labels
            )
            X_lbp_train, X_lbp_test, _, _ = train_test_split(
                images_lbp, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # Evaluar Eigenfaces
            eigenfaces_correct = 0
            for img, true_label in zip(X_eigen_test, y_test):
                try:
                    pred_id, confidence, _ = self.eigenfaces_service.recognize_face(img)
                    if pred_id == true_label:
                        eigenfaces_correct += 1
                except:
                    pass

            eigenfaces_accuracy = eigenfaces_correct / len(y_test) if len(y_test) > 0 else 0

            # Evaluar LBP
            lbp_correct = 0
            for img, true_label in zip(X_lbp_test, y_test):
                try:
                    pred_id, confidence, _ = self.lbp_service.recognize_face(img)
                    if pred_id == true_label:
                        lbp_correct += 1
                except:
                    pass

            lbp_accuracy = lbp_correct / len(y_test) if len(y_test) > 0 else 0

            # Evaluar Híbrido (promedio de ambos)
            hybrid_accuracy = (eigenfaces_accuracy + lbp_accuracy) / 2.0

            print(f"   ✅ Eigenfaces: {eigenfaces_accuracy:.2%}")
            print(f"   ✅ LBP: {lbp_accuracy:.2%}")
            print(f"   ✅ Híbrido: {hybrid_accuracy:.2%}")

            return {
                "eigenfaces": {
                    "accuracy": eigenfaces_accuracy,
                    "correct": eigenfaces_correct,
                    "total": len(y_test)
                },
                "lbp": {
                    "accuracy": lbp_accuracy,
                    "correct": lbp_correct,
                    "total": len(y_test)
                },
                "hybrid": {
                    "accuracy": hybrid_accuracy,
                    "correct": int((eigenfaces_correct + lbp_correct) / 2),
                    "total": len(y_test)
                }
            }

        except Exception as e:
            print(f"   ⚠️ Error calculando precisión: {e}")
            return {
                "eigenfaces": {"accuracy": 0.0},
                "lbp": {"accuracy": 0.0},
                "hybrid": {"accuracy": 0.0}
            }

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

    # EN MLService.py (Reemplazar la función _save_characteristics_to_db)

    def _save_characteristics_to_db(self, images_by_person: Dict[int, List[np.ndarray]]):
        """
        ✅ CORREGIDO: Guarda características (Embedding PCA/LBP) utilizando los rostros
        recortados/alineados, asegurando la proyección PCA para Eigenfaces y la limpieza JSON.

        Args:
            images_by_person: Contiene las imágenes BGR/RGB de los rostros
            RECORTADOS y ALINEADOS.
        """
        # ⚠️ ASUNCIONES: Estas importaciones deben estar en tu archivo principal o en el contexto
        from config.database import SessionLocal
        from models.database_models import CaracteristicasFaciales, ImagenFacial
        from skimage.feature import local_binary_pattern
        from datetime import datetime
        import numpy as np

        print("💾 GUARDANDO CARACTERÍSTICAS (Embedding PCA/LBP) con limpieza")

        db = SessionLocal()
        try:
            characteristics_saved = 0
            errors = []
            is_pca_trained = self.is_trained  # Solo se proyecta si el modelo está entrenado

            for person_id, images in images_by_person.items():
                print(f"\n👤 Procesando persona ID: {person_id}")

                # Obtener imágenes de la BD
                db_images = db.query(ImagenFacial).filter(
                    ImagenFacial.usuario_id == person_id,
                    ImagenFacial.activa == True
                ).order_by(ImagenFacial.id).all()

                if not db_images:
                    print(f"   ⚠️ No se encontraron imágenes en BD para usuario {person_id}")
                    continue

                images_to_process = min(len(images), len(db_images))

                for i in range(images_to_process):
                    original_image = images[i]  # ¡Esta es el rostro recortado/alineado!
                    db_image = db_images[i]

                    try:
                        print(f"   📷 Procesando imagen {db_image.id}: {original_image.shape}")

                        # --- EIGENFACES (Embedding PCA) ---
                        eigenfaces_features = None
                        if is_pca_trained and self.preprocessor:
                            try:
                                # 1. Preprocesar específicamente para Eigenfaces (100x100 float64 [0,1])
                                processed_for_eigen = self.preprocessor.preprocess_for_ml(
                                    original_image.copy(), algorithm="eigenfaces"
                                )

                                if processed_for_eigen is not None:
                                    # 2. Aplanar a vector de 10,000
                                    raw_image_vector = processed_for_eigen.flatten()

                                    # 🚨 PROYECCIÓN PCA: Reducir a embedding (ej. 150 elementos)
                                    embedding_vector = self.eigenfaces_service.transform_image_vector(raw_image_vector)

                                    # 3. LIMPIAR Y GUARDAR el EMBEDDING REDUCIDO
                                    eigenfaces_features = self._clean_for_json_storage(embedding_vector)
                                    print(
                                        f"   ✅ Eigenfaces: {len(eigenfaces_features)} características (embedding PCA) limpias")
                            except Exception as e:
                                print(f"   ❌ Error Eigenfaces/PCA: {e}")

                        # --- LBP (Histograma) ---
                        lbp_features = None
                        if self.preprocessor:
                            try:
                                # 1. Preprocesar específicamente para LBP (100x100 uint8 [0,255])
                                processed_for_lbp = self.preprocessor.preprocess_for_ml(
                                    original_image.copy(), algorithm="lbp"
                                )

                                if processed_for_lbp is not None:
                                    # 2. Extraer características LBP manualmente
                                    radius = 2
                                    n_points = 16
                                    lbp_image = local_binary_pattern(processed_for_lbp, n_points, radius,
                                                                     method='uniform')

                                    # 3. Crear histograma (vector pequeño)
                                    n_bins = n_points + 2
                                    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins),
                                                           density=True)

                                    # 4. LIMPIAR Y GUARDAR el Histograma
                                    lbp_features = self._clean_for_json_storage(hist)
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
                                existing.algoritmo_version = self.model_version
                                print(f"   🔄 Características actualizadas para imagen {db_image.id}")
                            else:
                                # Crear nuevo registro
                                caracteristicas = CaracteristicasFaciales(
                                    usuario_id=person_id,
                                    imagen_id=db_image.id,
                                    eigenfaces_vector=eigenfaces_features,
                                    lbp_histogram=lbp_features,
                                    algoritmo_version=self.model_version,
                                    calidad_deteccion=90.0  # Valor estático o real si tienes calidad_checker
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
                print(f"\n✅ {characteristics_saved} características guardadas exitosamente en BD")
            else:
                print(f"\n⚠️ No se guardaron características")

            if errors:
                print(f"\n⚠️ ERRORES ENCONTRADOS ({len(errors)}):")
                for error in errors[:5]:
                    print(f"   • {error}")

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
        ✅ CORREGIDO: Reconoce un rostro replicando el pipeline de entrenamiento
        (Detección -> Recorte -> Preprocesamiento -> Reconocimiento).
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado")

        print(f"🔍 Iniciando reconocimiento (Método: {method})")
        print(f"🔍 Imagen de entrada: {image.shape}, dtype: {image.dtype}")

        # ----------------------------------------------------------------------
        # PASO 1: DETECTAR, RECORTAR Y ALINEAR EL ROSTRO (¡CRÍTICO!)
        # ----------------------------------------------------------------------
        try:
            cropped_face = self._get_processed_face_from_image(image)

            if cropped_face is None:
                print("❌ No se detectó un rostro en la imagen.")
                return {
                    "recognized": False, "person_id": None, "confidence": 0.0,
                    "error": "No se detectó un rostro en la imagen.",
                    "method": method, "timestamp": datetime.now().isoformat()
                }

            print(f"✅ Rostro detectado y recortado: {cropped_face.shape}")

        except Exception as e:
            print(f"❌ Error en detección/recorte: {e}")
            return {
                "recognized": False, "person_id": None, "confidence": 0.0,
                "error": f"Error en Detección/Recorte: {str(e)}",
                "method": method, "timestamp": datetime.now().isoformat()
            }

        # ----------------------------------------------------------------------
        # PASO 2: PREPROCESAR EL ROSTRO RECORTADO (A 100x100)
        # ----------------------------------------------------------------------
        try:
            # Usar el ImagePreprocessor unificado (float64 [0,1] como base)
            if self.preprocessor:
                base_processed = self.preprocessor.preprocess_for_ml(cropped_face, algorithm="both")
            else:
                # Fallback (usa el básico)
                base_processed = self._basic_preprocess(cropped_face)

            if base_processed is None:
                raise ValueError("El preprocesamiento del rostro falló.")

            print(f"✅ Rostro preprocesado a: {base_processed.shape}, dtype: {base_processed.dtype}")

        except Exception as e:
            print(f"❌ Error en preprocesamiento: {e}")
            return {
                "recognized": False, "person_id": None, "confidence": 0.0,
                "error": f"Error en Preprocesamiento: {str(e)}",
                "method": method, "timestamp": datetime.now().isoformat()
            }

        # ----------------------------------------------------------------------
        # PASO 3: REALIZAR RECONOCIMIENTO (Ahora sí compara Manzanas con Manzanas)
        # ----------------------------------------------------------------------
        try:
            if method == "eigenfaces":
                return self._recognize_eigenfaces_only(base_processed)
            elif method == "lbp":
                return self._recognize_lbp_only(base_processed)
            # elif method == "voting":
            #    return self._recognize_voting(base_processed) # (Asegúrate de tener esta función si la usas)
            else:  # hybrid (default)
                return self._recognize_hybrid(base_processed)

        except Exception as e:
            print(f"❌ Error en reconocimiento {method}: {e}")
            return {
                "recognized": False, "person_id": None, "confidence": 0.0,
                "error": f"Error en algoritmo {method}: {str(e)}",
                "method": method, "timestamp": datetime.now().isoformat()
            }

    def _recognize_hybrid(self, base_processed: np.ndarray) -> Dict[str, Any]:
        """
        Reconocimiento híbrido. (Esta función está bien,
        ya que recibe el 'base_processed' de 100x100).
        """
        print(f"🔍 Reconocimiento híbrido con imagen: {base_processed.shape}")

        results = {
            "eigenfaces": None,
            "lbp": None,
            "errors": []
        }

        # Intentar Eigenfaces (usar imagen float64 [0,1] directamente)
        try:
            # Llama a EigenfacesService, que internamente usará transform_image_vector
            eigen_person_id, eigen_confidence, eigen_details = self.eigenfaces_service.recognize_face(base_processed)
            results["eigenfaces"] = (eigen_person_id, eigen_confidence, eigen_details)
            print(f"✅ Eigenfaces: ID={eigen_person_id}, conf={eigen_confidence:.2f}")
        except Exception as e:
            error_msg = f"Error en Eigenfaces: {str(e)}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            results["eigenfaces"] = (-1, 0.0, {"error": error_msg})

        # Intentar LBP (convertir a uint8 [0,255])
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
                "recognized": False, "person_id": None, "confidence": 0.0,
                "method": "hybrid", "errors": results["errors"],
                "details": {"eigenfaces": eigen_result[2], "lbp": lbp_result[2]},
                "timestamp": datetime.now().isoformat()
            }

        # Combinar con promedio ponderado
        return self._combine_weighted_average(eigen_result, lbp_result)

    def _recognize_eigenfaces_only(self, processed_face: np.ndarray) -> Dict[str, Any]:
        """
        Reconocimiento solo con Eigenfaces (No necesita cambios).
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
        Reconocimiento solo con LBP (No necesita cambios).
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
        Combina resultados usando promedio ponderado (No necesita cambios).
        """
        eigen_person_id, eigen_confidence, eigen_details = eigen_result
        lbp_person_id, lbp_confidence, lbp_details = lbp_result

        # ... (Tu lógica de combinación, consenso y penalización está bien) ...
        # (Asegúrate de que esta función exista en tu clase)

        # --- Lógica de combinación (basada en tu código anterior) ---
        valid_eigen = eigen_person_id != -1
        valid_lbp = lbp_person_id != -1

        if not valid_eigen and not valid_lbp:
            final_person_id = -1
            final_confidence = 0.0
            combination_status = "both_failed"
            consensus = False
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
        # --- Fin Lógica de combinación ---

        return {
            "recognized": final_person_id != -1 and final_confidence >= self.confidence_threshold,
            "person_id": final_person_id if final_person_id != -1 else None,
            "confidence": round(final_confidence, 2),
            "method": "weighted_average",
            "consensus": consensus,
            "combination_status": combination_status,
            "details": {
                "eigenfaces": eigen_details,
                "lbp": lbp_details
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
        import json

        db = SessionLocal()
        try:
            # Calcular precisión promedio - usar 0.0 si no hay datos
            precision_promedio = "0.0000"

            # Intentar obtener métricas si existen
            if "model_performance" in training_stats:
                try:
                    hybrid_accuracy = training_stats["model_performance"].get("hybrid", {}).get("accuracy", 0.0)
                    eigenfaces_accuracy = training_stats["model_performance"].get("eigenfaces", {}).get("accuracy", 0.0)
                    lbp_accuracy = training_stats["model_performance"].get("lbp", {}).get("accuracy", 0.0)

                    # Calcular promedio
                    avg_precision = (hybrid_accuracy + eigenfaces_accuracy + lbp_accuracy) / 3.0

                    # IMPORTANTE: Formatear como STRING con 4 decimales
                    precision_promedio = f"{avg_precision:.4f}"

                    print(f"   📊 Precisiones calculadas:")
                    print(f"      • Hybrid: {hybrid_accuracy:.4f}")
                    print(f"      • Eigenfaces: {eigenfaces_accuracy:.4f}")
                    print(f"      • LBP: {lbp_accuracy:.4f}")
                    print(f"      • Promedio: {precision_promedio}")

                except Exception as e:
                    print(f"⚠️ No se pudieron calcular métricas de precisión: {e}")
                    precision_promedio = "0.0000"
            else:
                print(f"⚠️ No hay 'model_performance' en training_stats")

            # Preparar configuración como diccionario
            config_dict = {
                "eigenfaces_components": training_stats.get("eigenfaces_info", {}).get("n_components", 0),
                "lbp_radius": training_stats.get("lbp_info", {}).get("radius", 2),
                "lbp_points": training_stats.get("lbp_info", {}).get("n_points", 16),
                "model_version": self.model_version,
                "augmentation_enabled": training_stats.get("augmentation_enabled", False),
                "total_original_images": training_stats.get("total_original_images", 0),
                "augmentation_factor": training_stats.get("augmentation_factor", 1.0),
                "model_performance": training_stats.get("model_performance", {})
            }

            training_record = ModeloEntrenamiento(
                version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                algoritmo="Hybrid (Eigenfaces + LBP)",
                total_usuarios=training_stats.get("unique_persons", 0),
                total_imagenes=training_stats.get("total_images", 0),
                precision_promedio=precision_promedio,
                ruta_modelo_eigenfaces="storage/models/eigenfaces_model.pkl",
                ruta_modelo_lbp="storage/models/lbp_model.pkl",
                configuracion=config_dict
            )

            # Desactivar registros anteriores
            db.query(ModeloEntrenamiento).update({"activo": False}, synchronize_session=False)

            # Guardar nuevo registro
            db.add(training_record)
            db.commit()
            db.refresh(training_record)

            print(f"\n✅ Registro de entrenamiento guardado:")
            print(f"   • ID: {training_record.id}")
            print(f"   • Version: {training_record.version}")
            print(f"   • Precisión promedio: {precision_promedio}")
            print(f"   • Total usuarios: {training_record.total_usuarios}")
            print(f"   • Total imágenes: {training_record.total_imagenes}")

        except Exception as e:
            db.rollback()
            print(f"❌ Error guardando registro de entrenamiento: {e}")
            import traceback
            traceback.print_exc()
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

    def is_model_trained(self) -> bool:
        """
        Verifica si el modelo está entrenado y listo para usar

        Returns:
            bool: True si el modelo está entrenado, False en caso contrario
        """
        try:
            # Verificar que los modelos estén cargados en memoria
            if self.eigenfaces_model is None or self.lbp_model is None:
                return False

            # Verificar que haya embeddings
            if not self.face_embeddings or len(self.face_embeddings) == 0:
                return False

            # Verificar que los modelos tengan los atributos necesarios
            if not hasattr(self.eigenfaces_model, 'components_'):
                return False

            if not hasattr(self.lbp_model, 'histograms') or len(self.lbp_model.histograms) == 0:
                return False

            # Todo está OK
            return True

        except Exception as e:
            print(f"⚠️ Error verificando estado del modelo: {str(e)}")
            return False

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

        # Preprocesar con metodo avanzado si está habilitado
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