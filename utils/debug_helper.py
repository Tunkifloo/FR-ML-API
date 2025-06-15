import cv2
import numpy as np
import os
from typing import Dict, Any, List
import json
from datetime import datetime


class DebugHelper:
    """
    Ayudante para debugging del sistema de reconocimiento
    """

    @staticmethod
    def validate_image_pipeline(image_path: str) -> Dict[str, Any]:
        """
        Valida todo el pipeline de procesamiento de una imagen
        """
        results = {
            "file_exists": False,
            "opencv_readable": False,
            "shape": None,
            "dtype": None,
            "face_detection": False,
            "eigenfaces_processing": False,
            "lbp_processing": False,
            "errors": [],
            "warnings": [],
            "processing_times": {}
        }

        try:
            start_time = datetime.now()

            # 1. Verificar archivo
            results["file_exists"] = os.path.exists(image_path)
            if not results["file_exists"]:
                results["errors"].append(f"Archivo no existe: {image_path}")
                return results

            # 2. Leer con OpenCV
            img = cv2.imread(image_path)
            if img is not None:
                results["opencv_readable"] = True
                results["shape"] = img.shape
                results["dtype"] = str(img.dtype)
                results["file_size_mb"] = round(os.path.getsize(image_path) / (1024 * 1024), 2)
            else:
                results["errors"].append("OpenCV no puede leer el archivo")
                return results

            # 3. Validar dimensiones mínimas
            if img.shape[0] < 50 or img.shape[1] < 50:
                results["warnings"].append(f"Imagen muy pequeña: {img.shape}")

            # 4. Detección de rostros
            face_start = datetime.now()
            try:
                from services.face_detection_service import FaceDetectionService
                face_detector = FaceDetectionService()
                faces = face_detector.detect_faces(img)
                results["face_detection"] = len(faces) > 0
                results["faces_detected"] = len(faces)
                results["face_coordinates"] = faces
                results["processing_times"]["face_detection"] = (datetime.now() - face_start).total_seconds()

                if len(faces) == 0:
                    results["warnings"].append("No se detectaron rostros")
                elif len(faces) > 1:
                    results["warnings"].append(f"Múltiples rostros detectados: {len(faces)}")

            except Exception as e:
                results["errors"].append(f"Error detección rostros: {e}")

            # 5. Procesamiento Eigenfaces
            eigen_start = datetime.now()
            try:
                from services.image_preprocessor import ImagePreprocessor
                preprocessor = ImagePreprocessor()
                processed = preprocessor.preprocess_for_ml(img, "eigenfaces")
                results["eigenfaces_processing"] = True
                results["eigenfaces_shape"] = processed.shape
                results["eigenfaces_dtype"] = str(processed.dtype)
                results["eigenfaces_range"] = [float(processed.min()), float(processed.max())]
                results["processing_times"]["eigenfaces"] = (datetime.now() - eigen_start).total_seconds()
            except Exception as e:
                results["errors"].append(f"Error Eigenfaces: {e}")

            # 6. Procesamiento LBP
            lbp_start = datetime.now()
            try:
                processed = preprocessor.preprocess_for_ml(img, "lbp")
                results["lbp_processing"] = True
                results["lbp_shape"] = processed.shape
                results["lbp_dtype"] = str(processed.dtype)
                results["lbp_range"] = [float(processed.min()), float(processed.max())]
                results["processing_times"]["lbp"] = (datetime.now() - lbp_start).total_seconds()
            except Exception as e:
                results["errors"].append(f"Error LBP: {e}")

            # 7. Tiempo total
            results["processing_times"]["total"] = (datetime.now() - start_time).total_seconds()

            # 8. Evaluar calidad general
            results["quality_score"] = DebugHelper._calculate_quality_score(results)

        except Exception as e:
            results["errors"].append(f"Error general: {e}")

        return results

    @staticmethod
    def _calculate_quality_score(results: Dict[str, Any]) -> float:
        """
        Calcula puntuación de calidad de 0-100
        """
        score = 0

        # Archivo legible (20 puntos)
        if results["opencv_readable"]:
            score += 20

        # Detección de rostros (30 puntos)
        if results["face_detection"]:
            score += 30
            # Bonus si es exactamente 1 rostro
            if results.get("faces_detected") == 1:
                score += 10

        # Procesamiento exitoso (40 puntos)
        if results["eigenfaces_processing"]:
            score += 20
        if results["lbp_processing"]:
            score += 20

        # Penalizar errores
        score -= len(results["errors"]) * 10
        score -= len(results["warnings"]) * 5

        return max(0, min(100, score))

    @staticmethod
    def test_all_user_images() -> Dict[str, Any]:
        """
        Prueba todas las imágenes de usuarios
        """
        from config.database import SessionLocal
        from models.database_models import ImagenFacial

        db = SessionLocal()
        try:
            imagenes = db.query(ImagenFacial).filter(ImagenFacial.activa == True).all()

            results = {
                "total_images": len(imagenes),
                "successful": 0,
                "failed": 0,
                "with_warnings": 0,
                "quality_distribution": {"high": 0, "medium": 0, "low": 0},
                "errors": [],
                "summary": {},
                "problematic_images": []
            }

            quality_scores = []
            processing_times = []

            for i, imagen in enumerate(imagenes):
                test_result = DebugHelper.validate_image_pipeline(imagen.ruta_archivo)

                quality_score = test_result.get("quality_score", 0)
                quality_scores.append(quality_score)

                total_time = test_result.get("processing_times", {}).get("total", 0)
                processing_times.append(total_time)

                if len(test_result["errors"]) == 0:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["problematic_images"].append({
                        "image_id": imagen.id,
                        "usuario_id": imagen.usuario_id,
                        "path": imagen.ruta_archivo,
                        "errors": test_result["errors"],
                        "quality_score": quality_score
                    })

                if len(test_result["warnings"]) > 0:
                    results["with_warnings"] += 1

                # Distribución de calidad
                if quality_score >= 80:
                    results["quality_distribution"]["high"] += 1
                elif quality_score >= 50:
                    results["quality_distribution"]["medium"] += 1
                else:
                    results["quality_distribution"]["low"] += 1

                # Progreso
                if (i + 1) % 10 == 0:
                    print(f"Procesadas {i + 1}/{len(imagenes)} imágenes...")

            # Estadísticas finales
            if quality_scores:
                results["summary"] = {
                    "average_quality": round(np.mean(quality_scores), 2),
                    "min_quality": min(quality_scores),
                    "max_quality": max(quality_scores),
                    "average_processing_time": round(np.mean(processing_times), 3),
                    "success_rate": round(results["successful"] / results["total_images"] * 100, 2)
                }

            return results

        finally:
            db.close()

    @staticmethod
    def test_model_recognition() -> Dict[str, Any]:
        """
        Prueba el reconocimiento con imágenes existentes
        """
        from config.database import SessionLocal
        from models.database_models import Usuario, ImagenFacial
        from services.ml_service import MLService
        import cv2

        db = SessionLocal()
        ml_service = MLService()

        try:
            # Verificar que el modelo esté entrenado
            if not ml_service.load_models():
                return {"error": "Modelo no está entrenado"}

            # Obtener usuarios con múltiples imágenes
            usuarios = db.query(Usuario).filter(Usuario.activo == True).all()

            test_results = {
                "total_tests": 0,
                "successful_recognitions": 0,
                "failed_recognitions": 0,
                "average_confidence": 0,
                "per_user_results": [],
                "algorithm_comparison": {
                    "eigenfaces": {"correct": 0, "total": 0, "avg_confidence": 0},
                    "lbp": {"correct": 0, "total": 0, "avg_confidence": 0},
                    "hybrid": {"correct": 0, "total": 0, "avg_confidence": 0}
                }
            }

            all_confidences = []

            for usuario in usuarios:
                imagenes = db.query(ImagenFacial).filter(
                    ImagenFacial.usuario_id == usuario.id,
                    ImagenFacial.activa == True
                ).limit(3).all()  # Máximo 3 imágenes por usuario

                if not imagenes:
                    continue

                user_results = {
                    "usuario_id": usuario.id,
                    "nombre": f"{usuario.nombre} {usuario.apellido}",
                    "total_images": len(imagenes),
                    "correct_recognitions": 0,
                    "avg_confidence": 0,
                    "tests": []
                }

                user_confidences = []

                for imagen in imagenes:
                    if not os.path.exists(imagen.ruta_archivo):
                        continue

                    img = cv2.imread(imagen.ruta_archivo)
                    if img is None:
                        continue

                    # Probar con diferentes algoritmos
                    for algorithm in ["eigenfaces", "lbp", "hybrid"]:
                        try:
                            result = ml_service.recognize_face(img, method=algorithm)

                            test_results["total_tests"] += 1
                            test_results["algorithm_comparison"][algorithm]["total"] += 1

                            recognized_id = result.get("person_id")
                            confidence = result.get("confidence", 0)

                            is_correct = (recognized_id == usuario.id)

                            if is_correct:
                                test_results["successful_recognitions"] += 1
                                test_results["algorithm_comparison"][algorithm]["correct"] += 1
                                user_results["correct_recognitions"] += 1
                            else:
                                test_results["failed_recognitions"] += 1

                            all_confidences.append(confidence)
                            user_confidences.append(confidence)

                            user_results["tests"].append({
                                "imagen_id": imagen.id,
                                "algorithm": algorithm,
                                "recognized_id": recognized_id,
                                "expected_id": usuario.id,
                                "correct": is_correct,
                                "confidence": confidence
                            })

                        except Exception as e:
                            print(f"Error probando {algorithm} en imagen {imagen.id}: {e}")

                if user_confidences:
                    user_results["avg_confidence"] = round(np.mean(user_confidences), 2)

                test_results["per_user_results"].append(user_results)

            # Calcular estadísticas finales
            if all_confidences:
                test_results["average_confidence"] = round(np.mean(all_confidences), 2)

            # Estadísticas por algoritmo
            for algorithm in test_results["algorithm_comparison"]:
                algo_stats = test_results["algorithm_comparison"][algorithm]
                if algo_stats["total"] > 0:
                    algo_stats["accuracy"] = round(algo_stats["correct"] / algo_stats["total"] * 100, 2)
                else:
                    algo_stats["accuracy"] = 0

            return test_results

        finally:
            db.close()

    @staticmethod
    def export_debug_report(output_path: str = "storage/debug_report.json") -> str:
        """
        Exporta un reporte completo de debugging
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "image_validation": DebugHelper.test_all_user_images(),
            "model_recognition": DebugHelper.test_model_recognition()
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return output_path