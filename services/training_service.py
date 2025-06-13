import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .ml_service import MLService
from .eigenfaces_service import EigenfacesService
from .lbp_service import LBPService


class TrainingService:
    """
    Servicio especializado en entrenamiento y validación de modelos
    """

    def __init__(self):
        """
        Inicializa el servicio de entrenamiento
        """
        self.ml_service = MLService()
        self.eigenfaces_service = EigenfacesService()
        self.lbp_service = LBPService()

        # Configuración de entrenamiento
        self.validation_split = 0.2  # 20% para validación
        self.random_state = 42

        # Métricas de entrenamiento
        self.training_metrics = {}
        self.validation_results = {}

        # Rutas de almacenamiento
        self.metrics_path = "storage/models/training_metrics.json"
        self.reports_path = "storage/models/training_reports/"

        os.makedirs(self.reports_path, exist_ok=True)

    def prepare_training_data(self, images_by_person: Dict[int, List[np.ndarray]]) -> Tuple[
        List[np.ndarray], List[int], List[np.ndarray], List[int]]:
        """
        Prepara los datos para entrenamiento dividiendo en conjuntos de entrenamiento y validación

        Args:
            images_by_person: Diccionario {person_id: [lista_de_imagenes]}

        Returns:
            Tupla (X_train, y_train, X_val, y_val)
        """
        print("📊 Preparando datos para entrenamiento...")

        all_images = []
        all_labels = []

        # Aplanar datos
        for person_id, images in images_by_person.items():
            for image in images:
                all_images.append(image)
                all_labels.append(person_id)

        # División estratificada
        X_train, X_val, y_train, y_val = train_test_split(
            all_images,
            all_labels,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=all_labels
        )

        print(f"📈 Datos preparados:")
        print(f"  • Entrenamiento: {len(X_train)} imágenes")
        print(f"  • Validación: {len(X_val)} imágenes")
        print(f"  • Personas únicas: {len(set(all_labels))}")

        return X_train, y_train, X_val, y_val

    def train_with_validation(self, images_by_person: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
        """
        Entrena los modelos con validación cruzada

        Args:
            images_by_person: Diccionario con imágenes por persona

        Returns:
            Diccionario con métricas de entrenamiento y validación
        """
        print("🎓 Iniciando entrenamiento con validación...")

        start_time = datetime.now()

        # Preparar datos
        X_train, y_train, X_val, y_val = self.prepare_training_data(images_by_person)

        # Entrenar modelos individuales
        print("🔧 Entrenando Eigenfaces...")
        self.eigenfaces_service.train(X_train, y_train)

        print("🔧 Entrenando LBP...")
        self.lbp_service.train(X_train, y_train)

        # Validar modelos
        print("📊 Validando modelos...")
        validation_results = self._validate_models(X_val, y_val)

        # Calcular métricas finales
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # Compilar resultados
        results = {
            "training_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "training_time_seconds": training_time,
                "total_images": len(X_train) + len(X_val),
                "training_images": len(X_train),
                "validation_images": len(X_val),
                "unique_persons": len(set(y_train + y_val))
            },
            "model_performance": validation_results,
            "eigenfaces_info": self.eigenfaces_service.get_model_info(),
            "lbp_info": self.lbp_service.get_model_info()
        }

        # Guardar métricas
        self._save_training_metrics(results)

        # Generar reporte detallado
        self._generate_training_report(results, X_val, y_val)

        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")

        return results

    def _validate_models(self, X_val: List[np.ndarray], y_val: List[int]) -> Dict[str, Any]:
        """
        Valida los modelos entrenados
        """
        results = {}

        # Validar Eigenfaces
        eigenfaces_predictions = []
        eigenfaces_confidences = []

        for image, true_label in zip(X_val, y_val):
            pred_id, confidence, _ = self.eigenfaces_service.recognize_face(image)
            eigenfaces_predictions.append(pred_id)
            eigenfaces_confidences.append(confidence)

        # Validar LBP
        lbp_predictions = []
        lbp_confidences = []

        for image, true_label in zip(X_val, y_val):
            pred_id, confidence, _ = self.lbp_service.recognize_face(image)
            lbp_predictions.append(pred_id)
            lbp_confidences.append(confidence)

        # Validar sistema híbrido
        hybrid_predictions = []
        hybrid_confidences = []

        for image, true_label in zip(X_val, y_val):
            result = self.ml_service._recognize_hybrid(image)
            hybrid_predictions.append(result.get("person_id", -1))
            hybrid_confidences.append(result.get("confidence", 0))

        # Calcular métricas
        results["eigenfaces"] = self._calculate_metrics(y_val, eigenfaces_predictions, eigenfaces_confidences)
        results["lbp"] = self._calculate_metrics(y_val, lbp_predictions, lbp_confidences)
        results["hybrid"] = self._calculate_metrics(y_val, hybrid_predictions, hybrid_confidences)

        return results

    def _calculate_metrics(self, y_true: List[int], y_pred: List[int], confidences: List[float]) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento
        """
        # Convertir -1 (no reconocido) a una etiqueta especial
        y_pred_clean = [pred if pred != -1 else -999 for pred in y_pred]

        # Calcular accuracy
        correct = sum(1 for true, pred in zip(y_true, y_pred_clean) if true == pred)
        accuracy = correct / len(y_true)

        # Calcular confianza promedio
        avg_confidence = np.mean(confidences)

        # Distribución de confianzas
        confidence_ranges = {
            "0-50": sum(1 for c in confidences if 0 <= c < 50),
            "50-70": sum(1 for c in confidences if 50 <= c < 70),
            "70-85": sum(1 for c in confidences if 70 <= c < 85),
            "85-100": sum(1 for c in confidences if 85 <= c <= 100)
        }

        return {
            "accuracy": round(accuracy, 4),
            "correct_predictions": correct,
            "total_predictions": len(y_true),
            "average_confidence": round(avg_confidence, 2),
            "confidence_distribution": confidence_ranges,
            "recognition_rate": round(sum(1 for pred in y_pred if pred != -1) / len(y_pred), 4)
        }

    def _save_training_metrics(self, results: Dict[str, Any]) -> None:
        """
        Guarda las métricas de entrenamiento
        """
        try:
            # Cargar métricas existentes
            existing_metrics = []
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    existing_metrics = json.load(f)

            # Añadir nuevas métricas
            existing_metrics.append(results)

            # Mantener solo las últimas 10 sesiones
            existing_metrics = existing_metrics[-10:]

            # Guardar
            with open(self.metrics_path, 'w') as f:
                json.dump(existing_metrics, f, indent=2)

            print(f"💾 Métricas guardadas en: {self.metrics_path}")

        except Exception as e:
            print(f"⚠️ Error al guardar métricas: {e}")

    def _generate_training_report(self, results: Dict[str, Any], X_val: List[np.ndarray], y_val: List[int]) -> None:
        """
        Genera un reporte detallado del entrenamiento
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.reports_path, f"training_report_{timestamp}.md")

            # Crear reporte en Markdown
            report = f"""# Reporte de Entrenamiento - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Información General

- **Tiempo de Entrenamiento**: {results['training_info']['training_time_seconds']:.2f} segundos
- **Total de Imágenes**: {results['training_info']['total_images']}
- **Imágenes de Entrenamiento**: {results['training_info']['training_images']}
- **Imágenes de Validación**: {results['training_info']['validation_images']}
- **Personas Únicas**: {results['training_info']['unique_persons']}

## 🤖 Rendimiento de Algoritmos

### Eigenfaces
- **Precisión**: {results['model_performance']['eigenfaces']['accuracy']:.2%}
- **Confianza Promedio**: {results['model_performance']['eigenfaces']['average_confidence']:.2f}%
- **Tasa de Reconocimiento**: {results['model_performance']['eigenfaces']['recognition_rate']:.2%}

### Local Binary Patterns (LBP)
- **Precisión**: {results['model_performance']['lbp']['accuracy']:.2%}
- **Confianza Promedio**: {results['model_performance']['lbp']['average_confidence']:.2f}%
- **Tasa de Reconocimiento**: {results['model_performance']['lbp']['recognition_rate']:.2%}

### Sistema Híbrido
- **Precisión**: {results['model_performance']['hybrid']['accuracy']:.2%}
- **Confianza Promedio**: {results['model_performance']['hybrid']['average_confidence']:.2f}%
- **Tasa de Reconocimiento**: {results['model_performance']['hybrid']['recognition_rate']:.2%}

## 📈 Distribución de Confianzas

### Eigenfaces
"""

            # Añadir distribución de confianzas
            for range_name, count in results['model_performance']['eigenfaces']['confidence_distribution'].items():
                percentage = (count / results['training_info']['validation_images']) * 100
                report += f"- **{range_name}%**: {count} imágenes ({percentage:.1f}%)\n"

            report += f"""
### LBP
"""
            for range_name, count in results['model_performance']['lbp']['confidence_distribution'].items():
                percentage = (count / results['training_info']['validation_images']) * 100
                report += f"- **{range_name}%**: {count} imágenes ({percentage:.1f}%)\n"

            report += f"""
### Sistema Híbrido
"""
            for range_name, count in results['model_performance']['hybrid']['confidence_distribution'].items():
                percentage = (count / results['training_info']['validation_images']) * 100
                report += f"- **{range_name}%**: {count} imágenes ({percentage:.1f}%)\n"

            # Añadir recomendaciones
            best_algorithm = max(
                results['model_performance'].items(),
                key=lambda x: x[1]['accuracy']
            )

            report += f"""
## 🎯 Recomendaciones

- **Mejor Algoritmo**: {best_algorithm[0].title()} (Precisión: {best_algorithm[1]['accuracy']:.2%})
- **Calidad de Datos**: {'Excelente' if best_algorithm[1]['accuracy'] > 0.9 else 'Buena' if best_algorithm[1]['accuracy'] > 0.8 else 'Necesita Mejora'}
"""

            if best_algorithm[1]['accuracy'] < 0.8:
                report += """
### Sugerencias para Mejorar:
- Añadir más imágenes por persona
- Mejorar la calidad de las imágenes (iluminación, resolución)
- Verificar que las imágenes estén bien etiquetadas
- Considerar re-entrenar con diferentes parámetros
"""

            # Guardar reporte
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"📄 Reporte generado: {report_file}")

        except Exception as e:
            print(f"⚠️ Error al generar reporte: {e}")

    def load_training_history(self) -> List[Dict[str, Any]]:
        """
        Carga el historial de entrenamientos
        """
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"⚠️ Error al cargar historial: {e}")
            return []

    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene información del mejor modelo entrenado
        """
        history = self.load_training_history()
        if not history:
            return None

        # Encontrar el modelo con mejor rendimiento
        best_session = max(
            history,
            key=lambda x: max(
                x.get('model_performance', {}).get('hybrid', {}).get('accuracy', 0),
                x.get('model_performance', {}).get('eigenfaces', {}).get('accuracy', 0),
                x.get('model_performance', {}).get('lbp', {}).get('accuracy', 0)
            )
        )

        return best_session

    def compare_training_sessions(self, num_sessions: int = 5) -> Dict[str, Any]:
        """
        Compara las últimas sesiones de entrenamiento
        """
        history = self.load_training_history()
        recent_sessions = history[-num_sessions:] if len(history) >= num_sessions else history

        if not recent_sessions:
            return {"error": "No hay sesiones de entrenamiento disponibles"}

        comparison = {
            "total_sessions": len(recent_sessions),
            "session_comparison": [],
            "trends": {
                "accuracy_trend": [],
                "confidence_trend": [],
                "training_time_trend": []
            }
        }

        for i, session in enumerate(recent_sessions):
            session_info = {
                "session_index": i + 1,
                "timestamp": session['training_info']['start_time'],
                "training_time": session['training_info']['training_time_seconds'],
                "algorithms": {
                    "eigenfaces": session['model_performance']['eigenfaces']['accuracy'],
                    "lbp": session['model_performance']['lbp']['accuracy'],
                    "hybrid": session['model_performance']['hybrid']['accuracy']
                }
            }
            comparison["session_comparison"].append(session_info)

            # Tendencias
            comparison["trends"]["accuracy_trend"].append(session['model_performance']['hybrid']['accuracy'])
            comparison["trends"]["confidence_trend"].append(
                session['model_performance']['hybrid']['average_confidence'])
            comparison["trends"]["training_time_trend"].append(session['training_info']['training_time_seconds'])

        # Calcular tendencias
        if len(comparison["trends"]["accuracy_trend"]) > 1:
            acc_trend = comparison["trends"]["accuracy_trend"]
            comparison["trends"]["accuracy_improving"] = acc_trend[-1] > acc_trend[0]
            comparison["trends"]["accuracy_change"] = acc_trend[-1] - acc_trend[0]

        return comparison

    def export_training_data(self, format: str = "json") -> str:
        """
        Exporta datos de entrenamiento en el formato especificado
        """
        try:
            history = self.load_training_history()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format.lower() == "json":
                export_file = f"storage/models/training_export_{timestamp}.json"
                with open(export_file, 'w') as f:
                    json.dump(history, f, indent=2)

            elif format.lower() == "csv":
                import pandas as pd

                # Aplanar datos para CSV
                flattened_data = []
                for session in history:
                    row = {
                        "timestamp": session['training_info']['start_time'],
                        "training_time": session['training_info']['training_time_seconds'],
                        "total_images": session['training_info']['total_images'],
                        "unique_persons": session['training_info']['unique_persons'],
                        "eigenfaces_accuracy": session['model_performance']['eigenfaces']['accuracy'],
                        "lbp_accuracy": session['model_performance']['lbp']['accuracy'],
                        "hybrid_accuracy": session['model_performance']['hybrid']['accuracy'],
                        "eigenfaces_confidence": session['model_performance']['eigenfaces']['average_confidence'],
                        "lbp_confidence": session['model_performance']['lbp']['average_confidence'],
                        "hybrid_confidence": session['model_performance']['hybrid']['average_confidence']
                    }
                    flattened_data.append(row)

                df = pd.DataFrame(flattened_data)
                export_file = f"storage/models/training_export_{timestamp}.csv"
                df.to_csv(export_file, index=False)

            else:
                raise ValueError(f"Formato no soportado: {format}")

            return export_file

        except Exception as e:
            raise Exception(f"Error al exportar datos: {e}")

    def optimize_hyperparameters(self, images_by_person: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros de los algoritmos
        """
        print("🔧 Optimizando hiperparámetros...")

        # Preparar datos
        X_train, y_train, X_val, y_val = self.prepare_training_data(images_by_person)

        optimization_results = {}

        # Optimizar Eigenfaces (número de componentes)
        print("📊 Optimizando Eigenfaces...")
        eigenfaces_results = []

        for n_components in [50, 100, 150, 200, 250]:
            try:
                eigenfaces_service = EigenfacesService(n_components=n_components)
                eigenfaces_service.train(X_train, y_train)

                # Validar
                predictions = []
                confidences = []

                for image, true_label in zip(X_val, y_val):
                    pred_id, confidence, _ = eigenfaces_service.recognize_face(image)
                    predictions.append(pred_id)
                    confidences.append(confidence)

                accuracy = sum(1 for true, pred in zip(y_val, predictions) if true == pred) / len(y_val)
                avg_confidence = np.mean(confidences)

                eigenfaces_results.append({
                    "n_components": n_components,
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence
                })

                print(f"  • {n_components} componentes: {accuracy:.3f} precisión")

            except Exception as e:
                print(f"  ⚠️ Error con {n_components} componentes: {e}")

        optimization_results["eigenfaces"] = eigenfaces_results

        # Optimizar LBP (radio y puntos)
        print("🔍 Optimizando LBP...")
        lbp_results = []

        for radius in [1, 2, 3]:
            for n_points in [8, 16, 24]:
                try:
                    lbp_service = LBPService(radius=radius, n_points=n_points)
                    lbp_service.train(X_train, y_train)

                    # Validar
                    predictions = []
                    confidences = []

                    for image, true_label in zip(X_val, y_val):
                        pred_id, confidence, _ = lbp_service.recognize_face(image)
                        predictions.append(pred_id)
                        confidences.append(confidence)

                    accuracy = sum(1 for true, pred in zip(y_val, predictions) if true == pred) / len(y_val)
                    avg_confidence = np.mean(confidences)

                    lbp_results.append({
                        "radius": radius,
                        "n_points": n_points,
                        "accuracy": accuracy,
                        "avg_confidence": avg_confidence
                    })

                    print(f"  • R={radius}, P={n_points}: {accuracy:.3f} precisión")

                except Exception as e:
                    print(f"  ⚠️ Error con R={radius}, P={n_points}: {e}")

        optimization_results["lbp"] = lbp_results

        # Encontrar mejores parámetros
        if eigenfaces_results:
            best_eigenfaces = max(eigenfaces_results, key=lambda x: x["accuracy"])
            optimization_results["best_eigenfaces"] = best_eigenfaces

        if lbp_results:
            best_lbp = max(lbp_results, key=lambda x: x["accuracy"])
            optimization_results["best_lbp"] = best_lbp

        print("✅ Optimización de hiperparámetros completada")

        return optimization_results