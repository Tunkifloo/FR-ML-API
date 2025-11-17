import cv2
import numpy as np
import os
from typing import Tuple, List, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import json
from datetime import datetime


class ImageProcessor:
    """
    Procesador de imágenes especializado para reconocimiento facial
    """

    def __init__(self):
        """
        Inicializa el procesador de imágenes
        """
        # Configuración de procesamiento
        self.target_size = (100, 100)  # Tamaño estándar para procesamiento
        self.quality_threshold = 50  # Umbral mínimo de calidad

        # Configuración de mejoras
        self.auto_enhance = True
        self.noise_reduction = True
        self.contrast_enhancement = True

        # Métricas de calidad
        self.quality_metrics = {}

    def process_face_image(self, image: np.ndarray, enhance: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Procesa una imagen facial completa

        Args:
            image: Imagen original
            enhance: Si aplicar mejoras automáticas

        Returns:
            Tupla (imagen_procesada, métricas_calidad)
        """
        print("🖼️ Procesando imagen facial...")

        # Métricas iniciales
        initial_metrics = self.calculate_image_quality(image)

        # Pipeline de procesamiento
        processed_image = image.copy()
        processing_steps = []

        # Paso 1: Corrección de orientación
        processed_image, rotation_info = self.correct_orientation(processed_image)
        if rotation_info["rotated"]:
            processing_steps.append(f"Rotación: {rotation_info['angle']}°")

        # Paso 2: Redimensionado
        processed_image = self.resize_image(processed_image, self.target_size)
        processing_steps.append(f"Redimensionado a {self.target_size}")

        # Paso 3: Mejoras (si están habilitadas)
        if enhance and self.auto_enhance:
            processed_image, enhancement_info = self.enhance_image(processed_image)
            processing_steps.extend(enhancement_info["applied_enhancements"])

        # Paso 4: Reducción de ruido
        if self.noise_reduction:
            processed_image = self.reduce_noise(processed_image)
            processing_steps.append("Reducción de ruido")

        # Paso 5: Normalización
        processed_image = self.normalize_image(processed_image)
        processing_steps.append("Normalización")

        # Métricas finales
        final_metrics = self.calculate_image_quality(processed_image)

        # Compilar información del procesamiento
        processing_info = {
            "initial_quality": initial_metrics,
            "final_quality": final_metrics,
            "quality_improvement": final_metrics["overall_quality"] - initial_metrics["overall_quality"],
            "processing_steps": processing_steps,
            "processing_timestamp": datetime.now().isoformat()
        }

        print(
            f"✅ Procesamiento completado. Calidad: {initial_metrics['overall_quality']:.1f} → {final_metrics['overall_quality']:.1f}")

        return processed_image, processing_info

    def calculate_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de calidad de la imagen
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        metrics = {}

        # 1. Nitidez (Varianza del Laplaciano)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics["sharpness"] = min(100, laplacian_var / 10)

        # 2. Contraste (Desviación estándar)
        contrast = gray.std()
        metrics["contrast"] = min(100, contrast * 2)

        # 3. Brillo (Media, penalizar extremos)
        brightness = gray.mean()
        brightness_score = 100 - abs(brightness - 128) / 1.28
        metrics["brightness"] = max(0, brightness_score)

        # 4. Uniformidad de iluminación
        # Dividir en bloques y calcular variación de brillo
        h, w = gray.shape
        block_size = 32
        block_means = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i + block_size, j:j + block_size]
                block_means.append(block.mean())

        if block_means:
            illumination_uniformity = 100 - (np.std(block_means) * 2)
            metrics["illumination_uniformity"] = max(0, illumination_uniformity)
        else:
            metrics["illumination_uniformity"] = 50

        # 5. Nivel de ruido (usando filtro de mediana)
        denoised = cv2.medianBlur(gray, 5)
        noise_level = np.mean(np.abs(gray.astype(float) - denoised.astype(float)))
        noise_score = max(0, 100 - noise_level * 4)
        metrics["noise_level"] = noise_score

        # 6. Resolución efectiva (basada en gradientes)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        resolution_score = min(100, np.mean(gradient_magnitude) / 2)
        metrics["resolution"] = resolution_score

        # Puntuación general (promedio ponderado)
        weights = {
            "sharpness": 0.25,
            "contrast": 0.20,
            "brightness": 0.15,
            "illumination_uniformity": 0.20,
            "noise_level": 0.10,
            "resolution": 0.10
        }

        overall_quality = sum(metrics[key] * weights[key] for key in weights.keys())
        metrics["overall_quality"] = overall_quality

        return metrics

    def correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Corrige la orientación de la imagen basándose en características faciales
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detectar ojos para determinar orientación
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        rotation_info = {"rotated": False, "angle": 0, "confidence": 0}
        best_image = image.copy()
        best_eye_count = 0

        # Probar diferentes ángulos de rotación
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = gray
                rotated_color = image
            else:
                # Rotar imagen
                center = (gray.shape[1] // 2, gray.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, rotation_matrix, (gray.shape[1], gray.shape[0]))
                rotated_color = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

            # Detectar ojos en la imagen rotada
            eyes = eye_cascade.detectMultiScale(rotated, 1.1, 5)
            eye_count = len(eyes)

            if eye_count > best_eye_count:
                best_eye_count = eye_count
                best_image = rotated_color
                rotation_info = {
                    "rotated": angle != 0,
                    "angle": angle,
                    "confidence": eye_count
                }

        return best_image, rotation_info

    import numpy as np

    def convert_numpy_types(data):
        """
        Recorre recursivamente un diccionario o lista y convierte todos los
        tipos de NumPy (np.bool_, np.int64, np.float64, etc.)
        a tipos nativos de Python (bool, int, float) que son JSON serializables.
        """
        if isinstance(data, dict):
            # Es un diccionario, ¡limpiemos dentro!~
            return {k: convert_numpy_types(v) for k, v in data.items()}

        elif isinstance(data, list):
            # Es una lista, ¡limpiemos cada cosita!~
            return [convert_numpy_types(item) for item in data]

        elif isinstance(data, np.bool_):
            # ¡Un bool raro! Conviértelo a bool normalito
            return bool(data)

        elif isinstance(data, np.integer):
            # ¡Un int raro! Conviértelo a int normalito
            return int(data)

        elif isinstance(data, np.floating):
            # ¡Un float raro! Conviértelo a float normalito
            return float(data)

        elif isinstance(data, np.ndarray):
            # ¡Un array! Conviértelo a lista normalita
            return data.tolist()

        else:
            # Es un tipo normalito, ¡déjalo así!~
            return data

    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        ✅ CORREGIDO: Redimensiona la imagen "estirándola" (stretch/warp)
        para que coincida con el tamaño exacto requerido por Eigenfaces/LBP,
        sin mantener la proporción ni añadir padding.
        """

        # Simplemente redimensiona (estira) la imagen al tamaño objetivo
        # INTER_AREA es robusto para reducir imágenes
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        return resized

    def enhance_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Mejora la imagen usando técnicas avanzadas
        """
        enhanced = image.copy()
        applied_enhancements = []

        # Convertir a PIL para algunas operaciones
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        # 1. Mejora de contraste adaptativa
        if self.contrast_enhancement:
            if len(image.shape) == 3:
                # Para imágenes en color, aplicar CLAHE por canal
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Para escala de grises
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(enhanced)

            applied_enhancements.append("Mejora de contraste adaptativa (CLAHE)")

        # 2. Corrección gamma automática
        gamma = self.calculate_optimal_gamma(enhanced)
        if abs(gamma - 1.0) > 0.1:  # Solo aplicar si hay diferencia significativa
            enhanced = self.adjust_gamma(enhanced, gamma)
            applied_enhancements.append(f"Corrección gamma (γ={gamma:.2f})")

        # 3. Filtro de nitidez
        if self.calculate_image_quality(enhanced)["sharpness"] < 60:
            # Aplicar filtro de nitidez unsharp mask
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            applied_enhancements.append("Filtro de nitidez (Unsharp Mask)")

        # 4. Balanceador de blancos (para imágenes en color)
        if len(image.shape) == 3:
            enhanced = self.white_balance(enhanced)
            applied_enhancements.append("Balance de blancos")

        enhancement_info = {
            "applied_enhancements": applied_enhancements,
            "total_enhancements": len(applied_enhancements)
        }

        return enhanced, enhancement_info

    def calculate_optimal_gamma(self, image: np.ndarray) -> float:
        """
        Calcula el valor gamma óptimo para corrección de iluminación
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calcular el percentil 50 (mediana) de la imagen
        median_brightness = np.median(gray)

        # Gamma óptimo basado en la mediana
        # Si la imagen es muy oscura, gamma < 1 (aclarar)
        # Si la imagen es muy clara, gamma > 1 (oscurecer)
        target_median = 128  # Valor objetivo

        if median_brightness > 0:
            gamma = np.log(target_median / 255.0) / np.log(median_brightness / 255.0)
            # Limitar gamma a un rango razonable
            gamma = np.clip(gamma, 0.4, 2.5)
        else:
            gamma = 1.0

        return gamma

    def adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Aplica corrección gamma a la imagen
        """
        # Crear tabla de lookup para gamma
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # Aplicar corrección gamma
        return cv2.LUT(image, table)

    def white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica balance de blancos a la imagen
        """
        # Método Gray World
        b, g, r = cv2.split(image.astype(np.float32))

        # Calcular promedios por canal
        b_avg = np.mean(b)
        g_avg = np.mean(g)
        r_avg = np.mean(r)

        # Calcular factor de escala
        gray_avg = (b_avg + g_avg + r_avg) / 3

        # Aplicar corrección
        b = b * (gray_avg / b_avg) if b_avg > 0 else b
        g = g * (gray_avg / g_avg) if g_avg > 0 else g
        r = r * (gray_avg / r_avg) if r_avg > 0 else r

        # Combinar canales y convertir de vuelta a uint8
        balanced = cv2.merge([b, g, r])
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)

        return balanced

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce el ruido en la imagen
        """
        if len(image.shape) == 3:
            # Para imágenes en color, usar filtro bilateral
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            # Para escala de grises, usar Non-Local Means
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

        return denoised

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza la imagen para consistencia
        """
        # Normalización min-max
        normalized = image.astype(np.float32)

        if len(image.shape) == 3:
            # Normalizar cada canal por separado
            for i in range(3):
                channel = normalized[:, :, i]
                min_val = np.min(channel)
                max_val = np.max(channel)

                if max_val > min_val:
                    normalized[:, :, i] = ((channel - min_val) / (max_val - min_val)) * 255
        else:
            min_val = np.min(normalized)
            max_val = np.max(normalized)

            if max_val > min_val:
                normalized = ((normalized - min_val) / (max_val - min_val)) * 255

        return normalized.astype(np.uint8)

    def batch_process_images(self, image_paths: List[str], output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Procesa un lote de imágenes
        """
        if not output_dir:
            output_dir = "storage/processed_images"

        os.makedirs(output_dir, exist_ok=True)

        results = []

        for i, image_path in enumerate(image_paths):
            try:
                print(f"📸 Procesando imagen {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

                # Cargar imagen
                image = cv2.imread(image_path)
                if image is None:
                    results.append({
                        "input_path": image_path,
                        "success": False,
                        "error": "No se pudo cargar la imagen"
                    })
                    continue

                # Procesar imagen
                processed_image, processing_info = self.process_face_image(image)

                # Guardar imagen procesada
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_processed{ext}")

                cv2.imwrite(output_path, processed_image)

                # Guardar resultado
                result = {
                    "input_path": image_path,
                    "output_path": output_path,
                    "success": True,
                    "processing_info": processing_info
                }
                results.append(result)

            except Exception as e:
                results.append({
                    "input_path": image_path,
                    "success": False,
                    "error": str(e)
                })

        # Guardar reporte del lote
        report_path = os.path.join(output_dir, "batch_processing_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_images": len(image_paths),
                "successful": sum(1 for r in results if r["success"]),
                "failed": sum(1 for r in results if not r["success"]),
                "results": results
            }, f, indent=2)

        print(f"✅ Procesamiento en lote completado. Reporte: {report_path}")

        return results

    def create_face_thumbnail(self, image: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Crea una miniatura optimizada de la imagen facial
        """
        # Procesar imagen básica
        processed, _ = self.process_face_image(image, enhance=True)

        # Redimensionar a thumbnail
        thumbnail = cv2.resize(processed, size, interpolation=cv2.INTER_AREA)

        return thumbnail

    def compare_image_quality(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """
        Compara la calidad entre dos imágenes
        """
        metrics1 = self.calculate_image_quality(image1)
        metrics2 = self.calculate_image_quality(image2)

        comparison = {
            "image1_quality": metrics1,
            "image2_quality": metrics2,
            "quality_differences": {},
            "better_image": None
        }

        # Calcular diferencias
        for metric in metrics1.keys():
            if metric in metrics2:
                diff = metrics2[metric] - metrics1[metric]
                comparison["quality_differences"][metric] = {
                    "difference": diff,
                    "improvement": diff > 0
                }

        # Determinar imagen con mejor calidad general
        if metrics1["overall_quality"] > metrics2["overall_quality"]:
            comparison["better_image"] = "image1"
        elif metrics2["overall_quality"] > metrics1["overall_quality"]:
            comparison["better_image"] = "image2"
        else:
            comparison["better_image"] = "equal"

        return comparison

    def detect_image_problems(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta problemas comunes en imágenes faciales
        """
        problems = []
        metrics = self.calculate_image_quality(image)

        # Problema: Imagen borrosa
        if metrics["sharpness"] < 30:
            problems.append({
                "type": "blur",
                "severity": "high" if metrics["sharpness"] < 15 else "medium",
                "description": "La imagen está borrosa o desenfocada",
                "recommendation": "Usar una cámara con mejor enfoque o aumentar la iluminación"
            })

        # Problema: Bajo contraste
        if metrics["contrast"] < 25:
            problems.append({
                "type": "low_contrast",
                "severity": "high" if metrics["contrast"] < 15 else "medium",
                "description": "La imagen tiene bajo contraste",
                "recommendation": "Mejorar la iluminación o ajustar la configuración de la cámara"
            })

        # Problema: Iluminación inadecuada
        if metrics["brightness"] < 20 or metrics["brightness"] > 80:
            severity = "high" if metrics["brightness"] < 10 or metrics["brightness"] > 90 else "medium"
            problems.append({
                "type": "lighting",
                "severity": severity,
                "description": "Iluminación muy oscura o muy brillante" if metrics[
                                                                               "brightness"] < 50 else "Imagen sobreexpuesta",
                "recommendation": "Ajustar la iluminación ambiental o la exposición de la cámara"
            })

        # Problema: Iluminación desigual
        if metrics["illumination_uniformity"] < 30:
            problems.append({
                "type": "uneven_lighting",
                "severity": "medium",
                "description": "Iluminación desigual en la imagen",
                "recommendation": "Usar iluminación más uniforme o difusa"
            })

        # Problema: Ruido excesivo
        if metrics["noise_level"] < 40:
            problems.append({
                "type": "noise",
                "severity": "high" if metrics["noise_level"] < 20 else "medium",
                "description": "Exceso de ruido en la imagen",
                "recommendation": "Mejorar la iluminación o usar una cámara con mejor sensor"
            })

        # Problema: Resolución baja
        if metrics["resolution"] < 30:
            problems.append({
                "type": "low_resolution",
                "severity": "high",
                "description": "Resolución o detalle insuficiente",
                "recommendation": "Usar una cámara de mayor resolución o acercarse más al sujeto"
            })

        return problems

    def generate_processing_report(self, image_path: str, output_dir: str = "storage/reports") -> str:
        """
        Genera un reporte detallado del procesamiento de una imagen
        """
        os.makedirs(output_dir, exist_ok=True)

        # Cargar y procesar imagen
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        processed_image, processing_info = self.process_face_image(original_image)

        # Detectar problemas
        problems = self.detect_image_problems(original_image)

        # Crear reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(image_path)
        report_filename = f"processing_report_{filename}_{timestamp}.md"
        report_path = os.path.join(output_dir, report_filename)

        # Generar contenido del reporte
        report_content = f"""# Reporte de Procesamiento de Imagen

**Archivo:** {filename}  
**Fecha de procesamiento:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Ruta original:** {image_path}

## 📊 Métricas de Calidad

### Imagen Original
- **Calidad general:** {processing_info['initial_quality']['overall_quality']:.1f}/100
- **Nitidez:** {processing_info['initial_quality']['sharpness']:.1f}/100
- **Contraste:** {processing_info['initial_quality']['contrast']:.1f}/100  
- **Brillo:** {processing_info['initial_quality']['brightness']:.1f}/100
- **Uniformidad de iluminación:** {processing_info['initial_quality']['illumination_uniformity']:.1f}/100
- **Nivel de ruido:** {processing_info['initial_quality']['noise_level']:.1f}/100
- **Resolución:** {processing_info['initial_quality']['resolution']:.1f}/100

### Imagen Procesada
- **Calidad general:** {processing_info['final_quality']['overall_quality']:.1f}/100
- **Nitidez:** {processing_info['final_quality']['sharpness']:.1f}/100
- **Contraste:** {processing_info['final_quality']['contrast']:.1f}/100
- **Brillo:** {processing_info['final_quality']['brightness']:.1f}/100
- **Uniformidad de iluminación:** {processing_info['final_quality']['illumination_uniformity']:.1f}/100
- **Nivel de ruido:** {processing_info['final_quality']['noise_level']:.1f}/100
- **Resolución:** {processing_info['final_quality']['resolution']:.1f}/100

### Mejora Total
**{processing_info['quality_improvement']:+.1f} puntos** ({"✅ Mejorada" if processing_info['quality_improvement'] > 0 else "⚠️ Sin mejora significativa"})

## 🔧 Procesamiento Aplicado

"""

        for i, step in enumerate(processing_info['processing_steps'], 1):
            report_content += f"{i}. {step}\n"

        if problems:
            report_content += f"""
## ⚠️ Problemas Detectados

"""
            for problem in problems:
                severity_icon = "🔴" if problem["severity"] == "high" else "🟡"
                report_content += f"""### {severity_icon} {problem['type'].replace('_', ' ').title()}
**Severidad:** {problem['severity'].title()}  
**Descripción:** {problem['description']}  
**Recomendación:** {problem['recommendation']}

"""
        else:
            report_content += """
## ✅ Sin Problemas Detectados

La imagen tiene una calidad aceptable para reconocimiento facial.
"""

        # Recomendaciones generales
        report_content += f"""
## 💡 Recomendaciones

"""

        final_quality = processing_info['final_quality']['overall_quality']

        if final_quality >= 80:
            report_content += "- ✅ **Excelente calidad** - La imagen es óptima para reconocimiento facial\n"
        elif final_quality >= 60:
            report_content += "- ✅ **Buena calidad** - La imagen es adecuada para reconocimiento facial\n"
        elif final_quality >= 40:
            report_content += "- ⚠️ **Calidad regular** - Se recomienda mejorar la imagen si es posible\n"
        else:
            report_content += "- ❌ **Calidad baja** - Se recomienda tomar una nueva imagen\n"

        report_content += f"""
- Para mejores resultados, asegúrese de:
  - Iluminación uniforme y adecuada
  - Rostro completamente visible y enfocado
  - Fondo simple y sin distracciones
  - Resolución mínima de 224x224 píxeles
  - Evitar sombras fuertes o reflejos

---
*Reporte generado automáticamente por ImageProcessor v1.0*
"""

        # Guardar reporte
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📄 Reporte generado: {report_path}")
        return report_path