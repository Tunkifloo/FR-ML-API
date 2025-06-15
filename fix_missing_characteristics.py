#!/usr/bin/env python3
"""
Script para generar características faltantes en la BD
Ejecutar una vez para arreglar usuarios sin características
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import SessionLocal
from models.database_models import Usuario, ImagenFacial, CaracteristicasFaciales
from services.ml_service import MLService
import cv2
from datetime import datetime


def fix_missing_characteristics():
    """
    Genera características para usuarios que no las tienen
    """
    db = SessionLocal()
    ml_service = MLService()

    try:
        # Cargar modelos ML
        print("[INFO] Cargando modelos ML...")
        ml_service.load_models()

        if not ml_service.is_trained:
            print("[ERROR] Modelos ML no están entrenados. Ejecuta primero el entrenamiento.")
            return

        print("[SUCCESS] Modelos ML cargados correctamente")

        # Encontrar imágenes sin características
        imagenes_sin_caracteristicas = db.query(ImagenFacial).filter(
            ImagenFacial.activa == True,
            ~ImagenFacial.id.in_(
                db.query(CaracteristicasFaciales.imagen_id).filter(
                    CaracteristicasFaciales.activa == True
                )
            )
        ).all()

        print(f"[INFO] Encontradas {len(imagenes_sin_caracteristicas)} imágenes sin características")

        if not imagenes_sin_caracteristicas:
            print("[SUCCESS] Todas las imágenes ya tienen características")
            return

        # Procesar cada imagen
        procesadas = 0
        errores = 0

        for imagen in imagenes_sin_caracteristicas:
            try:
                print(f"[PROCESS] Procesando imagen {imagen.id} (usuario {imagen.usuario_id})")

                # Verificar que el archivo existe
                if not os.path.exists(imagen.ruta_archivo):
                    print(f"[WARNING] Archivo no encontrado: {imagen.ruta_archivo}")
                    errores += 1
                    continue

                # Leer imagen
                img = cv2.imread(imagen.ruta_archivo)
                if img is None:
                    print(f"[WARNING] No se pudo leer imagen: {imagen.ruta_archivo}")
                    errores += 1
                    continue

                # Extraer características
                eigenfaces_features = None
                lbp_features = None

                try:
                    eigenfaces_features = ml_service.eigenfaces_service.extract_features(img)
                    print(f"[SUCCESS] Eigenfaces extraído para imagen {imagen.id}")
                except Exception as e:
                    print(f"[WARNING] Error extrayendo Eigenfaces: {e}")

                try:
                    lbp_features = ml_service.lbp_service.extract_lbp_features(img)
                    print(f"[SUCCESS] LBP extraído para imagen {imagen.id}")
                except Exception as e:
                    print(f"[WARNING] Error extrayendo LBP: {e}")

                # Guardar características si se extrajo al menos una
                if eigenfaces_features is not None or lbp_features is not None:
                    caracteristicas = CaracteristicasFaciales(
                        usuario_id=imagen.usuario_id,
                        imagen_id=imagen.id,
                        eigenfaces_vector=eigenfaces_features.tolist() if eigenfaces_features is not None else None,
                        lbp_histogram=lbp_features.tolist() if lbp_features is not None else None,
                        algoritmo_version="2.0",
                        calidad_deteccion=85
                    )

                    db.add(caracteristicas)
                    db.commit()

                    print(f"[SUCCESS] Características guardadas para imagen {imagen.id}")
                    procesadas += 1
                else:
                    print(f"[ERROR] No se pudieron extraer características para imagen {imagen.id}")
                    errores += 1

            except Exception as e:
                print(f"[ERROR] Error procesando imagen {imagen.id}: {e}")
                errores += 1
                continue

        print(f"\n[SUMMARY] RESUMEN:")
        print(f"  - Imágenes procesadas: {procesadas}")
        print(f"  - Errores: {errores}")
        print(f"  - Total: {len(imagenes_sin_caracteristicas)}")

        if procesadas > 0:
            print(f"[SUCCESS] Se generaron características para {procesadas} imágenes")

    except Exception as e:
        print(f"[ERROR] Error general: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    fix_missing_characteristics()