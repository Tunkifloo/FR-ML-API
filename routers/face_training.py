from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime
import os

from config.database import get_db
from models.database_models import Usuario, ImagenFacial, ModeloEntrenamiento
from models.pydantic_models import (
    EntrenamientoRequest, EntrenamientoResponse, ResponseWithData,
    AlgoritmoReconocimiento, ModeloInfo
)
from services.ml_service import MLService

router = APIRouter(prefix="/entrenamiento", tags=["Entrenamiento ML"])

# Servicio ML global
ml_service = MLService()

# Estado del entrenamiento
training_status = {
    "is_training": False,
    "progress": 0,
    "current_step": "",
    "start_time": None,
    "estimated_finish": None
}


@router.post("/iniciar", response_model=ResponseWithData, summary="Iniciar entrenamiento del modelo")
async def iniciar_entrenamiento(
        request: EntrenamientoRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    """
    Inicia el proceso de entrenamiento del modelo de reconocimiento facial

    - **reentrenar_completo**: Si eliminar modelo anterior y reentrenar desde cero
    - **algoritmos**: Lista de algoritmos a entrenar (hybrid, eigenfaces, lbp)
    """
    try:
        # Verificar si ya hay un entrenamiento en progreso
        if training_status["is_training"]:
            raise HTTPException(
                status_code=409,
                detail="Ya hay un entrenamiento en progreso"
            )

        # Verificar que hay usuarios con imágenes
        usuarios_con_imagenes = db.query(Usuario).filter(
            Usuario.activo == True
        ).join(ImagenFacial).filter(
            ImagenFacial.activa == True
        ).distinct().count()

        if usuarios_con_imagenes < 2:
            raise HTTPException(
                status_code=400,
                detail="Se necesitan al menos 2 usuarios con imágenes para entrenar"
            )

        # Inicializar estado de entrenamiento
        training_status.update({
            "is_training": True,
            "progress": 0,
            "current_step": "Inicializando...",
            "start_time": datetime.now(),
            "estimated_finish": None
        })

        # Ejecutar entrenamiento en background
        background_tasks.add_task(
            ejecutar_entrenamiento_background,
            request,
            db.bind.connect()  # Pasar conexión para background task
        )

        return ResponseWithData(
            success=True,
            message="Entrenamiento iniciado en segundo plano",
            data={
                "training_id": f"train_{int(datetime.now().timestamp())}",
                "usuarios_disponibles": usuarios_con_imagenes,
                "algoritmos_seleccionados": request.algoritmos,
                "reentrenamiento_completo": request.reentrenar_completo,
                "estado": "iniciado"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar entrenamiento: {str(e)}")


async def ejecutar_entrenamiento_background(request: EntrenamientoRequest, db_connection):
    """
    Ejecuta el entrenamiento en segundo plano
    """
    try:
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine
        import cv2

        # Crear nueva sesión para background task
        engine = create_engine(str(db_connection.engine.url))
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()

        try:
            # Paso 1: Recopilar datos
            training_status.update({
                "progress": 10,
                "current_step": "Recopilando imágenes de usuarios..."
            })

            # Obtener usuarios con imágenes
            usuarios = db.query(Usuario).filter(Usuario.activo == True).all()
            images_by_person = {}

            total_images = 0
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
                                total_images += 1

                    if user_images:
                        images_by_person[usuario.id] = user_images

            training_status.update({
                "progress": 30,
                "current_step": f"Procesando {total_images} imágenes de {len(images_by_person)} usuarios..."
            })

            # Paso 2: Entrenar modelos
            if request.reentrenar_completo:
                training_status["current_step"] = "Entrenamiento completo iniciado..."
                training_stats = ml_service.train_models(images_by_person)
            else:
                training_status["current_step"] = "Entrenamiento incremental..."
                # Para entrenamiento incremental, añadir nuevos usuarios
                for person_id, images in images_by_person.items():
                    ml_service.add_new_person(person_id, images)

                training_stats = {
                    "total_images": total_images,
                    "unique_persons": len(images_by_person),
                    "training_type": "incremental"
                }

            training_status.update({
                "progress": 80,
                "current_step": "Guardando modelos entrenados..."
            })

            # Paso 3: Guardar registro en BD
            modelo_registro = ModeloEntrenamiento(
                version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                algoritmo="hybrid",
                total_usuarios=len(images_by_person),
                total_imagenes=total_images,
                precision_promedio="N/A",  # Se calculará después
                configuracion={
                    "algoritmos": [alg.value for alg in request.algoritmos],
                    "reentrenamiento_completo": request.reentrenar_completo,
                    "training_stats": training_stats
                }
            )

            db.add(modelo_registro)
            db.commit()

            # Finalizar
            training_status.update({
                "progress": 100,
                "current_step": "Entrenamiento completado exitosamente",
                "is_training": False
            })

        finally:
            db.close()

    except Exception as e:
        training_status.update({
            "is_training": False,
            "progress": 0,
            "current_step": f"Error: {str(e)}"
        })


@router.get("/estado", response_model=ResponseWithData, summary="Estado del entrenamiento")
async def obtener_estado_entrenamiento():
    """
    Obtiene el estado actual del entrenamiento
    """
    try:
        estado_actual = training_status.copy()

        # Calcular tiempo transcurrido
        if estado_actual["start_time"]:
            elapsed = datetime.now() - estado_actual["start_time"]
            estado_actual["tiempo_transcurrido"] = str(elapsed).split('.')[0]  # Sin microsegundos

        return ResponseWithData(
            success=True,
            message="Estado del entrenamiento obtenido",
            data=estado_actual
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estado: {str(e)}")


@router.post("/detener", response_model=ResponseWithData, summary="Detener entrenamiento")
async def detener_entrenamiento():
    """
    Detiene el entrenamiento en progreso (si existe)
    """
    try:
        if not training_status["is_training"]:
            raise HTTPException(
                status_code=400,
                detail="No hay entrenamiento en progreso"
            )

        # Marcar como detenido
        training_status.update({
            "is_training": False,
            "current_step": "Entrenamiento detenido por el usuario",
            "progress": 0
        })

        return ResponseWithData(
            success=True,
            message="Entrenamiento detenido exitosamente",
            data={"estado": "detenido"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al detener entrenamiento: {str(e)}")


@router.get("/historial", response_model=ResponseWithData, summary="Historial de entrenamientos")
async def historial_entrenamientos(
        limite: int = Query(20, ge=1, le=100, description="Número de entrenamientos a obtener"),
        db: Session = Depends(get_db)
):
    """
    Obtiene el historial de entrenamientos realizados
    """
    try:
        entrenamientos = db.query(ModeloEntrenamiento).order_by(
            ModeloEntrenamiento.fecha_entrenamiento.desc()
        ).limit(limite).all()

        historial_data = []
        for entrenamiento in entrenamientos:
            historial_data.append({
                "id": entrenamiento.id,
                "version": entrenamiento.version,
                "algoritmo": entrenamiento.algoritmo,
                "total_usuarios": entrenamiento.total_usuarios,
                "total_imagenes": entrenamiento.total_imagenes,
                "precision_promedio": entrenamiento.precision_promedio,
                "fecha_entrenamiento": entrenamiento.fecha_entrenamiento.isoformat(),
                "activo": entrenamiento.activo,
                "configuracion": entrenamiento.configuracion
            })

        return ResponseWithData(
            success=True,
            message=f"Historial obtenido ({len(historial_data)} registros)",
            data={
                "total_entrenamientos": len(historial_data),
                "entrenamientos": historial_data
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {str(e)}")


@router.get("/modelos/info", response_model=ResponseWithData, summary="Información de modelos")
async def info_modelos_disponibles():
    """
    Obtiene información detallada de los modelos disponibles
    """
    try:
        # Cargar modelos si no están cargados
        ml_service.load_models()

        # Información del sistema
        system_info = ml_service.get_system_info()

        # Información adicional de archivos
        model_files = {}
        model_dir = "storage/models"

        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(model_dir, filename)
                    stat = os.stat(filepath)
                    model_files[filename] = {
                        "tamaño_mb": round(stat.st_size / (1024 * 1024), 2),
                        "ultima_modificacion": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }

        return ResponseWithData(
            success=True,
            message="Información de modelos obtenida",
            data={
                "system_info": system_info,
                "archivos_modelo": model_files,
                "estado_entrenamiento": training_status
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener información: {str(e)}")


@router.post("/validar", response_model=ResponseWithData, summary="Validar modelo entrenado")
async def validar_modelo(
        porcentaje_prueba: float = Query(0.2, ge=0.1, le=0.5, description="Porcentaje de datos para validación"),
        db: Session = Depends(get_db)
):
    """
    Valida el modelo entrenado usando un conjunto de prueba
    """
    try:
        # Verificar que el modelo esté entrenado
        if not ml_service.load_models():
            raise HTTPException(status_code=503, detail="Modelo no está entrenado")

        # Obtener imágenes para validación
        import cv2
        import random

        usuarios = db.query(Usuario).filter(Usuario.activo == True).all()
        test_images = []

        for usuario in usuarios:
            imagenes = db.query(ImagenFacial).filter(
                ImagenFacial.usuario_id == usuario.id,
                ImagenFacial.activa == True
            ).all()

            # Tomar una muestra para prueba
            num_test = max(1, int(len(imagenes) * porcentaje_prueba))
            imagenes_test = random.sample(imagenes, min(num_test, len(imagenes)))

            for imagen in imagenes_test:
                if os.path.exists(imagen.ruta_archivo):
                    img = cv2.imread(imagen.ruta_archivo)
                    if img is not None:
                        test_images.append((img, usuario.id))

        if not test_images:
            raise HTTPException(status_code=400, detail="No hay imágenes disponibles para validación")

        # Realizar benchmark
        resultados = ml_service.benchmark_algorithms(test_images)

        return ResponseWithData(
            success=True,
            message=f"Validación completada con {len(test_images)} imágenes",
            data={
                "total_pruebas": len(test_images),
                "porcentaje_usado": porcentaje_prueba,
                "resultados_por_algoritmo": resultados,
                "recomendacion": "hybrid" if resultados.get("hybrid", {}).get("accuracy", 0) > 0.8 else "revisar_datos"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en validación: {str(e)}")


@router.delete("/limpiar-modelos", response_model=ResponseWithData, summary="Limpiar modelos antiguos")
async def limpiar_modelos_antiguos(
        mantener_ultimos: int = Query(3, ge=1, le=10, description="Número de modelos recientes a mantener"),
        db: Session = Depends(get_db)
):
    """
    Limpia modelos antiguos manteniendo solo los más recientes
    """
    try:
        # Obtener modelos antiguos
        modelos_antiguos = db.query(ModeloEntrenamiento).order_by(
            ModeloEntrenamiento.fecha_entrenamiento.desc()
        ).offset(mantener_ultimos).all()

        archivos_eliminados = []

        for modelo in modelos_antiguos:
            # Eliminar archivos de modelo si existen
            rutas_posibles = [
                modelo.ruta_modelo_eigenfaces,
                modelo.ruta_modelo_lbp
            ]

            for ruta in rutas_posibles:
                if ruta and os.path.exists(ruta):
                    os.remove(ruta)
                    archivos_eliminados.append(ruta)

            # Marcar como inactivo en lugar de eliminar
            modelo.activo = False

        db.commit()

        return ResponseWithData(
            success=True,
            message=f"Limpieza completada. {len(modelos_antiguos)} modelos marcados como inactivos",
            data={
                "modelos_desactivados": len(modelos_antiguos),
                "archivos_eliminados": len(archivos_eliminados),
                "modelos_activos_restantes": mantener_ultimos
            }
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al limpiar modelos: {str(e)}")