from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import cv2
import numpy as np
import os
import uuid
from datetime import datetime
import json

from config.database import get_db
from models.database_models import Usuario, HistorialReconocimiento
from models.pydantic_models import (
    ReconocimientoResponse, ResponseWithData, AlgoritmoReconocimiento
)
from services.ml_service import MLService
from utils.alert_system import AlertSystem, AlertInfo

router = APIRouter(prefix="/reconocimiento", tags=["Reconocimiento Facial"])

# Inicializar servicios
ml_service = MLService()
alert_system = AlertSystem()

# Configuración
TEMP_DIR = "storage/temp"
os.makedirs(TEMP_DIR, exist_ok=True)


def validate_image_file(file: UploadFile) -> bool:
    """Valida si el archivo es una imagen válida"""
    if not file.filename:
        return False

    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    return file_ext in allowed_extensions


@router.post("/identificar", response_model=ResponseWithData, summary="Identificar persona en imagen")
async def identificar_persona(
        imagen: UploadFile = File(..., description="Imagen a analizar"),
        algoritmo: AlgoritmoReconocimiento = Form(AlgoritmoReconocimiento.HYBRID, description="Algoritmo a usar"),
        incluir_detalles: bool = Form(True, description="Incluir detalles técnicos"),
        request: Request = None,
        db: Session = Depends(get_db)
):
    """
    Identifica una persona en una imagen usando algoritmos de reconocimiento facial

    - **imagen**: Archivo de imagen a analizar
    - **algoritmo**: Algoritmo de reconocimiento (hybrid, eigenfaces, lbp, voting)
    - **incluir_detalles**: Si incluir detalles técnicos del procesamiento

    ⚠️ **IMPORTANTE**: Si se detecta una persona requisitoriada, se generará una alerta de seguridad automáticamente.
    """
    temp_file_path = None

    try:
        # Validar archivo
        if not validate_image_file(imagen):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser una imagen válida (jpg, jpeg, png, bmp)"
            )

        # Verificar que el modelo esté entrenado
        if not ml_service.load_models():
            raise HTTPException(
                status_code=503,
                detail="El modelo de reconocimiento no está entrenado. Entrene el modelo primero."
            )

        # Crear archivo temporal
        file_extension = os.path.splitext(imagen.filename)[1]
        temp_filename = f"recognition_{uuid.uuid4().hex}{file_extension}"
        temp_file_path = os.path.join(TEMP_DIR, temp_filename)

        # Guardar imagen temporalmente
        with open(temp_file_path, "wb") as buffer:
            content = await imagen.read()
            buffer.write(content)

        # Leer imagen con OpenCV
        img = cv2.imread(temp_file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")

        # Obtener IP del cliente
        client_ip = None
        if request:
            client_ip = request.client.host

        # Realizar reconocimiento
        start_time = datetime.now()
        recognition_result = ml_service.recognize_face(img, method=algoritmo.value)
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()

        # Preparar respuesta base
        response_data = {
            "reconocido": recognition_result.get("recognized", False),
            "persona_id": recognition_result.get("person_id"),
            "confianza": round(recognition_result.get("confidence", 0.0), 2),
            "metodo": recognition_result.get("method", algoritmo.value),
            "tiempo_procesamiento": round(processing_time, 3),
            "timestamp": recognition_result.get("timestamp")
        }

        # Incluir detalles técnicos si se solicita
        if incluir_detalles:
            response_data["detalles_tecnicos"] = recognition_result.get("details", {})

        # Variables para información de la persona y alerta
        persona_info = None
        alerta_seguridad = None

        # Si se reconoció una persona, obtener información de la BD
        if response_data["reconocido"] and response_data["persona_id"]:
            usuario = db.query(Usuario).filter(Usuario.id == response_data["persona_id"]).first()

            if usuario:
                persona_info = {
                    "id": usuario.id,
                    "nombre": usuario.nombre,
                    "apellido": usuario.apellido,
                    "id_estudiante": usuario.id_estudiante,
                    "email": usuario.email,
                    "requisitoriado": usuario.requisitoriado,
                    "tipo_requisitoria": usuario.tipo_requisitoria
                }

                # Si la persona está requisitoriada, generar alerta
                if usuario.requisitoriado:
                    alert_info = AlertInfo(
                        person_id=usuario.id,
                        person_name=usuario.nombre,
                        person_lastname=usuario.apellido,
                        student_id=usuario.id_estudiante,
                        requisition_type=usuario.tipo_requisitoria,
                        confidence=response_data["confianza"],
                        detection_timestamp=datetime.now().isoformat(),
                        image_path=temp_file_path,
                        alert_level="",  # Se determinará automáticamente
                        location="Sistema de Reconocimiento Facial",
                        additional_info={
                            "algorithm": algoritmo.value,
                            "processing_time": processing_time,
                            "client_ip": client_ip
                        }
                    )

                    # Generar alerta
                    alerta_seguridad = alert_system.generate_security_alert(alert_info)

        # Registrar en historial
        historial = HistorialReconocimiento(
            usuario_id=response_data["persona_id"],
            imagen_consulta_path=temp_file_path,
            confianza=int(response_data["confianza"]),
            distancia_euclidiana=str(recognition_result.get("details", {}).get("distance", "N/A")),
            reconocido=response_data["reconocido"],
            alerta_generada=alerta_seguridad is not None,
            caracteristicas_consulta=recognition_result.get("details", {}),
            ip_origen=client_ip
        )

        db.add(historial)
        db.commit()

        # Añadir información adicional a la respuesta
        response_data["persona_info"] = persona_info
        response_data["alerta_seguridad"] = alerta_seguridad
        response_data["historial_id"] = historial.id

        # Mensaje de respuesta
        if response_data["reconocido"]:
            if alerta_seguridad:
                mensaje = f"🚨 PERSONA REQUISITORIADA IDENTIFICADA: {persona_info['nombre']} {persona_info['apellido']} - ALERTA GENERADA"
            else:
                mensaje = f"✅ Persona identificada: {persona_info['nombre']} {persona_info['apellido']}"
        else:
            mensaje = "❌ No se pudo identificar a la persona en la imagen"

        return ResponseWithData(
            success=True,
            message=mensaje,
            data=response_data
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el reconocimiento: {str(e)}")

    finally:
        # Limpiar archivo temporal (opcional, mantener para historial)
        if temp_file_path and os.path.exists(temp_file_path):
            # Decidir si mantener o eliminar el archivo temporal
            # Por ahora lo mantenemos para el historial
            pass


@router.get("/historial", response_model=ResponseWithData, summary="Historial de reconocimientos")
async def obtener_historial_reconocimientos(
        pagina: int = Query(1, ge=1, description="Número de página"),
        items_por_pagina: int = Query(20, ge=1, le=100, description="Items por página"),
        usuario_id: Optional[int] = Query(None, description="Filtrar por ID de usuario"),
        reconocido: Optional[bool] = Query(None, description="Filtrar por reconocidos/no reconocidos"),
        alerta_generada: Optional[bool] = Query(None, description="Filtrar por alertas generadas"),
        confianza_minima: Optional[float] = Query(None, ge=0.0, le=100.0, description="Confianza mínima"),
        db: Session = Depends(get_db)
):
    """
    Obtiene el historial de reconocimientos con filtros y paginación
    """
    try:
        # Construir query
        query = db.query(HistorialReconocimiento)

        # Aplicar filtros
        if usuario_id:
            query = query.filter(HistorialReconocimiento.usuario_id == usuario_id)
        if reconocido is not None:
            query = query.filter(HistorialReconocimiento.reconocido == reconocido)
        if alerta_generada is not None:
            query = query.filter(HistorialReconocimiento.alerta_generada == alerta_generada)
        if confianza_minima:
            query = query.filter(HistorialReconocimiento.confianza >= confianza_minima)

        # Ordenar por fecha descendente
        query = query.order_by(HistorialReconocimiento.fecha_reconocimiento.desc())

        # Contar total
        total = query.count()

        # Aplicar paginación
        offset = (pagina - 1) * items_por_pagina
        reconocimientos = query.offset(offset).limit(items_por_pagina).all()

        # Convertir a diccionarios con información del usuario
        historial_data = []
        for rec in reconocimientos:
            rec_data = {
                "id": rec.id,
                "usuario_id": rec.usuario_id,
                "confianza": rec.confianza,
                "distancia_euclidiana": rec.distancia_euclidiana,
                "reconocido": rec.reconocido,
                "alerta_generada": rec.alerta_generada,
                "fecha_reconocimiento": rec.fecha_reconocimiento.isoformat(),
                "ip_origen": rec.ip_origen
            }

            # Añadir información del usuario si existe
            if rec.usuario_id:
                usuario = db.query(Usuario).filter(Usuario.id == rec.usuario_id).first()
                if usuario:
                    rec_data["usuario_info"] = {
                        "nombre": usuario.nombre,
                        "apellido": usuario.apellido,
                        "id_estudiante": usuario.id_estudiante,
                        "requisitoriado": usuario.requisitoriado,
                        "tipo_requisitoria": usuario.tipo_requisitoria
                    }

            historial_data.append(rec_data)

        # Calcular total de páginas
        total_paginas = (total + items_por_pagina - 1) // items_por_pagina

        return ResponseWithData(
            success=True,
            message=f"Historial obtenido exitosamente",
            data={
                "reconocimientos": historial_data,
                "paginacion": {
                    "total": total,
                    "pagina": pagina,
                    "items_por_pagina": items_por_pagina,
                    "total_paginas": total_paginas
                }
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {str(e)}")


@router.get("/estadisticas", response_model=ResponseWithData, summary="Estadísticas de reconocimientos")
async def estadisticas_reconocimientos(
        dias: int = Query(30, ge=1, le=365, description="Días a incluir en las estadísticas"),
        db: Session = Depends(get_db)
):
    """
    Obtiene estadísticas de reconocimientos de los últimos N días
    """
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import func, and_

        # Calcular fecha de inicio
        fecha_inicio = datetime.now() - timedelta(days=dias)

        # Query base para el período
        base_query = db.query(HistorialReconocimiento).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio
        )

        # Estadísticas básicas
        total_reconocimientos = base_query.count()
        reconocimientos_exitosos = base_query.filter(HistorialReconocimiento.reconocido == True).count()
        alertas_generadas = base_query.filter(HistorialReconocimiento.alerta_generada == True).count()

        # Tasa de éxito
        tasa_exito = (reconocimientos_exitosos / total_reconocimientos * 100) if total_reconocimientos > 0 else 0

        # Confianza promedio
        avg_confianza = db.query(func.avg(HistorialReconocimiento.confianza)).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio,
            HistorialReconocimiento.reconocido == True
        ).scalar() or 0

        # Reconocimientos por día
        reconocimientos_por_dia = db.query(
            func.date(HistorialReconocimiento.fecha_reconocimiento).label('fecha'),
            func.count(HistorialReconocimiento.id).label('total'),
            func.sum(func.case([(HistorialReconocimiento.reconocido == True, 1)], else_=0)).label('exitosos'),
            func.sum(func.case([(HistorialReconocimiento.alerta_generada == True, 1)], else_=0)).label('alertas')
        ).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio
        ).group_by(func.date(HistorialReconocimiento.fecha_reconocimiento)).all()

        # Convertir a diccionario
        daily_stats = {}
        for fecha, total, exitosos, alertas in reconocimientos_por_dia:
            daily_stats[fecha.isoformat()] = {
                "total": total,
                "exitosos": exitosos or 0,
                "alertas": alertas or 0,
                "tasa_exito": (exitosos / total * 100) if total > 0 else 0
            }

        # Distribución por confianza
        rangos_confianza = [
            (0, 50, "Baja"),
            (50, 70, "Media"),
            (70, 85, "Alta"),
            (85, 100, "Muy Alta")
        ]

        distribucion_confianza = {}
        for min_val, max_val, label in rangos_confianza:
            count = base_query.filter(
                and_(
                    HistorialReconocimiento.confianza >= min_val,
                    HistorialReconocimiento.confianza < max_val,
                    HistorialReconocimiento.reconocido == True
                )
            ).count()
            distribucion_confianza[label] = count

        # Top usuarios reconocidos
        top_usuarios = db.query(
            HistorialReconocimiento.usuario_id,
            func.count(HistorialReconocimiento.id).label('total_reconocimientos'),
            func.avg(HistorialReconocimiento.confianza).label('confianza_promedio')
        ).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio,
            HistorialReconocimiento.reconocido == True,
            HistorialReconocimiento.usuario_id.isnot(None)
        ).group_by(HistorialReconocimiento.usuario_id).order_by(
            func.count(HistorialReconocimiento.id).desc()
        ).limit(10).all()

        # Obtener información de usuarios
        top_usuarios_info = []
        for user_id, total_rec, avg_conf in top_usuarios:
            usuario = db.query(Usuario).filter(Usuario.id == user_id).first()
            if usuario:
                top_usuarios_info.append({
                    "usuario_id": user_id,
                    "nombre": f"{usuario.nombre} {usuario.apellido}",
                    "id_estudiante": usuario.id_estudiante,
                    "total_reconocimientos": total_rec,
                    "confianza_promedio": round(avg_conf, 2),
                    "requisitoriado": usuario.requisitoriado
                })

        estadisticas = {
            "periodo": {
                "dias": dias,
                "fecha_inicio": fecha_inicio.isoformat(),
                "fecha_fin": datetime.now().isoformat()
            },
            "resumen": {
                "total_reconocimientos": total_reconocimientos,
                "reconocimientos_exitosos": reconocimientos_exitosos,
                "alertas_generadas": alertas_generadas,
                "tasa_exito": round(tasa_exito, 2),
                "confianza_promedio": round(avg_confianza, 2),
                "promedio_diario": round(total_reconocimientos / dias, 2)
            },
            "por_dia": daily_stats,
            "distribucion_confianza": distribucion_confianza,
            "top_usuarios_reconocidos": top_usuarios_info
        }

        return ResponseWithData(
            success=True,
            message="Estadísticas obtenidas exitosamente",
            data=estadisticas
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


@router.post("/test-reconocimiento", response_model=ResponseWithData, summary="Probar reconocimiento")
async def test_reconocimiento(
        algoritmo: AlgoritmoReconocimiento = Form(AlgoritmoReconocimiento.HYBRID),
        db: Session = Depends(get_db)
):
    """
    Prueba el sistema de reconocimiento con imágenes existentes
    """
    try:
        # Verificar que el modelo esté cargado
        if not ml_service.load_models():
            raise HTTPException(status_code=503, detail="Modelo no está entrenado")

        # Obtener información del sistema
        system_info = ml_service.get_system_info()

        # Obtener algunas imágenes de ejemplo para prueba
        from models.database_models import ImagenFacial
        imagenes_ejemplo = db.query(ImagenFacial).filter(
            ImagenFacial.activa == True
        ).limit(5).all()

        resultados_prueba = []

        for imagen in imagenes_ejemplo:
            if os.path.exists(imagen.ruta_archivo):
                # Leer imagen
                img = cv2.imread(imagen.ruta_archivo)
                if img is not None:
                    # Realizar reconocimiento
                    resultado = ml_service.recognize_face(img, method=algoritmo.value)

                    # Obtener usuario real
                    usuario_real = db.query(Usuario).filter(Usuario.id == imagen.usuario_id).first()

                    resultados_prueba.append({
                        "imagen_id": imagen.id,
                        "usuario_real": imagen.usuario_id,
                        "nombre_real": f"{usuario_real.nombre} {usuario_real.apellido}" if usuario_real else "Desconocido",
                        "usuario_predicho": resultado.get("person_id"),
                        "confianza": resultado.get("confidence"),
                        "correcto": resultado.get("person_id") == imagen.usuario_id,
                        "reconocido": resultado.get("recognized")
                    })

        # Calcular métricas
        total_pruebas = len(resultados_prueba)
        aciertos = sum(1 for r in resultados_prueba if r["correcto"])
        precision = (aciertos / total_pruebas * 100) if total_pruebas > 0 else 0

        return ResponseWithData(
            success=True,
            message=f"Prueba completada con {aciertos}/{total_pruebas} aciertos",
            data={
                "algoritmo_usado": algoritmo.value,
                "sistema_info": system_info,
                "metricas": {
                    "total_pruebas": total_pruebas,
                    "aciertos": aciertos,
                    "precision": round(precision, 2)
                },
                "resultados_detallados": resultados_prueba
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la prueba: {str(e)}")


@router.get("/modelo/info", response_model=ResponseWithData, summary="Información del modelo")
async def info_modelo():
    """
    Obtiene información detallada del modelo de reconocimiento
    """
    try:
        # Cargar modelos si no están cargados
        ml_service.load_models()

        # Obtener información del sistema
        system_info = ml_service.get_system_info()

        return ResponseWithData(
            success=True,
            message="Información del modelo obtenida exitosamente",
            data=system_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener información: {str(e)}")


@router.post("/modelo/recargar", response_model=ResponseWithData, summary="Recargar modelo")
async def recargar_modelo():
    """
    Recarga los modelos de reconocimiento desde disco
    """
    try:
        # Recargar modelos
        success = ml_service.load_models()

        if success:
            system_info = ml_service.get_system_info()
            return ResponseWithData(
                success=True,
                message="Modelos recargados exitosamente",
                data=system_info
            )
        else:
            raise HTTPException(status_code=503, detail="No se pudieron cargar los modelos")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al recargar modelos: {str(e)}")


@router.get("/alertas/historial", response_model=ResponseWithData, summary="Historial de alertas")
async def historial_alertas(
        limite: int = Query(50, ge=1, le=200, description="Número máximo de alertas"),
        nivel: Optional[str] = Query(None, description="Filtrar por nivel (HIGH, MEDIUM, LOW)"),
):
    """
    Obtiene el historial de alertas de seguridad
    """
    try:
        alertas = alert_system.get_alert_history(limit=limite, alert_level=nivel)

        return ResponseWithData(
            success=True,
            message=f"Historial de alertas obtenido ({len(alertas)} alertas)",
            data={
                "total_alertas": len(alertas),
                "filtro_nivel": nivel,
                "alertas": alertas
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener alertas: {str(e)}")


@router.get("/alertas/estadisticas", response_model=ResponseWithData, summary="Estadísticas de alertas")
async def estadisticas_alertas():
    """
    Obtiene estadísticas de las alertas de seguridad
    """
    try:
        stats = alert_system.get_alert_statistics()

        return ResponseWithData(
            success=True,
            message="Estadísticas de alertas obtenidas exitosamente",
            data=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas de alertas: {str(e)}")


@router.post("/alertas/test", response_model=ResponseWithData, summary="Probar sistema de alertas")
async def test_sistema_alertas():
    """
    Prueba el sistema de alertas con datos de ejemplo
    """
    try:
        test_result = alert_system.test_alert_system()

        return ResponseWithData(
            success=True,
            message="Prueba del sistema de alertas completada",
            data=test_result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la prueba de alertas: {str(e)}")


@router.delete("/historial/limpiar", response_model=ResponseWithData, summary="Limpiar historial antiguo")
async def limpiar_historial(
        dias_mantener: int = Query(90, ge=7, le=365, description="Días de historial a mantener"),
        db: Session = Depends(get_db)
):
    """
    Limpia el historial de reconocimientos antiguo
    """
    try:
        from datetime import datetime, timedelta

        # Calcular fecha límite
        fecha_limite = datetime.now() - timedelta(days=dias_mantener)

        # Contar registros a eliminar
        registros_a_eliminar = db.query(HistorialReconocimiento).filter(
            HistorialReconocimiento.fecha_reconocimiento < fecha_limite
        ).count()

        # Eliminar registros antiguos
        db.query(HistorialReconocimiento).filter(
            HistorialReconocimiento.fecha_reconocimiento < fecha_limite
        ).delete()

        db.commit()

        # Limpiar también alertas antiguas
        alertas_eliminadas = alert_system.clear_old_alerts(days_to_keep=dias_mantener)

        return ResponseWithData(
            success=True,
            message=f"Limpieza completada",
            data={
                "reconocimientos_eliminados": registros_a_eliminar,
                "alertas_eliminadas": alertas_eliminadas,
                "dias_mantenidos": dias_mantener,
                "fecha_limite": fecha_limite.isoformat()
            }
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al limpiar historial: {str(e)}")