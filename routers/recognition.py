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
MAX_FILE_SIZE = 10 * 1024 * 1024
os.makedirs(TEMP_DIR, exist_ok=True)


def validate_image_file(file: UploadFile) -> bool:
    """Valida si el archivo es una imagen válida"""
    if not file.filename:
        return False

    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    return file_ext in allowed_extensions

def convert_numpy_types(obj):
    """
    Convierte tipos numpy a tipos nativos de Python para serialización JSON
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def save_recognition_to_history(self, recognition_result: Dict, image_path: str, client_ip: str, db: Session):
    """
    Guarda el reconocimiento en el historial
    """
    try:
        historial = HistorialReconocimiento(
            usuario_id=recognition_result.get("person_id"),
            imagen_consulta_path=image_path,
            confianza=int(recognition_result.get("confidence", 0)),
            distancia_euclidiana=str(recognition_result.get("details", {}).get("distance", "N/A")),
            reconocido=recognition_result.get("recognized", False),
            alerta_generada=recognition_result.get("alerta_seguridad") is not None,
            caracteristicas_consulta=self._clean_for_json(recognition_result.get("details", {})),
            ip_origen=client_ip
        )

        db.add(historial)
        db.commit()

        return historial.id

    except Exception as e:
        print(f"⚠️ Error guardando historial: {e}")
        return None


def _clean_for_json(self, data):
    """
    Limpia datos para JSON (convierte numpy types)
    """
    if isinstance(data, dict):
        return {k: self._clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [self._clean_for_json(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data


@router.post("/identificar", response_model=ResponseWithData, summary="Identificar persona en imagen")
async def identificar_persona(
        imagen: UploadFile = File(..., description="Imagen a analizar"),
        algoritmo: AlgoritmoReconocimiento = Form(AlgoritmoReconocimiento.HYBRID, description="Algoritmo a usar"),
        incluir_detalles: bool = Form(True, description="Incluir detalles técnicos"),
        request: Request = None,
        db: Session = Depends(get_db)
):
    """
    CORREGIDO: Identificación robusta con manejo de errores mejorado
    """
    temp_file_path = None

    try:
        print(f"🔍 Iniciando identificación con algoritmo: {algoritmo.value}")

        # PASO 1: Validaciones básicas
        if not validate_image_file(imagen):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser una imagen válida (jpg, jpeg, png, bmp)"
            )

        # Verificar tamaño de archivo
        if imagen.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"El archivo excede el tamaño máximo permitido ({MAX_FILE_SIZE / (1024 * 1024):.1f} MB)"
            )

        # PASO 2: Verificar modelo entrenado
        if not ml_service.load_models():
            raise HTTPException(
                status_code=503,
                detail="El modelo de reconocimiento no está entrenado. Entrene el modelo primero usando /api/v1/usuarios/entrenar-modelo"
            )

        # PASO 3: Guardar archivo temporal
        file_extension = os.path.splitext(imagen.filename)[1]
        temp_filename = f"recognition_{uuid.uuid4().hex}{file_extension}"
        temp_file_path = os.path.join(TEMP_DIR, temp_filename)

        # Guardar con validación
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await imagen.read()
                if len(content) == 0:
                    raise HTTPException(status_code=400, detail="El archivo está vacío")
                buffer.write(content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error guardando archivo: {str(e)}")

        # PASO 4: Leer y validar imagen
        img = cv2.imread(temp_file_path)
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="No se pudo leer la imagen. El archivo puede estar corrupto o en un formato no soportado."
            )

        # Validar dimensiones mínimas
        if img.shape[0] < 50 or img.shape[1] < 50:
            raise HTTPException(
                status_code=400,
                detail=f"La imagen es demasiado pequeña ({img.shape[1]}x{img.shape[0]}). Mínimo requerido: 50x50 píxeles."
            )

        print(f"✅ Imagen cargada correctamente: {img.shape}, tamaño: {imagen.size} bytes")

        # PASO 5: Realizar reconocimiento con manejo de errores
        start_time = datetime.now()

        try:
            recognition_result = ml_service.recognize_face(img, method=algoritmo.value)
            end_time = datetime.now()

            # Convertir tipos numpy a tipos nativos Python
            recognition_result = convert_numpy_types(recognition_result)
            processing_time = (end_time - start_time).total_seconds()

            print(f"✅ Reconocimiento completado en {processing_time:.3f}s")

        except Exception as e:
            print(f"❌ Error en reconocimiento facial: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error procesando la imagen con {algoritmo.value}: {str(e)}"
            )

        # PASO 6: Preparar respuesta base
        response_data = {
            "reconocido": recognition_result.get("recognized", False),
            "persona_id": recognition_result.get("person_id"),
            "confianza": round(recognition_result.get("confidence", 0.0), 2),
            "metodo": recognition_result.get("method", algoritmo.value),
            "tiempo_procesamiento": round(processing_time, 3),
            "timestamp": recognition_result.get("timestamp", datetime.now().isoformat()),
            "imagen_info": {
                "dimensiones": f"{img.shape[1]}x{img.shape[0]}",
                "canales": img.shape[2] if len(img.shape) == 3 else 1,
                "tamano_bytes": imagen.size
            }
        }

        # Incluir detalles técnicos si se solicita
        if incluir_detalles:
            response_data["detalles_tecnicos"] = convert_numpy_types(recognition_result.get("details", {}))

            # Incluir errores si los hay
            if "errors" in recognition_result:
                response_data["errores_algoritmos"] = recognition_result["errors"]

        # PASO 7: Obtener información de la persona y manejar alertas
        persona_info = None
        alerta_seguridad = None

        if response_data["reconocido"] and response_data["persona_id"]:
            try:
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

                    # PASO 8: Generar alerta si la persona está requisitoriada
                    if usuario.requisitoriado:
                        try:
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
                                    "client_ip": request.client.host if request else "unknown"
                                }
                            )

                            alerta_seguridad = alert_system.generate_security_alert(alert_info)
                            print(f"🚨 Alerta de seguridad generada para persona requisitoriada")

                        except Exception as e:
                            print(f"⚠️ Error generando alerta de seguridad: {e}")
                            # No fallar el reconocimiento por error en alertas

            except Exception as e:
                print(f"⚠️ Error obteniendo información del usuario: {e}")
                # No fallar el reconocimiento por error en BD

        # PASO 9: Registrar en historial
        try:
            client_ip = request.client.host if request else "unknown"

            historial = HistorialReconocimiento(
                usuario_id=response_data["persona_id"],
                imagen_consulta_path=temp_file_path,
                confianza=int(response_data["confianza"]),
                distancia_euclidiana=str(recognition_result.get("details", {}).get("distance", "N/A")),
                reconocido=response_data["reconocido"],
                alerta_generada=alerta_seguridad is not None,
                caracteristicas_consulta=convert_numpy_types(recognition_result.get("details", {})),
                ip_origen=client_ip
            )

            db.add(historial)
            db.commit()
            response_data["historial_id"] = historial.id
            print(f"💾 Historial guardado: ID={historial.id}")

        except Exception as e:
            print(f"⚠️ Error guardando historial: {e}")
            # No fallar el reconocimiento por error en historial

        # PASO 10: Añadir información adicional a la respuesta
        response_data["persona_info"] = persona_info
        response_data["alerta_seguridad"] = alerta_seguridad

        # PASO 11: Determinar mensaje de respuesta
        if response_data["reconocido"]:
            if alerta_seguridad:
                mensaje = f"🚨 PERSONA REQUISITORIADA IDENTIFICADA: {persona_info['nombre']} {persona_info['apellido']} - ALERTA GENERADA"
            else:
                mensaje = f"✅ Persona identificada: {persona_info['nombre']} {persona_info['apellido']} (Confianza: {response_data['confianza']}%)"
        else:
            mensaje = f"❌ No se pudo identificar a la persona en la imagen (Confianza insuficiente: {response_data['confianza']}%)"

        return ResponseWithData(
            success=True,
            message=mensaje,
            data=response_data
        )

    except HTTPException:
        # Re-lanzar HTTPExceptions tal como están
        raise
    except Exception as e:
        # Capturar cualquier otro error no manejado
        print(f"❌ Error inesperado en identificación: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

    finally:
        # OPCIONAL: Limpiar archivo temporal
        # (Comentado para mantener archivos para historial)
        # if temp_file_path and os.path.exists(temp_file_path):
        #     try:
        #         os.remove(temp_file_path)
        #     except Exception as e:
        #         print(f"⚠️ Error eliminando archivo temporal: {e}")
        pass


@router.post("/debug/test-image", response_model=ResponseWithData, summary="Probar procesamiento de imagen")
async def test_image_processing(
        imagen: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    """
    Endpoint para debugging del procesamiento de imágenes
    """
    temp_file_path = None

    try:
        # Validar archivo
        if not validate_image_file(imagen):
            raise HTTPException(status_code=400, detail="Archivo de imagen inválido")

        # Guardar imagen temporal
        file_extension = os.path.splitext(imagen.filename)[1]
        temp_filename = f"debug_{uuid.uuid4().hex}{file_extension}"
        temp_file_path = os.path.join(TEMP_DIR, temp_filename)

        with open(temp_file_path, "wb") as buffer:
            content = await imagen.read()
            buffer.write(content)

        # Ejecutar validación completa
        from utils.debug_helper import DebugHelper
        debug_results = DebugHelper.validate_image_pipeline(temp_file_path)

        return ResponseWithData(
            success=True,
            message="Debugging de imagen completado",
            data=debug_results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en debugging: {str(e)}")

    finally:
        # Limpiar archivo temporal
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@router.get("/debug/test-all-images", response_model=ResponseWithData, summary="Probar todas las imágenes")
async def test_all_images():
    """
    Prueba el procesamiento de todas las imágenes en la BD
    """
    try:
        from utils.debug_helper import DebugHelper
        results = DebugHelper.test_all_user_images()

        return ResponseWithData(
            success=True,
            message=f"Testing completado: {results['successful']}/{results['total_images']} exitosas",
            data=results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en testing: {str(e)}")


@router.get("/debug/model-test", response_model=ResponseWithData, summary="Probar reconocimiento del modelo")
async def test_model_recognition():
    """
    Prueba el reconocimiento con imágenes existentes en la BD
    """
    try:
        from utils.debug_helper import DebugHelper
        results = DebugHelper.test_model_recognition()

        if "error" in results:
            return ResponseWithData(
                success=False,
                message=results["error"],
                data=results
            )

        return ResponseWithData(
            success=True,
            message=f"Testing de reconocimiento completado: {results['successful_recognitions']}/{results['total_tests']} exitosos",
            data=results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en testing de modelo: {str(e)}")


@router.post("/debug/export-report", response_model=ResponseWithData, summary="Exportar reporte de debugging")
async def export_debug_report():
    """
    Exporta un reporte completo de debugging del sistema
    """
    try:
        from utils.debug_helper import DebugHelper
        report_path = DebugHelper.export_debug_report()

        return ResponseWithData(
            success=True,
            message="Reporte de debugging exportado exitosamente",
            data={
                "report_path": report_path,
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando reporte: {str(e)}")


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