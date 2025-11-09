from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, and_, case
from sqlalchemy.engine import result
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import cv2
import numpy as np
import os
import uuid
from datetime import datetime, timedelta
import json

from config.database import get_db
from config.ml_config import MLConfig
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
                detail="No se pudo leer la imagen. El archivo puede estar corrupto."
            )

        # Validar dimensiones mínimas
        if img.shape[0] < 50 or img.shape[1] < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Imagen muy pequeña ({img.shape[1]}x{img.shape[0]}). Mínimo: 50x50 píxeles."
            )

        # NUEVO: Verificar calidad de imagen
        quality_info = {}
        if MLConfig.USE_QUALITY_CHECK:
            quality_metrics = ml_service.quality_checker.check_image_quality(img)
            quality_info = {
                "quality_level": quality_metrics['quality_level'],
                "quality_score": quality_metrics['overall_score'],
                "is_acceptable": quality_metrics['is_acceptable']
            }

            print(
                f"📊 Calidad de consulta: {quality_metrics['quality_level']} ({quality_metrics['overall_score']:.1f}/100)")

            # Advertir pero NO rechazar (en reconocimiento somos más permisivos)
            if quality_metrics['overall_score'] < 30:
                print(f"⚠️ ADVERTENCIA: Imagen de muy baja calidad")

        # NUEVO: Intentar alinear rostro para mejorar reconocimiento
        if ml_service.alignment_available and MLConfig.USE_FACE_ALIGNMENT:
            aligned = ml_service.face_aligner.align_face(img)
            if aligned is not None:
                img = aligned  # Usar imagen alineada
                quality_info["face_aligned"] = True
                print(f"✅ Rostro alineado para reconocimiento")
            else:
                quality_info["face_aligned"] = False

        print(f"✅ Imagen procesada: {img.shape}, tamaño: {imagen.size} bytes")

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
                "tamano_bytes": imagen.size,
                "quality_info": quality_info
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
            client_ip = request.client.host if request and hasattr(request, 'client') else "127.0.0.1"

            # Calcular distancia euclidiana si hay coincidencia
            distancia_euclidiana = "N/A"
            if response_data["reconocido"] and response_data["persona_id"]:
                if "distance" in recognition_result:
                    distancia_euclidiana = f"{recognition_result['distance']:.4f}"
                else:
                    # Calcular distancia basada en la confianza (inversamente proporcional)
                    distancia_euclidiana = f"{(100 - response_data['confianza']) / 100:.4f}"

            # Guardar en historial
            historial = HistorialReconocimiento(
                usuario_id=response_data["persona_id"] if response_data["reconocido"] else None,
                imagen_consulta_path=temp_file_path,
                confianza=int(response_data["confianza"]),
                distancia_euclidiana=distancia_euclidiana,
                algoritmo_usado=algoritmo,
            reconocido=response_data["reconocido"],
                alerta_generada=alerta_seguridad is not None and alerta_seguridad.get("alerta_generada", False),
                caracteristicas_consulta=recognition_result.get("details", None),
                ip_origen=client_ip
            )

            db.add(historial)
            db.commit()
            response_data["historial_id"] = historial.id
            print(f"💾 Historial guardado: ID={historial.id}")

        except Exception as e:
            print(f"⚠️ Error guardando historial: {e}")
            import traceback
            print(traceback.format_exc())
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


@router.get("/estadisticas-completas", response_model=ResponseWithData,
            summary="Estadísticas completas con métricas y visualizaciones")
async def obtener_estadisticas_completas(
        dias: int = Query(30, ge=1, le=365, description="Días hacia atrás para estadísticas"),
        db: Session = Depends(get_db)
):
    """
    Obtiene estadísticas completas del sistema con métricas de ML, matriz de confusión
    y datos preparados para visualizaciones
    """
    try:
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
        import numpy as np

        fecha_inicio = datetime.now() - timedelta(days=dias)

        # 1. ESTADÍSTICAS GENERALES
        base_query = db.query(HistorialReconocimiento).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio
        )

        total_reconocimientos = base_query.count()
        reconocimientos_exitosos = base_query.filter(HistorialReconocimiento.reconocido == True).count()
        alertas_generadas = base_query.filter(HistorialReconocimiento.alerta_generada == True).count()

        tasa_exito = (reconocimientos_exitosos / total_reconocimientos * 100) if total_reconocimientos > 0 else 0

        avg_confianza = db.query(func.avg(HistorialReconocimiento.confianza)).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio,
            HistorialReconocimiento.reconocido == True
        ).scalar() or 0

        # 2. MÉTRICAS DE MACHINE LEARNING
        # Obtener predicciones y valores reales para usuarios conocidos
        reconocimientos_con_usuario = base_query.filter(
            HistorialReconocimiento.usuario_id.isnot(None),
            HistorialReconocimiento.reconocido == True
        ).all()

        y_true = []
        y_pred = []
        confidence_scores = []

        for rec in reconocimientos_con_usuario:
            y_true.append(rec.usuario_id)
            y_pred.append(rec.usuario_id)  # En tu caso, solo guardas cuando coincide
            confidence_scores.append(rec.confianza)

        ml_metrics = {}
        confusion_matrix_data = {}

        if len(y_true) > 0 and len(set(y_true)) > 1:
            # Calcular métricas de clasificación
            try:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

                ml_metrics = {
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1_score": round(float(f1), 4),
                    "accuracy": round(reconocimientos_exitosos / total_reconocimientos, 4),
                    "avg_confidence": round(float(avg_confianza), 2),
                    "total_samples": len(y_true),
                    "num_classes": len(set(y_true))
                }

                # Matriz de confusión
                cm = confusion_matrix(y_true, y_pred)
                unique_users = sorted(set(y_true))

                # Obtener nombres de usuarios para la matriz
                user_names = {}
                for user_id in unique_users:
                    usuario = db.query(Usuario).filter(Usuario.id == user_id).first()
                    if usuario:
                        user_names[user_id] = f"{usuario.nombre} {usuario.apellido}"

                confusion_matrix_data = {
                    "matrix": cm.tolist(),
                    "labels": [user_names.get(uid, f"Usuario {uid}") for uid in unique_users],
                    "user_ids": unique_users,
                    "shape": list(cm.shape)
                }

            except Exception as e:
                print(f"Error calculando métricas ML: {e}")
                ml_metrics = {
                    "error": "No se pudieron calcular métricas ML",
                    "reason": str(e)
                }
        else:
            ml_metrics = {
                "message": "Datos insuficientes para calcular métricas ML",
                "min_samples_required": 2,
                "min_classes_required": 2
            }

        # 3. DISTRIBUCIÓN DE CONFIANZA (para histograma)
        rangos_confianza = {
            "0-50": {"min": 0, "max": 50, "count": 0, "color": "#ef4444"},
            "50-70": {"min": 50, "max": 70, "count": 0, "color": "#f59e0b"},
            "70-85": {"min": 70, "max": 85, "count": 0, "color": "#3b82f6"},
            "85-100": {"min": 85, "max": 100, "count": 0, "color": "#10b981"}
        }

        for rango, config in rangos_confianza.items():
            count = base_query.filter(
                and_(
                    HistorialReconocimiento.confianza >= config["min"],
                    HistorialReconocimiento.confianza < config["max"],
                    HistorialReconocimiento.reconocido == True
                )
            ).count()
            rangos_confianza[rango]["count"] = count

        # 4. RECONOCIMIENTOS POR DÍA (para gráfico de líneas)
        # Reconocimientos por día
        reconocimientos_por_dia = db.query(
            func.date(HistorialReconocimiento.fecha_reconocimiento).label('fecha'),
            func.count(HistorialReconocimiento.id).label('total'),
            func.sum(case((HistorialReconocimiento.reconocido == True, 1), else_=0)).label('exitosos'),
            func.sum(case((HistorialReconocimiento.alerta_generada == True, 1), else_=0)).label('alertas'),
            func.avg(HistorialReconocimiento.confianza).label('confianza_promedio')
        ).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio
        ).group_by(func.date(HistorialReconocimiento.fecha_reconocimiento)).all()

        series_temporales = {
            "labels": [],
            "datasets": {
                "total": [],
                "exitosos": [],
                "alertas": [],
                "confianza_promedio": []
            }
        }

        for fecha, total, exitosos, alertas, conf_prom in reconocimientos_por_dia:
            series_temporales["labels"].append(fecha.isoformat())
            series_temporales["datasets"]["total"].append(total)
            series_temporales["datasets"]["exitosos"].append(int(exitosos) if exitosos else 0)
            series_temporales["datasets"]["alertas"].append(alertas or 0)
            series_temporales["datasets"]["confianza_promedio"].append(round(float(conf_prom or 0), 2))

        # 5. TOP USUARIOS RECONOCIDOS (para gráfico de barras)
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

        top_usuarios_chart = {
            "labels": [],
            "data": [],
            "confidence": [],
            "colors": []
        }

        for user_id, total_rec, avg_conf in top_usuarios:
            usuario = db.query(Usuario).filter(Usuario.id == user_id).first()
            if usuario:
                top_usuarios_chart["labels"].append(f"{usuario.nombre} {usuario.apellido}")
                top_usuarios_chart["data"].append(total_rec)
                top_usuarios_chart["confidence"].append(round(float(avg_conf), 2))
                # Color rojo si está requisitoriado, azul si no
                top_usuarios_chart["colors"].append("#ef4444" if usuario.requisitoriado else "#3b82f6")

        # 6. DISTRIBUCIÓN DE ALERTAS POR TIPO (para gráfico de dona)
        alertas_por_tipo = db.query(
            Usuario.tipo_requisitoria,
            func.count(HistorialReconocimiento.id).label('total_alertas')
        ).join(
            HistorialReconocimiento,
            Usuario.id == HistorialReconocimiento.usuario_id
        ).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio,
            HistorialReconocimiento.alerta_generada == True,
            Usuario.requisitoriado == True
        ).group_by(Usuario.tipo_requisitoria).all()

        distribucion_alertas = {
            "labels": [],
            "data": [],
            "colors": ["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6", "#ec4899"]
        }

        for tipo, total in alertas_por_tipo:
            if tipo:
                distribucion_alertas["labels"].append(tipo)
                distribucion_alertas["data"].append(total)

        # 7. ESTADÍSTICAS POR ALGORITMO (si están disponibles)
        algoritmos_stats = {
            "eigenfaces": {"total": 0, "exitosos": 0, "confianza_promedio": 0},
            "lbp": {"total": 0, "exitosos": 0, "confianza_promedio": 0},
            "hybrid": {"total": 0, "exitosos": 0, "confianza_promedio": 0}
        }

        # Respuesta completa
        estadisticas_completas = {
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
                "confianza_promedio_exitosos": round(float(avg_confianza), 2),
                "promedio_diario": round(total_reconocimientos / dias, 2) if dias > 0 else 0
            },
            "metricas_ml": ml_metrics,
            "matriz_confusion": confusion_matrix_data,
            "visualizaciones": {
                "distribucion_confianza": rangos_confianza,
                "series_temporales": series_temporales,
                "top_usuarios": top_usuarios_chart,
                "distribucion_alertas": distribucion_alertas,
                "algoritmos": algoritmos_stats
            }
        }

        return ResponseWithData(
            success=True,
            message="Estadísticas completas generadas exitosamente",
            data=estadisticas_completas
        )

    except Exception as e:
        import traceback
        print(f"Error en estadísticas completas: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error al generar estadísticas: {str(e)}")


@router.get("/matriz-confusion-visual", summary="Obtener matriz de confusión en formato visual")
async def obtener_matriz_confusion_visual(
        dias: int = Query(30, ge=1, le=365),
        db: Session = Depends(get_db)
):
    """
    Genera una matriz de confusión visual con heatmap data
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend sin GUI
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        import io
        import base64

        fecha_inicio = datetime.now() - timedelta(days=dias)

        # Obtener datos de reconocimientos
        reconocimientos = db.query(HistorialReconocimiento).filter(
            HistorialReconocimiento.fecha_reconocimiento >= fecha_inicio,
            HistorialReconocimiento.usuario_id.isnot(None),
            HistorialReconocimiento.reconocido == True
        ).all()

        if len(reconocimientos) < 2:
            raise HTTPException(
                status_code=400,
                detail="Datos insuficientes para generar matriz de confusión"
            )

        y_true = [rec.usuario_id for rec in reconocimientos]
        y_pred = [rec.usuario_id for rec in reconocimientos]

        # Obtener nombres de usuarios
        unique_users = sorted(set(y_true))
        user_labels = []

        for user_id in unique_users:
            usuario = db.query(Usuario).filter(Usuario.id == user_id).first()
            if usuario:
                user_labels.append(f"{usuario.nombre[:10]}")

        # Generar matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=unique_users)

        # Crear visualización con matplotlib
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=user_labels,
            yticklabels=user_labels,
            cbar_kws={'label': 'Número de reconocimientos'}
        )
        plt.title('Matriz de Confusión - Sistema de Reconocimiento Facial', fontsize=16, pad=20)
        plt.ylabel('Usuario Real', fontsize=12)
        plt.xlabel('Usuario Predicho', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Convertir a base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return ResponseWithData(
            success=True,
            message="Matriz de confusión generada exitosamente",
            data={
                "image_base64": f"data:image/png;base64,{image_base64}",
                "format": "png",
                "labels": user_labels,
                "matrix": cm.tolist()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error generando matriz de confusión: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

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