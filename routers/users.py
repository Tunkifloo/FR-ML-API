from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Tuple
import shutil
import os
from datetime import datetime
import uuid

from config.database import get_db
from models.database_models import Usuario, ImagenFacial, CaracteristicasFaciales, asignar_requisitoriado_aleatorio
from models.pydantic_models import (
    UsuarioCreate, UsuarioUpdate, Usuario as UsuarioResponse, UsuarioDetallado,
    ResponseWithData, ResponsePaginado, ErrorResponse, FiltroUsuarios, Paginacion
)
from services.ml_service import MLService
from utils.alert_system import AlertSystem

router = APIRouter(prefix="/usuarios", tags=["Usuarios"])

# Inicializar servicios
ml_service = MLService()
alert_system = AlertSystem()

# Configuración de archivos
UPLOAD_DIR = "storage/images"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_DIR, exist_ok=True)


def validate_image_file(file: UploadFile) -> bool:
    """Valida si el archivo es una imagen válida"""
    if not file.filename:
        return False

    file_ext = os.path.splitext(file.filename)[1].lower()
    return file_ext in ALLOWED_EXTENSIONS


def convert_image_to_base64(image_path: str, max_size: Tuple[int, int] = (150, 150)) -> Optional[str]:
    """
    Convierte una imagen a base64 para preview, redimensionándola si es necesario

    Args:
        image_path: Ruta de la imagen
        max_size: Tamaño máximo para el preview (ancho, alto)

    Returns:
        String base64 de la imagen o None si hay error
    """
    try:
        import base64
        from PIL import Image
        import io

        if not os.path.exists(image_path):
            print(f"⚠️ Imagen no encontrada: {image_path}")
            return None

        # Abrir imagen
        with Image.open(image_path) as img:
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Redimensionar manteniendo proporción
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convertir a bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=85, optimize=True)
            img_bytes = img_buffer.getvalue()

            # Convertir a base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # Retornar con prefijo para uso directo en HTML/frontend
            return f"data:image/jpeg;base64,{img_base64}"

    except Exception as e:
        print(f"⚠️ Error convirtiendo imagen a base64: {e}")
        return None


@router.post("/", response_model=ResponseWithData, summary="Crear nuevo usuario")
async def crear_usuario(
        nombre: str = Form(...),
        apellido: str = Form(...),
        email: str = Form(...),
        id_estudiante: Optional[str] = Form(None),
        imagenes: List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    """
    Crea un nuevo usuario con sus imágenes faciales (mínimo 1, máximo 5)
    CON ENTRENAMIENTO AUTOMÁTICO INTELIGENTE
    """
    try:
        # Validar número de imágenes
        if len(imagenes) < 1 or len(imagenes) > 5:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar entre 1 y 5 imágenes"
            )

        # Validar archivos de imagen
        for img in imagenes:
            if not validate_image_file(img):
                raise HTTPException(
                    status_code=400,
                    detail=f"Archivo '{img.filename}' no es una imagen válida"
                )

            if img.size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Archivo '{img.filename}' excede el tamaño máximo (10MB)"
                )

        # Verificar si el email ya existe
        existing_user = db.query(Usuario).filter(Usuario.email == email).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="El email ya está registrado"
            )

        # Verificar si el ID de estudiante ya existe (si se proporciona)
        if id_estudiante:
            existing_student = db.query(Usuario).filter(Usuario.id_estudiante == id_estudiante).first()
            if existing_student:
                raise HTTPException(
                    status_code=400,
                    detail="El ID de estudiante ya está registrado"
                )

        # Asignar estado de requisitoriado aleatoriamente
        es_requisitoriado, tipo_requisitoria = asignar_requisitoriado_aleatorio()

        # Crear usuario
        nuevo_usuario = Usuario(
            nombre=nombre.title(),
            apellido=apellido.title(),
            email=email.lower(),
            id_estudiante=id_estudiante,
            requisitoriado=es_requisitoriado,
            tipo_requisitoria=tipo_requisitoria
        )

        db.add(nuevo_usuario)
        db.commit()
        db.refresh(nuevo_usuario)

        # Procesar y guardar imágenes
        imagenes_guardadas = []
        imagenes_procesadas = []

        for i, imagen in enumerate(imagenes):
            # Generar nombre único para el archivo
            file_extension = os.path.splitext(imagen.filename)[1]
            unique_filename = f"user_{nuevo_usuario.id}_{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)

            # Guardar archivo
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(imagen.file, buffer)

            # Obtener información del archivo
            file_stat = os.stat(file_path)

            # Crear registro de imagen
            imagen_facial = ImagenFacial(
                usuario_id=nuevo_usuario.id,
                nombre_archivo=imagen.filename,
                ruta_archivo=file_path,
                es_principal=(i == 0),  # Primera imagen como principal
                formato=file_extension[1:],  # Sin el punto
                tamano_bytes=file_stat.st_size
            )

            db.add(imagen_facial)
            imagenes_guardadas.append(imagen_facial)

        db.commit()

        # ENTRENAMIENTO AUTOMÁTICO INTELIGENTE
        try:
            import cv2
            import numpy as np

            for imagen_facial in imagenes_guardadas:
                # Leer imagen
                img = cv2.imread(imagen_facial.ruta_archivo)
                if img is not None:
                    imagenes_procesadas.append(img)

                    # Actualizar dimensiones de la imagen
                    imagen_facial.alto = img.shape[0]
                    imagen_facial.ancho = img.shape[1]

            # CÓDIGO CORREGIDO - Entrenamiento automático inteligente
            if imagenes_procesadas:
                training_result = ml_service.add_new_person(nuevo_usuario.id, imagenes_procesadas)

                # Preparar respuesta del usuario
                usuario_creado = {
                    "id": nuevo_usuario.id,
                    "nombre": nuevo_usuario.nombre,
                    "apellido": nuevo_usuario.apellido,
                    "email": nuevo_usuario.email,
                    "id_estudiante": nuevo_usuario.id_estudiante,
                    "requisitoriado": nuevo_usuario.requisitoriado,
                    "tipo_requisitoria": nuevo_usuario.tipo_requisitoria,
                    "total_imagenes": len(imagenes_guardadas),
                    "fecha_registro": nuevo_usuario.fecha_registro.isoformat(),
                    # Información del entrenamiento ML
                    "ml_training_status": training_result.get("status", "unknown"),
                    "ml_message": training_result.get("message", ""),
                    "model_trained": training_result.get("status") in ["added_incremental", "auto_trained"]
                }

                print(f"🤖 Resultado ML: {training_result.get('message', 'Sin mensaje')}")

            else:
                # Si no se procesaron imágenes
                usuario_creado = {
                    "id": nuevo_usuario.id,
                    "nombre": nuevo_usuario.nombre,
                    "apellido": nuevo_usuario.apellido,
                    "email": nuevo_usuario.email,
                    "id_estudiante": nuevo_usuario.id_estudiante,
                    "requisitoriado": nuevo_usuario.requisitoriado,
                    "tipo_requisitoria": nuevo_usuario.tipo_requisitoria,
                    "total_imagenes": len(imagenes_guardadas),
                    "fecha_registro": nuevo_usuario.fecha_registro.isoformat(),
                    "ml_training_status": "no_processed",
                    "ml_message": "No se procesaron imágenes para ML",
                    "model_trained": False
                }

        except Exception as e:
            print(f"⚠️ Error en procesamiento ML: {e}")
            # No fallar la creación del usuario por errores de ML
            usuario_creado = {
                "id": nuevo_usuario.id,
                "nombre": nuevo_usuario.nombre,
                "apellido": nuevo_usuario.apellido,
                "email": nuevo_usuario.email,
                "id_estudiante": nuevo_usuario.id_estudiante,
                "requisitoriado": nuevo_usuario.requisitoriado,
                "tipo_requisitoria": nuevo_usuario.tipo_requisitoria,
                "total_imagenes": len(imagenes_guardadas),
                "fecha_registro": nuevo_usuario.fecha_registro.isoformat(),
                "ml_training_status": "error",
                "ml_message": f"Error en entrenamiento: {str(e)}",
                "model_trained": False
            }

        db.commit()

        return ResponseWithData(
            success=True,
            message=f"Usuario creado exitosamente con {len(imagenes_guardadas)} imágenes",
            data=usuario_creado
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


@router.get("/", response_model=ResponsePaginado, summary="Listar usuarios con imágenes principales")
async def listar_usuarios(
        pagina: int = Query(1, ge=1, description="Número de página"),
        items_por_pagina: int = Query(10, ge=1, le=100, description="Items por página"),
        nombre: Optional[str] = Query(None, description="Filtrar por nombre"),
        apellido: Optional[str] = Query(None, description="Filtrar por apellido"),
        email: Optional[str] = Query(None, description="Filtrar por email"),
        requisitoriado: Optional[bool] = Query(None, description="Filtrar por estado de requisitoriado"),
        activo: Optional[bool] = Query(True, description="Filtrar por estado activo"),
        incluir_imagen: bool = Query(True, description="Incluir imagen principal para preview"),
        db: Session = Depends(get_db)
):
    """
    Lista usuarios con paginación, filtros e imágenes principales para preview
    """
    try:
        # Construir query base
        query = db.query(Usuario)

        # Aplicar filtros
        if nombre:
            query = query.filter(Usuario.nombre.ilike(f"%{nombre}%"))
        if apellido:
            query = query.filter(Usuario.apellido.ilike(f"%{apellido}%"))
        if email:
            query = query.filter(Usuario.email.ilike(f"%{email}%"))
        if requisitoriado is not None:
            query = query.filter(Usuario.requisitoriado == requisitoriado)
        if activo is not None:
            query = query.filter(Usuario.activo == activo)

        # Contar total de registros
        total = query.count()

        # Aplicar paginación
        offset = (pagina - 1) * items_por_pagina
        usuarios = query.offset(offset).limit(items_por_pagina).all()

        # Calcular total de páginas
        total_paginas = (total + items_por_pagina - 1) // items_por_pagina

        # Convertir a diccionarios con información extendida
        usuarios_data = []
        for usuario in usuarios:
            usuario_dict = {
                "id": usuario.id,
                "nombre": usuario.nombre,
                "apellido": usuario.apellido,
                "email": usuario.email,
                "id_estudiante": usuario.id_estudiante,
                "requisitoriado": usuario.requisitoriado,
                "tipo_requisitoria": usuario.tipo_requisitoria,
                "fecha_registro": usuario.fecha_registro.isoformat(),
                "activo": usuario.activo
            }

            # Añadir información de imagen principal si se solicita
            if incluir_imagen:
                # Buscar imagen principal
                imagen_principal = db.query(ImagenFacial).filter(
                    ImagenFacial.usuario_id == usuario.id,
                    ImagenFacial.activa == True,
                    ImagenFacial.es_principal == True
                ).first()

                # Si no hay imagen principal, tomar la primera imagen disponible
                if not imagen_principal:
                    imagen_principal = db.query(ImagenFacial).filter(
                        ImagenFacial.usuario_id == usuario.id,
                        ImagenFacial.activa == True
                    ).order_by(ImagenFacial.fecha_subida.asc()).first()

                if imagen_principal:
                    # Convertir imagen a base64 para preview
                    imagen_base64 = convert_image_to_base64(imagen_principal.ruta_archivo)

                    usuario_dict["imagen_principal"] = {
                        "id": imagen_principal.id,
                        "nombre_archivo": imagen_principal.nombre_archivo,
                        "formato": imagen_principal.formato,
                        "fecha_subida": imagen_principal.fecha_subida.isoformat(),
                        "es_principal": imagen_principal.es_principal,
                        "imagen_base64": imagen_base64,  # Para preview en frontend
                        "imagen_url": f"/images/{os.path.basename(imagen_principal.ruta_archivo)}"  # URL alternativa
                    }
                else:
                    usuario_dict["imagen_principal"] = None

                # Contar total de imágenes
                total_imagenes = db.query(ImagenFacial).filter(
                    ImagenFacial.usuario_id == usuario.id,
                    ImagenFacial.activa == True
                ).count()

                usuario_dict["total_imagenes"] = total_imagenes

            usuarios_data.append(usuario_dict)

        return ResponsePaginado(
            success=True,
            message=f"Usuarios obtenidos exitosamente",
            data=usuarios_data,
            total=total,
            pagina=pagina,
            items_por_pagina=items_por_pagina,
            total_paginas=total_paginas
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener usuarios: {str(e)}")


@router.get("/estudiante/{id_estudiante}", response_model=ResponseWithData,
            summary="Obtener usuario por ID de estudiante")
async def obtener_usuario_por_id_estudiante(
        id_estudiante: str,
        incluir_imagenes: bool = Query(True, description="Incluir todas las imágenes del usuario"),
        incluir_reconocimientos: bool = Query(False, description="Incluir historial de reconocimientos"),
        incluir_caracteristicas: bool = Query(False, description="Incluir características ML"),
        db: Session = Depends(get_db)
):
    """
    Obtiene un usuario específico por su ID de estudiante con información completa
    """
    try:
        # Buscar usuario por ID de estudiante
        usuario = db.query(Usuario).filter(
            Usuario.id_estudiante == id_estudiante,
            Usuario.activo == True
        ).first()

        if not usuario:
            raise HTTPException(
                status_code=404,
                detail=f"Usuario con ID de estudiante '{id_estudiante}' no encontrado"
            )

        # Datos básicos del usuario
        usuario_data = {
            "id": usuario.id,
            "nombre": usuario.nombre,
            "apellido": usuario.apellido,
            "email": usuario.email,
            "id_estudiante": usuario.id_estudiante,
            "requisitoriado": usuario.requisitoriado,
            "tipo_requisitoria": usuario.tipo_requisitoria,
            "fecha_registro": usuario.fecha_registro.isoformat(),
            "fecha_actualizacion": usuario.fecha_actualizacion.isoformat(),
            "activo": usuario.activo
        }

        # Incluir imágenes completas si se solicita
        if incluir_imagenes:
            imagenes = db.query(ImagenFacial).filter(
                ImagenFacial.usuario_id == usuario.id,
                ImagenFacial.activa == True
            ).order_by(ImagenFacial.es_principal.desc(), ImagenFacial.fecha_subida.asc()).all()

            imagenes_data = []
            for img in imagenes:
                # Convertir cada imagen a base64
                imagen_base64 = convert_image_to_base64(img.ruta_archivo)

                imagen_info = {
                    "id": img.id,
                    "nombre_archivo": img.nombre_archivo,
                    "es_principal": img.es_principal,
                    "formato": img.formato,
                    "tamano_bytes": img.tamano_bytes,
                    "ancho": img.ancho,
                    "alto": img.alto,
                    "fecha_subida": img.fecha_subida.isoformat(),
                    "imagen_base64": imagen_base64,  # Para mostrar en frontend
                    "imagen_url": f"/images/{os.path.basename(img.ruta_archivo)}"
                }
                imagenes_data.append(imagen_info)

            usuario_data["imagenes"] = imagenes_data
            usuario_data["total_imagenes"] = len(imagenes_data)

            # Identificar imagen principal específicamente
            imagen_principal = next((img for img in imagenes_data if img["es_principal"]),
                                    imagenes_data[0] if imagenes_data else None)
            usuario_data["imagen_principal"] = imagen_principal

        # Incluir reconocimientos si se solicita
        if incluir_reconocimientos:
            from models.database_models import HistorialReconocimiento
            reconocimientos = db.query(HistorialReconocimiento).filter(
                HistorialReconocimiento.usuario_id == usuario.id
            ).order_by(HistorialReconocimiento.fecha_reconocimiento.desc()).limit(20).all()

            reconocimientos_data = []
            for rec in reconocimientos:
                reconocimientos_data.append({
                    "id": rec.id,
                    "confianza": rec.confianza,
                    "reconocido": rec.reconocido,
                    "alerta_generada": rec.alerta_generada,
                    "fecha": rec.fecha_reconocimiento.isoformat(),
                    "ip_origen": rec.ip_origen,
                    "distancia_euclidiana": rec.distancia_euclidiana
                })

            usuario_data["reconocimientos_recientes"] = reconocimientos_data
            usuario_data["total_reconocimientos"] = len(reconocimientos_data)

            # Estadísticas de reconocimientos
            total_reconocimientos = db.query(HistorialReconocimiento).filter(
                HistorialReconocimiento.usuario_id == usuario.id
            ).count()

            reconocimientos_exitosos = db.query(HistorialReconocimiento).filter(
                HistorialReconocimiento.usuario_id == usuario.id,
                HistorialReconocimiento.reconocido == True
            ).count()

            usuario_data["estadisticas_reconocimiento"] = {
                "total_reconocimientos": total_reconocimientos,
                "reconocimientos_exitosos": reconocimientos_exitosos,
                "tasa_exito": (
                            reconocimientos_exitosos / total_reconocimientos * 100) if total_reconocimientos > 0 else 0
            }

        # Incluir características ML si se solicita
        if incluir_caracteristicas:
            caracteristicas = db.query(CaracteristicasFaciales).filter(
                CaracteristicasFaciales.usuario_id == usuario.id,
                CaracteristicasFaciales.activa == True
            ).all()

            caracteristicas_data = []
            for carac in caracteristicas:
                caracteristicas_data.append({
                    "id": carac.id,
                    "imagen_id": carac.imagen_id,
                    "algoritmo_version": carac.algoritmo_version,
                    "calidad_deteccion": carac.calidad_deteccion,
                    "fecha_procesamiento": carac.fecha_procesamiento.isoformat(),
                    "tiene_eigenfaces": carac.eigenfaces_vector is not None,
                    "tiene_lbp": carac.lbp_histogram is not None,
                    # No incluir los vectores completos para evitar respuesta muy grande
                    "eigenfaces_size": len(carac.eigenfaces_vector) if carac.eigenfaces_vector else 0,
                    "lbp_size": len(carac.lbp_histogram) if carac.lbp_histogram else 0
                })

            usuario_data["caracteristicas_ml"] = caracteristicas_data
            usuario_data["total_caracteristicas"] = len(caracteristicas_data)

        return ResponseWithData(
            success=True,
            message=f"Usuario '{id_estudiante}' obtenido exitosamente",
            data=usuario_data
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener usuario: {str(e)}")


@router.get("/{usuario_id}", response_model=ResponseWithData, summary="Obtener usuario por ID numérico")
async def obtener_usuario_por_id(
        usuario_id: int,
        incluir_imagenes: bool = Query(True, description="Incluir información de imágenes"),
        incluir_reconocimientos: bool = Query(False, description="Incluir historial de reconocimientos"),
        db: Session = Depends(get_db)
):
    """
    Obtiene un usuario específico por su ID numérico (compatibilidad)
    Redirige a la función principal usando el ID de estudiante si está disponible
    """
    try:
        # Buscar usuario por ID numérico
        usuario = db.query(Usuario).filter(
            Usuario.id == usuario_id,
            Usuario.activo == True
        ).first()

        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        # Si tiene ID de estudiante, redirigir a esa función para consistencia
        if usuario.id_estudiante:
            return await obtener_usuario_por_id_estudiante(
                id_estudiante=usuario.id_estudiante,
                incluir_imagenes=incluir_imagenes,
                incluir_reconocimientos=incluir_reconocimientos,
                incluir_caracteristicas=False,  # Por defecto no incluir
                db=db
            )

        # Si no tiene ID de estudiante, procesar directamente
        # (Lógica similar pero usando ID numérico)
        usuario_data = {
            "id": usuario.id,
            "nombre": usuario.nombre,
            "apellido": usuario.apellido,
            "email": usuario.email,
            "id_estudiante": usuario.id_estudiante,
            "requisitoriado": usuario.requisitoriado,
            "tipo_requisitoria": usuario.tipo_requisitoria,
            "fecha_registro": usuario.fecha_registro.isoformat(),
            "fecha_actualizacion": usuario.fecha_actualizacion.isoformat(),
            "activo": usuario.activo
        }

        # Resto de lógica similar al endpoint por ID de estudiante...
        if incluir_imagenes:
            # Lógica de imágenes igual que arriba
            pass

        return ResponseWithData(
            success=True,
            message="Usuario obtenido exitosamente",
            data=usuario_data
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener usuario: {str(e)}")



@router.put("/{usuario_id}", response_model=ResponseWithData, summary="Actualizar usuario")
async def actualizar_usuario(
        usuario_id: int,
        nombre: Optional[str] = Form(None),
        apellido: Optional[str] = Form(None),
        email: Optional[str] = Form(None),
        id_estudiante: Optional[str] = Form(None),
        activo: Optional[bool] = Form(None),
        db: Session = Depends(get_db)
):
    """
    Actualiza los datos de un usuario existente
    """
    try:
        # Buscar usuario
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        # Actualizar campos proporcionados
        if nombre:
            usuario.nombre = nombre.title()
        if apellido:
            usuario.apellido = apellido.title()
        if email:
            # Verificar que el email no esté en uso por otro usuario
            existing_user = db.query(Usuario).filter(
                Usuario.email == email.lower(),
                Usuario.id != usuario_id
            ).first()
            if existing_user:
                raise HTTPException(status_code=400, detail="El email ya está en uso")
            usuario.email = email.lower()
        if id_estudiante:
            # Verificar que el ID de estudiante no esté en uso
            existing_student = db.query(Usuario).filter(
                Usuario.id_estudiante == id_estudiante,
                Usuario.id != usuario_id
            ).first()
            if existing_student:
                raise HTTPException(status_code=400, detail="El ID de estudiante ya está en uso")
            usuario.id_estudiante = id_estudiante
        if activo is not None:
            usuario.activo = activo

        usuario.fecha_actualizacion = datetime.utcnow()

        db.commit()
        db.refresh(usuario)

        usuario_data = {
            "id": usuario.id,
            "nombre": usuario.nombre,
            "apellido": usuario.apellido,
            "email": usuario.email,
            "id_estudiante": usuario.id_estudiante,
            "requisitoriado": usuario.requisitoriado,
            "tipo_requisitoria": usuario.tipo_requisitoria,
            "activo": usuario.activo,
            "fecha_actualizacion": usuario.fecha_actualizacion.isoformat()
        }

        return ResponseWithData(
            success=True,
            message="Usuario actualizado exitosamente",
            data=usuario_data
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al actualizar usuario: {str(e)}")


@router.delete("/{usuario_id}", response_model=ResponseWithData, summary="Eliminar usuario")
async def eliminar_usuario(
        usuario_id: int,
        eliminar_definitivo: bool = Query(False,
                                          description="Eliminar definitivamente (true) o solo desactivar (false)"),
        db: Session = Depends(get_db)
):
    """
    Elimina un usuario (soft delete por defecto, hard delete opcional)
    """
    try:
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()

        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        if eliminar_definitivo:
            # Eliminar archivos de imágenes
            imagenes = db.query(ImagenFacial).filter(ImagenFacial.usuario_id == usuario_id).all()
            for imagen in imagenes:
                if os.path.exists(imagen.ruta_archivo):
                    os.remove(imagen.ruta_archivo)

            # Eliminar usuario y registros relacionados (CASCADE)
            db.delete(usuario)
            mensaje = "Usuario eliminado definitivamente"
        else:
            # Soft delete
            usuario.activo = False
            usuario.fecha_actualizacion = datetime.utcnow()
            mensaje = "Usuario desactivado"

        db.commit()

        return ResponseWithData(
            success=True,
            message=mensaje,
            data={"id": usuario_id, "eliminado_definitivo": eliminar_definitivo}
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al eliminar usuario: {str(e)}")


@router.post("/{usuario_id}/imagenes", response_model=ResponseWithData, summary="Añadir imágenes a usuario")
async def añadir_imagenes_usuario(
        usuario_id: int,
        imagenes: List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    """
    Añade nuevas imágenes a un usuario existente con entrenamiento automático
    """
    try:
        # Verificar que el usuario existe
        usuario = db.query(Usuario).filter(
            Usuario.id == usuario_id,
            Usuario.activo == True
        ).first()

        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        # Contar imágenes existentes
        imagenes_existentes = db.query(ImagenFacial).filter(
            ImagenFacial.usuario_id == usuario_id,
            ImagenFacial.activa == True
        ).count()

        # Verificar límite de imágenes
        if imagenes_existentes + len(imagenes) > 5:
            raise HTTPException(
                status_code=400,
                detail=f"El usuario ya tiene {imagenes_existentes} imágenes. Máximo permitido: 5"
            )

        # Validar archivos
        for img in imagenes:
            if not validate_image_file(img):
                raise HTTPException(
                    status_code=400,
                    detail=f"Archivo '{img.filename}' no es una imagen válida"
                )

        # Procesar y guardar imágenes
        imagenes_guardadas = []
        imagenes_procesadas = []

        for imagen in imagenes:
            # Generar nombre único
            file_extension = os.path.splitext(imagen.filename)[1]
            unique_filename = f"user_{usuario_id}_{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)

            # Guardar archivo
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(imagen.file, buffer)

            # Crear registro
            imagen_facial = ImagenFacial(
                usuario_id=usuario_id,
                nombre_archivo=imagen.filename,
                ruta_archivo=file_path,
                es_principal=False,  # Las nuevas imágenes no son principales
                formato=file_extension[1:],
                tamano_bytes=os.path.getsize(file_path)
            )

            db.add(imagen_facial)
            imagenes_guardadas.append(imagen_facial)

        db.commit()

        # ENTRENAMIENTO AUTOMÁTICO INTELIGENTE
        try:
            import cv2
            for imagen_facial in imagenes_guardadas:
                img = cv2.imread(imagen_facial.ruta_archivo)
                if img is not None:
                    imagenes_procesadas.append(img)

            # Añadir al modelo ML
            if imagenes_procesadas:
                training_result = ml_service.add_new_person(usuario_id, imagenes_procesadas)

                response_data = {
                    "usuario_id": usuario_id,
                    "imagenes_añadidas": len(imagenes_guardadas),
                    "total_imagenes": imagenes_existentes + len(imagenes_guardadas),
                    "ml_training_status": training_result.get("status", "unknown"),
                    "ml_message": training_result.get("message", ""),
                    "model_updated": training_result.get("status") == "added_incremental"
                }

                print(f"🤖 Resultado ML (imágenes adicionales): {training_result.get('message', 'Sin mensaje')}")
            else:
                response_data = {
                    "usuario_id": usuario_id,
                    "imagenes_añadidas": len(imagenes_guardadas),
                    "total_imagenes": imagenes_existentes + len(imagenes_guardadas),
                    "ml_training_status": "no_processed",
                    "ml_message": "No se procesaron imágenes para ML"
                }

        except Exception as e:
            print(f"⚠️ Error en procesamiento ML: {e}")

            response_data = {
                "usuario_id": usuario_id,
                "imagenes_añadidas": len(imagenes_guardadas),
                "total_imagenes": imagenes_existentes + len(imagenes_guardadas),
                "ml_training_status": "error",
                "ml_message": f"Error en entrenamiento: {str(e)}"
            }

        return ResponseWithData(
            success=True,
            message=f"Se añadieron {len(imagenes_guardadas)} imágenes al usuario",
            data=response_data
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al añadir imágenes: {str(e)}")


@router.get("/{usuario_id}/imagenes", response_model=ResponseWithData, summary="Obtener imágenes del usuario")
async def obtener_imagenes_usuario(
        usuario_id: int,
        db: Session = Depends(get_db)
):
    """
    Obtiene todas las imágenes de un usuario
    """
    try:
        # Verificar que el usuario existe
        usuario = db.query(Usuario).filter(Usuario.id == usuario_id).first()
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        # Obtener imágenes
        imagenes = db.query(ImagenFacial).filter(
            ImagenFacial.usuario_id == usuario_id,
            ImagenFacial.activa == True
        ).order_by(ImagenFacial.es_principal.desc(), ImagenFacial.fecha_subida.desc()).all()

        imagenes_data = [
            {
                "id": img.id,
                "nombre_archivo": img.nombre_archivo,
                "es_principal": img.es_principal,
                "formato": img.formato,
                "tamano_bytes": img.tamano_bytes,
                "ancho": img.ancho,
                "alto": img.alto,
                "fecha_subida": img.fecha_subida.isoformat()
            }
            for img in imagenes
        ]

        return ResponseWithData(
            success=True,
            message=f"Imágenes del usuario obtenidas exitosamente",
            data={
                "usuario_id": usuario_id,
                "total_imagenes": len(imagenes_data),
                "imagenes": imagenes_data
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener imágenes: {str(e)}")


@router.delete("/{usuario_id}/imagenes/{imagen_id}", response_model=ResponseWithData, summary="Eliminar imagen")
async def eliminar_imagen(
        usuario_id: int,
        imagen_id: int,
        db: Session = Depends(get_db)
):
    """
    Elimina una imagen específica de un usuario
    """
    try:
        # Buscar imagen
        imagen = db.query(ImagenFacial).filter(
            ImagenFacial.id == imagen_id,
            ImagenFacial.usuario_id == usuario_id
        ).first()

        if not imagen:
            raise HTTPException(status_code=404, detail="Imagen no encontrada")

        # Eliminar archivo físico
        if os.path.exists(imagen.ruta_archivo):
            os.remove(imagen.ruta_archivo)

        # Eliminar registro de BD
        db.delete(imagen)
        db.commit()

        return ResponseWithData(
            success=True,
            message="Imagen eliminada exitosamente",
            data={"imagen_id": imagen_id, "usuario_id": usuario_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error al eliminar imagen: {str(e)}")


@router.get("/estadisticas/resumen", response_model=ResponseWithData, summary="Estadísticas de usuarios")
async def estadisticas_usuarios(db: Session = Depends(get_db)):
    """
    Obtiene estadísticas generales de usuarios
    """
    try:
        # Conteos básicos
        total_usuarios = db.query(Usuario).count()
        usuarios_activos = db.query(Usuario).filter(Usuario.activo == True).count()
        usuarios_requisitoriados = db.query(Usuario).filter(Usuario.requisitoriado == True).count()

        # Distribución por tipo de requisitoria
        from sqlalchemy import func
        distribucion_requisitorias = db.query(
            Usuario.tipo_requisitoria,
            func.count(Usuario.id).label('count')
        ).filter(
            Usuario.requisitoriado == True
        ).group_by(Usuario.tipo_requisitoria).all()

        distribucion_dict = {tipo: count for tipo, count in distribucion_requisitorias}

        # Estadísticas de imágenes
        total_imagenes = db.query(ImagenFacial).filter(ImagenFacial.activa == True).count()

        estadisticas = {
            "usuarios": {
                "total": total_usuarios,
                "activos": usuarios_activos,
                "requisitoriados": usuarios_requisitoriados,
                "porcentaje_requisitoriados": round((usuarios_requisitoriados / total_usuarios * 100),
                                                    2) if total_usuarios > 0 else 0
            },
            "imagenes": {
                "total": total_imagenes,
                "promedio_por_usuario": round(total_imagenes / usuarios_activos, 2) if usuarios_activos > 0 else 0
            },
            "requisitorias": {
                "distribucion": distribucion_dict,
                "tipos_activos": len(distribucion_dict)
            }
        }

        return ResponseWithData(
            success=True,
            message="Estadísticas obtenidas exitosamente",
            data=estadisticas
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


@router.post("/entrenar-modelo", response_model=ResponseWithData, summary="Entrenar modelo ML con usuarios")
async def entrenar_modelo_usuarios(db: Session = Depends(get_db)):
    """
    Entrena el modelo de ML con todas las imágenes de usuarios activos
    """
    try:
        # Obtener todos los usuarios activos con imágenes
        usuarios_con_imagenes = db.query(Usuario).filter(
            Usuario.activo == True
        ).join(ImagenFacial).filter(
            ImagenFacial.activa == True
        ).distinct().all()

        if not usuarios_con_imagenes:
            raise HTTPException(status_code=400, detail="No hay usuarios con imágenes para entrenar")

        # Preparar datos para entrenamiento
        import cv2
        images_by_person = {}

        for usuario in usuarios_con_imagenes:
            imagenes = db.query(ImagenFacial).filter(
                ImagenFacial.usuario_id == usuario.id,
                ImagenFacial.activa == True
            ).all()

            images_list = []
            for imagen in imagenes:
                if os.path.exists(imagen.ruta_archivo):
                    img = cv2.imread(imagen.ruta_archivo)
                    if img is not None:
                        images_list.append(img)

            if images_list:
                images_by_person[usuario.id] = images_list

        # Entrenar modelo
        training_stats = ml_service.train_models(images_by_person)

        return ResponseWithData(
            success=True,
            message="Modelo entrenado exitosamente",
            data=training_stats
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar modelo: {str(e)}")


@router.get("/entrenamiento/estado", response_model=ResponseWithData, summary="Estado del entrenamiento")
async def estado_entrenamiento():
    """
    Verifica el estado actual del entrenamiento automático
    """
    try:
        status = ml_service.get_training_status()

        return ResponseWithData(
            success=True,
            message="Estado del entrenamiento obtenido",
            data=status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estado: {str(e)}")


@router.post("/entrenamiento/forzar", response_model=ResponseWithData, summary="Forzar reentrenamiento")
async def forzar_entrenamiento():
    """
    Fuerza un reentrenamiento completo desde la base de datos
    """
    try:
        result = ml_service.force_retrain_from_database()

        if result.get("success", True):
            message = "Reentrenamiento completado exitosamente"
        else:
            message = f"Error en reentrenamiento: {result.get('error', 'Error desconocido')}"

        return ResponseWithData(
            success=result.get("success", True),
            message=message,
            data=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al forzar entrenamiento: {str(e)}")
