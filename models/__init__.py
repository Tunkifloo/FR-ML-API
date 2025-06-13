# models/__init__.py
"""
Modelos del sistema de reconocimiento facial
"""

from .database_models import (
    Base,
    Usuario,
    ImagenFacial,
    CaracteristicasFaciales,
    HistorialReconocimiento,
    ModeloEntrenamiento,
    asignar_requisitoriado_aleatorio,
    TIPOS_REQUISITORIAS
)

from .pydantic_models import (
    # Modelos base
    UsuarioBase,
    UsuarioCreate,
    UsuarioUpdate,
    Usuario as UsuarioResponse,
    UsuarioDetallado,

    # Modelos de imagen
    ImagenFacialBase,
    ImagenFacialCreate,
    ImagenFacial as ImagenFacialResponse,

    # Modelos de reconocimiento
    ReconocimientoRequest,
    ReconocimientoResponse,
    HistorialReconocimiento as HistorialReconocimientoResponse,

    # Modelos de entrenamiento
    EntrenamientoRequest,
    EntrenamientoResponse,
    ModeloInfo,
    SistemaInfo,

    # Modelos de respuesta
    ResponseBase,
    ResponseWithData,
    ResponsePaginado,
    ErrorResponse,

    # Enums
    TipoRequisitoria,
    AlgoritmoReconocimiento,
    NivelAlerta
)

__all__ = [
    # Database models
    "Base",
    "Usuario",
    "ImagenFacial",
    "CaracteristicasFaciales",
    "HistorialReconocimiento",
    "ModeloEntrenamiento",
    "asignar_requisitoriado_aleatorio",
    "TIPOS_REQUISITORIAS",

    # Pydantic models
    "UsuarioBase",
    "UsuarioCreate",
    "UsuarioUpdate",
    "UsuarioResponse",
    "UsuarioDetallado",
    "ImagenFacialBase",
    "ImagenFacialCreate",
    "ImagenFacialResponse",
    "ReconocimientoRequest",
    "ReconocimientoResponse",
    "HistorialReconocimientoResponse",
    "EntrenamientoRequest",
    "EntrenamientoResponse",
    "ModeloInfo",
    "SistemaInfo",
    "ResponseBase",
    "ResponseWithData",
    "ResponsePaginado",
    "ErrorResponse",
    "TipoRequisitoria",
    "AlgoritmoReconocimiento",
    "NivelAlerta"
]