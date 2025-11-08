from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# Enums para validación
class TipoRequisitoria(str, Enum):
    HURTO = "Hurto"
    ROBO = "Robo"
    ESTAFA = "Estafa"
    VANDALISMO = "Vandalismo"
    DISTURBIOS = "Disturbios"
    VIOLENCIA_DOMESTICA = "Violencia doméstica"
    FRAUDE = "Fraude"
    TRAFICO = "Tráfico"
    FALSIFICACION = "Falsificación"
    AGRESION = "Agresión"
    AMENAZAS = "Amenazas"
    VIOLACION_MEDIDAS = "Violación de medidas cautelares"


class AlgoritmoReconocimiento(str, Enum):
    EIGENFACES = "eigenfaces"
    LBP = "lbp"
    HYBRID = "hybrid"
    VOTING = "voting"


class NivelAlerta(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Modelos base
class UsuarioBase(BaseModel):
    nombre: str = Field(..., min_length=2, max_length=100, description="Nombre de la persona")
    apellido: str = Field(..., min_length=2, max_length=100, description="Apellido de la persona")
    id_estudiante: Optional[str] = Field(None, max_length=20, description="ID de estudiante (opcional)")
    email: EmailStr = Field(..., description="Correo electrónico")

    @validator('nombre', 'apellido')
    def validate_names(cls, v):
        if not v.replace(' ', '').isalpha():
            raise ValueError('El nombre y apellido solo pueden contener letras y espacios')
        return v.title()

    @validator('id_estudiante')
    def validate_student_id(cls, v):
        if v and not v.isalnum():
            raise ValueError('El ID de estudiante debe ser alfanumérico')
        return v


class UsuarioCreate(UsuarioBase):
    pass


class UsuarioUpdate(BaseModel):
    nombre: Optional[str] = Field(None, min_length=2, max_length=100)
    apellido: Optional[str] = Field(None, min_length=2, max_length=100)
    id_estudiante: Optional[str] = Field(None, max_length=20)
    email: Optional[EmailStr] = None
    activo: Optional[bool] = None


class Usuario(UsuarioBase):
    id: int
    requisitoriado: bool
    tipo_requisitoria: Optional[str]
    fecha_registro: datetime
    fecha_actualizacion: datetime
    activo: bool

    class Config:
        from_attributes = True


class UsuarioDetallado(Usuario):
    total_imagenes: int = 0
    imagen_principal: Optional[str] = None
    ultima_imagen: Optional[datetime] = None
    total_reconocimientos: int = 0
    ultimo_reconocimiento: Optional[datetime] = None


# Modelos para imágenes
class ImagenFacialBase(BaseModel):
    es_principal: bool = Field(False, description="Si es la imagen principal del usuario")


class ImagenFacialCreate(ImagenFacialBase):
    usuario_id: int


class ImagenFacial(ImagenFacialBase):
    id: int
    usuario_id: int
    nombre_archivo: str
    ruta_archivo: str
    ancho: Optional[int]
    alto: Optional[int]
    formato: Optional[str]
    tamano_bytes: Optional[int]
    fecha_subida: datetime
    activa: bool

    class Config:
        from_attributes = True


# Modelos para características faciales
class CaracteristicasFaciales(BaseModel):
    id: int
    usuario_id: int
    imagen_id: int
    eigenfaces_vector: Optional[List[float]]
    lbp_histogram: Optional[List[float]]
    algoritmo_version: str
    calidad_deteccion: int
    fecha_procesamiento: datetime
    activa: bool

    class Config:
        from_attributes = True


# Modelos para reconocimiento
class ReconocimientoRequest(BaseModel):
    algoritmo: AlgoritmoReconocimiento = Field(AlgoritmoReconocimiento.HYBRID,
                                               description="Algoritmo de reconocimiento a usar")
    incluir_detalles: bool = Field(True, description="Incluir detalles técnicos en la respuesta")


class ReconocimientoResponse(BaseModel):
    reconocido: bool
    persona_id: Optional[int]
    confianza: float
    metodo: str
    detalles: Optional[Dict[str, Any]]
    tiempo_procesamiento: Optional[float]
    timestamp: str

    # Información de la persona (si se reconoce)
    persona_info: Optional[Dict[str, Any]] = None

    # Alerta de seguridad (si aplica)
    alerta_seguridad: Optional[Dict[str, Any]] = None


class HistorialReconocimiento(BaseModel):
    id: int
    usuario_id: Optional[int]
    imagen_consulta_path: str
    confianza: int
    distancia_euclidiana: Optional[str]
    reconocido: bool
    alerta_generada: bool
    fecha_reconocimiento: datetime
    ip_origen: Optional[str]

    class Config:
        from_attributes = True


# Modelos para entrenamiento
class EntrenamientoRequest(BaseModel):
    reentrenar_completo: bool = Field(False, description="Si reentrenar completamente o incremental")
    algoritmos: List[AlgoritmoReconocimiento] = Field([AlgoritmoReconocimiento.HYBRID],
                                                      description="Algoritmos a entrenar")


class EntrenamientoResponse(BaseModel):
    exito: bool
    mensaje: str
    estadisticas: Dict[str, Any]
    tiempo_entrenamiento: float
    timestamp: str


class ModeloInfo(BaseModel):
    algoritmo: str
    entrenado: bool
    total_embeddings: int
    personas_unicas: int
    version: str
    ultima_actualizacion: Optional[datetime]
    rendimiento: Optional[Dict[str, float]]


class SistemaInfo(BaseModel):
    version_sistema: str
    modelos: List[ModeloInfo]
    estadisticas_generales: Dict[str, Any]
    configuracion: Dict[str, Any]


# Modelos para alertas
class AlertaSeguridad(BaseModel):
    id: str
    persona_id: int
    nivel_alerta: NivelAlerta
    tipo_requisitoria: str
    confianza: float
    mensaje: str
    timestamp: str
    ubicacion: Optional[str]
    acciones_recomendadas: List[str]
    notificacion_autoridades: Dict[str, Any]


class AlertaHistorial(BaseModel):
    total_alertas: int
    por_nivel: Dict[str, int]
    por_tipo_requisitoria: Dict[str, int]
    promedio_diario: float
    ultima_alerta: Optional[datetime]


# Modelos para respuestas de la API
class ResponseBase(BaseModel):
    success: bool
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ResponseWithData(ResponseBase):
    data: Optional[Any] = None


class ResponsePaginado(ResponseBase):
    data: List[Any] = []
    total: int = 0
    pagina: int = 1
    items_por_pagina: int = 10
    total_paginas: int = 0


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detalles: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Modelos para configuración
class ConfiguracionSistema(BaseModel):
    eigenfaces_weight: float = Field(0.6, ge=0.0, le=1.0, description="Peso del algoritmo Eigenfaces")
    lbp_weight: float = Field(0.4, ge=0.0, le=1.0, description="Peso del algoritmo LBP")
    umbral_confianza: float = Field(70.0, ge=0.0, le=100.0, description="Umbral de confianza para reconocimiento")
    metodo_combinacion: str = Field("weighted_average", description="Método de combinación de algoritmos")
    max_imagenes_por_persona: int = Field(15, ge=1, le=20, description="Máximo de imágenes por persona")
    @validator('eigenfaces_weight', 'lbp_weight')
    def validate_weights_sum(cls, v, values):
        if 'eigenfaces_weight' in values and 'lbp_weight' in values:
            if abs((values['eigenfaces_weight'] + v) - 1.0) > 0.01:
                raise ValueError('La suma de pesos debe ser 1.0')
        return v


# Modelos para estadísticas
class EstadisticasUsuarios(BaseModel):
    total_usuarios: int
    usuarios_activos: int
    usuarios_requisitoriados: int
    porcentaje_requisitoriados: float
    distribucion_por_tipo: Dict[str, int]


class EstadisticasImagenes(BaseModel):
    total_imagenes: int
    imagenes_activas: int
    promedio_por_usuario: float
    tamaño_total_mb: float
    formatos_distribucion: Dict[str, int]


class EstadisticasReconocimientos(BaseModel):
    total_reconocimientos: int
    reconocimientos_exitosos: int
    tasa_exito: float
    confianza_promedio: float
    alertas_generadas: int
    reconocimientos_por_dia: Dict[str, int]


class Dashboard(BaseModel):
    usuarios: EstadisticasUsuarios
    imagenes: EstadisticasImagenes
    reconocimientos: EstadisticasReconocimientos
    alertas: AlertaHistorial
    sistema: SistemaInfo
    ultima_actualizacion: datetime


# Modelos para filtros y búsquedas
class FiltroUsuarios(BaseModel):
    nombre: Optional[str] = None
    apellido: Optional[str] = None
    email: Optional[str] = None
    requisitoriado: Optional[bool] = None
    tipo_requisitoria: Optional[TipoRequisitoria] = None
    activo: Optional[bool] = None
    fecha_inicio: Optional[datetime] = None
    fecha_fin: Optional[datetime] = None


class FiltroReconocimientos(BaseModel):
    usuario_id: Optional[int] = None
    reconocido: Optional[bool] = None
    alerta_generada: Optional[bool] = None
    confianza_minima: Optional[float] = None
    fecha_inicio: Optional[datetime] = None
    fecha_fin: Optional[datetime] = None


class Ordenamiento(BaseModel):
    campo: str
    direccion: str = Field("asc", pattern="^(asc|desc)$")  # Cambiado de regex a pattern


class Paginacion(BaseModel):
    pagina: int = Field(1, ge=1, description="Número de página")
    items_por_pagina: int = Field(10, ge=1, le=100, description="Items por página")


# Modelos para reportes
class ReporteRequest(BaseModel):
    tipo_reporte: str = Field(...,
                              pattern="^(usuarios|reconocimientos|alertas|rendimiento)$")  # Cambiado de regex a pattern
    fecha_inicio: datetime
    fecha_fin: datetime
    filtros: Optional[Dict[str, Any]] = None
    formato: str = Field("json", pattern="^(json|csv|pdf)$")  # Cambiado de regex a pattern


class ReporteResponse(BaseModel):
    tipo_reporte: str
    periodo: Dict[str, str]
    datos: Dict[str, Any]
    estadisticas: Dict[str, Any]
    graficos: Optional[Dict[str, Any]] = None
    generado_en: datetime
    formato: str