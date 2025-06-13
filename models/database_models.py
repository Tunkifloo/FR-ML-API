from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import random

Base = declarative_base()


class Usuario(Base):
    __tablename__ = "usuarios"

    # ID principal (PK autoincremental)
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Datos básicos del usuario
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    id_estudiante = Column(String(20), nullable=True, unique=True)  # Campo opcional alfanumérico
    email = Column(String(255), nullable=False, unique=True)

    # Estado de requisitoriado (se asigna aleatoriamente)
    requisitoriado = Column(Boolean, default=False)
    tipo_requisitoria = Column(String(100), nullable=True)  # Tipo de infracción/requisitoria

    # Metadatos
    fecha_registro = Column(DateTime, default=datetime.utcnow)
    fecha_actualizacion = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    activo = Column(Boolean, default=True)

    # Relaciones
    imagenes = relationship("ImagenFacial", back_populates="usuario", cascade="all, delete-orphan")
    caracteristicas = relationship("CaracteristicasFaciales", back_populates="usuario", cascade="all, delete-orphan")
    reconocimientos = relationship("HistorialReconocimiento", back_populates="usuario")


class ImagenFacial(Base):
    __tablename__ = "imagenes_faciales"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)

    # Información de la imagen
    nombre_archivo = Column(String(255), nullable=False)
    ruta_archivo = Column(String(500), nullable=False)
    es_principal = Column(Boolean, default=False)  # Una imagen principal por usuario

    # Metadatos de la imagen
    ancho = Column(Integer)
    alto = Column(Integer)
    formato = Column(String(10))  # jpg, png, etc.
    tamano_bytes = Column(Integer)

    fecha_subida = Column(DateTime, default=datetime.utcnow)
    activa = Column(Boolean, default=True)

    # Relaciones
    usuario = relationship("Usuario", back_populates="imagenes")


class CaracteristicasFaciales(Base):
    __tablename__ = "caracteristicas_faciales"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    imagen_id = Column(Integer, ForeignKey("imagenes_faciales.id"), nullable=False)

    # Características extraídas (embeddings)
    eigenfaces_vector = Column(JSON, nullable=True)  # Vector de características Eigenfaces
    lbp_histogram = Column(JSON, nullable=True)  # Histograma LBP

    # Metadatos del procesamiento
    algoritmo_version = Column(String(50), default="v1.0")
    calidad_deteccion = Column(Integer, default=0)  # Puntuación de calidad 0-100

    fecha_procesamiento = Column(DateTime, default=datetime.utcnow)
    activa = Column(Boolean, default=True)

    # Relaciones
    usuario = relationship("Usuario", back_populates="caracteristicas")


class HistorialReconocimiento(Base):
    __tablename__ = "historial_reconocimientos"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=True)  # Null si no se reconoce

    # Información del reconocimiento
    imagen_consulta_path = Column(String(500), nullable=False)
    confianza = Column(Integer, default=0)  # Porcentaje de confianza 0-100
    distancia_euclidiana = Column(String(20))  # Distancia calculada

    # Resultado del reconocimiento
    reconocido = Column(Boolean, default=False)
    alerta_generada = Column(Boolean, default=False)  # Si se generó alerta por requisitoriado

    # Características de la imagen consultada
    caracteristicas_consulta = Column(JSON, nullable=True)

    fecha_reconocimiento = Column(DateTime, default=datetime.utcnow)
    ip_origen = Column(String(45))  # IP desde donde se hizo la consulta

    # Relaciones
    usuario = relationship("Usuario", back_populates="reconocimientos")


class ModeloEntrenamiento(Base):
    __tablename__ = "modelos_entrenamiento"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Información del modelo
    version = Column(String(20), nullable=False)
    algoritmo = Column(String(50), nullable=False)  # eigenfaces, lbp, hibrido

    # Estadísticas del entrenamiento
    total_usuarios = Column(Integer, default=0)
    total_imagenes = Column(Integer, default=0)
    precision_promedio = Column(String(10))  # Precisión del modelo

    # Archivos del modelo
    ruta_modelo_eigenfaces = Column(String(500))
    ruta_modelo_lbp = Column(String(500))
    configuracion = Column(JSON)  # Parámetros de configuración

    fecha_entrenamiento = Column(DateTime, default=datetime.utcnow)
    activo = Column(Boolean, default=True)


# Lista de tipos de requisitorias para asignar aleatoriamente
TIPOS_REQUISITORIAS = [
    "Hurto", "Robo", "Estafa", "Vandalismo", "Disturbios",
    "Violencia doméstica", "Fraude", "Tráfico", "Falsificación",
    "Agresión", "Amenazas", "Violación de medidas cautelares"
]


def asignar_requisitoriado_aleatorio():
    """
    Asigna aleatoriamente si una persona está requisitoriada (30% de probabilidad)
    y el tipo de requisitoria si corresponde.
    """
    es_requisitoriado = random.choice([True, False, False, False])  # 25% probabilidad
    tipo_requisitoria = None

    if es_requisitoriado:
        tipo_requisitoria = random.choice(TIPOS_REQUISITORIAS)

    return es_requisitoriado, tipo_requisitoria