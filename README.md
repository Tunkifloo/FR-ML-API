# 🤖 Sistema de Gestión y Reconocimiento Facial

Sistema completo de reconocimiento facial desarrollado con **FastAPI + MySQL** implementando algoritmos de Machine Learning **desde cero** (sin modelos pre-entrenados).

## 📋 Características Principales

### ✅ Funcionalidades Implementadas

- **CRUD Completo de Usuarios** con 1-5 imágenes por persona
- **Reconocimiento Facial** usando algoritmos implementados desde cero
- **Sistema de Alertas** automático para personas requisitoriadas
- **Entrenamiento Continuo** (incremental) del modelo
- **Historial Completo** de reconocimientos y estadísticas
- **API REST** completamente documentada

### 🧠 Algoritmos de Machine Learning (Implementación Propia)

#### 1. **Eigenfaces (PCA)**
- Implementación completa de Análisis de Componentes Principales
- Extracción de características faciales usando vectores propios
- Reducción de dimensionalidad eficiente
- **No usa modelos pre-entrenados**

#### 2. **Local Binary Patterns (LBP)**  
- Implementación desde cero de patrones binarios locales
- Análisis de texturas faciales robustas
- Resistente a cambios de iluminación
- **Algoritmo implementado completamente**

#### 3. **Sistema Híbrido**
- Combinación inteligente de Eigenfaces + LBP
- Múltiples métodos de fusión (promedio ponderado, votación, cascada)
- Mejora significativa en precisión

### 🚨 Sistema de Seguridad

- **Detección Automática** de personas requisitoriadas
- **Alertas por Niveles** (HIGH, MEDIUM, LOW)
- **Simulación de Notificación** a autoridades
- **Registro Completo** de incidentes de seguridad
- **Acciones Recomendadas** automáticas según el tipo de requisitoria

## 🛠️ Tecnologías Utilizadas

- **Backend**: FastAPI + Python 3.9
- **Base de Datos**: MySQL + SQLAlchemy ORM
- **Machine Learning**: 
  - NumPy + scikit-learn (solo para PCA básico)
  - OpenCV (procesamiento de imágenes)
  - scikit-image (LBP implementation)
  - **Algoritmos principales implementados desde cero**
- **Almacenamiento**: Sistema de archivos + JSON para embeddings

## 📦 Instalación y Configuración

### Prerrequisitos

- Python 3.9+
- MySQL 8.0+
- Git

### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd FR-ML
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Base de Datos

1. **Crear base de datos MySQL:**
```sql
CREATE DATABASE face_recognition_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. **Configurar variables de entorno:**
   - Copiar `.env.example` a `.env`
   - Editar credenciales de MySQL en `.env`

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=tu_password
DB_NAME=face_recognition_db
```

### 5. Inicializar Sistema

```bash
python init_database.py
```

Este script:
- ✅ Crea la base de datos si no existe
- ✅ Crea todas las tablas necesarias
- ✅ Genera usuarios de ejemplo con estado de requisitoriado aleatorio
- ✅ Crea directorios necesarios

## 🚀 Ejecución

### Desarrollo
```bash
python main.py
```

### Producción
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

El sistema estará disponible en:
- **API**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 Uso del Sistema

### 1. Crear Usuario con Imágenes

```bash
curl -X POST "http://localhost:8000/api/v1/usuarios/" \
  -H "Content-Type: multipart/form-data" \
  -F "nombre=Juan" \
  -F "apellido=Pérez" \
  -F "email=juan@example.com" \
  -F "id_estudiante=000243425" \
  -F "imagenes=@foto1.jpg" \
  -F "imagenes=@foto2.jpg"
```

### 2. Entrenar Modelo ML

```bash
curl -X POST "http://localhost:8000/api/v1/usuarios/entrenar-modelo"
```

### 3. Reconocer Rostro

```bash
curl -X POST "http://localhost:8000/api/v1/reconocimiento/identificar" \
  -H "Content-Type: multipart/form-data" \
  -F "imagen=@rostro_desconocido.jpg" \
  -F "algoritmo=hybrid"
```

## 🏗️ Arquitectura del Sistema

```
FR-ML/
├── main.py                     # Aplicación FastAPI principal
├── config/
│   └── database.py            # Configuración de base de datos
├── models/
│   ├── database_models.py     # Modelos SQLAlchemy
│   └── pydantic_models.py     # Modelos API (validación)
├── routers/
│   ├── users.py              # CRUD de usuarios
│   └── recognition.py        # Reconocimiento facial
├── services/
│   ├── ml_service.py         # Servicio ML principal (híbrido)
│   ├── eigenfaces_service.py # Implementación Eigenfaces
│   ├── lbp_service.py        # Implementación LBP
│   └── face_detection_service.py # Detección de rostros
├── utils/
│   ├── alert_system.py       # Sistema de alertas
│   └── image_processor.py    # Procesamiento de imágenes
└── storage/
    ├── images/               # Imágenes de usuarios
    ├── models/               # Modelos ML serializados
    ├── embeddings/           # Características extraídas (JSON)
    └── logs/                 # Logs del sistema
```

## 🔬 Detalles Técnicos de los Algoritmos

### Eigenfaces (PCA)

1. **Preprocesamiento**: Normalización, redimensionado, ecualización
2. **Cálculo de cara promedio**: Media de todas las imágenes de entrenamiento
3. **Centrado de datos**: Substracción de la cara promedio
4. **Descomposición PCA**: Cálculo de vectores y valores propios
5. **Proyección**: Transformación al espacio de eigenfaces
6. **Reconocimiento**: Distancia euclidiana entre vectores de características

### Local Binary Patterns (LBP)

1. **División en grilla**: Imagen dividida en celdas 8x8
2. **Cálculo LBP**: Comparación con vecinos en patrón circular
3. **Histogramas**: Generación de histograma por celda
4. **Concatenación**: Vector de características final
5. **Comparación**: Similitud coseno entre histogramas

### Sistema Híbrido

- **Promedio Ponderado**: 60% Eigenfaces + 40% LBP
- **Votación**: Consenso entre algoritmos
- **Cascada**: Eigenfaces primario, LBP verificación

## 🚨 Sistema de Alertas

### Niveles de Alerta

- **HIGH**: Confianza ≥80% o delitos graves (Robo, Violencia, Tráfico)
- **MEDIUM**: Confianza ≥60% o delitos moderados
- **LOW**: Confianza ≥40% o delitos menores

### Tipos de Requisitoria Implementados

- Hurto, Robo, Estafa, Vandalismo
- Disturbios, Violencia doméstica
- Fraude, Tráfico, Falsificación
- Agresión, Amenazas, Violación de medidas cautelares

## 📊 Endpoints de la API

### Usuarios
- `GET /api/v1/usuarios/` - Listar usuarios (paginado + filtros)
- `POST /api/v1/usuarios/` - Crear usuario con imágenes
- `GET /api/v1/usuarios/{id}` - Obtener usuario específico  
- `PUT /api/v1/usuarios/{id}` - Actualizar usuario
- `DELETE /api/v1/usuarios/{id}` - Eliminar usuario
- `POST /api/v1/usuarios/{id}/imagenes` - Añadir imágenes
- `POST /api/v1/usuarios/entrenar-modelo` - Entrenar modelo ML

### Reconocimiento  
- `POST /api/v1/reconocimiento/identificar` - Identificar persona
- `GET /api/v1/reconocimiento/historial` - Historial de reconocimientos
- `GET /api/v1/reconocimiento/estadisticas` - Estadísticas del sistema
- `GET /api/v1/reconocimiento/modelo/info` - Información del modelo
- `GET /api/v1/reconocimiento/alertas/historial` - Historial de alertas

### Sistema
- `GET /` - Información general
- `GET /health` - Estado del sistema
- `GET /info/sistema` - Información detallada

## 🧪 Testing

### Probar Reconocimiento
```bash
curl -X POST "http://localhost:8000/api/v1/reconocimiento/test-reconocimiento"
```

### Probar Sistema de Alertas
```bash
curl -X POST "http://localhost:8000/api/v1/reconocimiento/alertas/test"
```

## 📈 Estadísticas y Reportes

El sistema genera automáticamente:

- **Estadísticas de usuarios**: Total, activos, requisitoriados
- **Métricas de reconocimiento**: Tasa de éxito, confianza promedio
- **Análisis de alertas**: Por nivel, tipo, tendencias temporales
- **Rendimiento del modelo**: Precisión por algoritmo

## 🔒 Consideraciones de Seguridad

- Validación exhaustiva de archivos de imagen
- Límites de tamaño y cantidad de archivos
- Logs de auditoría completos
- Separación de datos temporales y permanentes
- Simulación segura de notificaciones a autoridades

## 🚀 Características Avanzadas

### Entrenamiento Continuo
- Añadir nuevas personas sin reentrenar completamente
- Actualización automática de embeddings
- Versioning de modelos

### Optimizaciones
- Cache de características faciales en JSON
- Procesamiento asíncrono de imágenes
- Limpieza automática de archivos temporales

## 🤝 Cumplimiento del Proyecto

### ✅ Requerimientos Implementados

- **CRUD Completo**: ✅ Usuarios con 1-5 imágenes
- **ML Sin Pre-entrenados**: ✅ Eigenfaces + LBP desde cero
- **Reconocimiento Facial**: ✅ Multiple algoritmos
- **Sistema de Alertas**: ✅ Personas requisitoriadas
- **Entrenamiento Adaptativo**: ✅ Incremental
- **Base de Datos**: ✅ MySQL + SQLAlchemy
- **API REST**: ✅ FastAPI documentada
- **Campo ID Independiente**: ✅ PK + ID_Estudiante opcional
- **Asignación Aleatoria**: ✅ Requisitoriado automático

### 🧠 Innovaciones Implementadas

- **Sistema Híbrido**: Combinación de múltiples algoritmos
- **Alertas Inteligentes**: Niveles automáticos según contexto
- **Entrenamiento Incremental**: Eficiencia en recursos
- **Embeddings Persistentes**: Almacenamiento optimizado
- **API Comprehensive**: Documentación interactiva completa

## 📝 Licencia

MIT License - Ver archivo LICENSE para detalles.

---

**Desarrollado cumpliendo estrictamente con los requerimientos del proyecto académico de Sistemas de Gestión y Reconocimiento Facial.**