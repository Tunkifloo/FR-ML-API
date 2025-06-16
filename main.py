import os
from datetime import datetime

# ===== CREAR DIRECTORIOS ANTES DE CUALQUIER IMPORT =====
# Esto debe ir ANTES de cualquier import que pueda crear archivos
directories = [
    "storage/images",
    "storage/temp",
    "storage/models",
    "storage/embeddings",
    "storage/logs"
]

print("🔄 Creando directorios de almacenamiento...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"   ✅ {directory}")

# ===== AHORA SÍ IMPORTAR TODO =====
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Importar configuración de base de datos
from config.database import init_database, create_database_if_not_exists, test_connection

# Importar routers (ahora que los directorios ya existen)
from routers import users, recognition, face_training

# Importar servicios
from services.ml_service import MLService

# Detectar entorno Railway
RAILWAY_ENVIRONMENT = os.getenv('RAILWAY_ENVIRONMENT') is not None
PORT = int(os.getenv('PORT', 8000))

# Configuración de la aplicación
app = FastAPI(
    title="Sistema de Gestión y Reconocimiento Facial",
    description="""
    # Sistema de Reconocimiento Facial con ML

    Sistema completo de gestión y reconocimiento facial implementado con:

    ## 🤖 Algoritmos de Machine Learning (Sin modelos pre-entrenados)
    - **Eigenfaces (PCA)**: Análisis de componentes principales para extracción de características
    - **Local Binary Patterns (LBP)**: Análisis de patrones locales para reconocimiento robusto
    - **Algoritmo Híbrido**: Combinación inteligente de ambos métodos

    ## 🔍 Características Principales
    - ✅ **CRUD Completo** de usuarios con 1-5 imágenes por persona
    - ✅ **Reconocimiento Facial** en tiempo real
    - ✅ **Sistema de Alertas** automático para personas requisitoriadas
    - ✅ **Entrenamiento Continuo** (incremental) del modelo
    - ✅ **Historial Completo** de reconocimientos
    - ✅ **Estadísticas Avanzadas** y reportes

    ## 🚨 Sistema de Seguridad
    - Detección automática de personas requisitoriadas
    - Generación de alertas con niveles de prioridad
    - Simulación de notificación a autoridades
    - Registro completo de incidentes

    ## 📊 Tecnologías Utilizadas
    - **Backend**: FastAPI + Python 3.9
    - **Base de Datos**: MySQL con SQLAlchemy
    - **ML**: Eigenfaces (PCA) + LBP implementados desde cero
    - **Visión Computacional**: OpenCV + scikit-image
    - **Almacenamiento**: Sistema de archivos + JSON para embeddings

    ---

    **Desarrollado cumpliendo estrictamente con los requerimientos del proyecto académico.**
    **🚂 Desplegado en Railway con auto-detection**
    """,
    version="1.0.0",
    contact={
        "name": "Sistema de Reconocimiento Facial",
        "email": "admin@reconocimiento-facial.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Configurar CORS para Railway
allowed_origins = ["*"]  # En producción, especificar dominios específicos

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos (opcional)
if os.path.exists("storage/images"):
    app.mount("/images", StaticFiles(directory="storage/images"), name="images")

# Incluir routers
app.include_router(users.router, prefix="/api/v1")
app.include_router(recognition.router, prefix="/api/v1")
app.include_router(face_training.router, prefix="/api/v1")

# Inicializar servicios
ml_service = None


@app.on_event("startup")
async def startup_event():
    """
    Eventos de inicio de la aplicación
    """
    global ml_service

    print("🚀 Iniciando Sistema de Reconocimiento Facial...")
    print(f"🌍 Entorno: {'Railway' if RAILWAY_ENVIRONMENT else 'Local'}")
    print(f"🔌 Puerto: {PORT}")

    try:
        # Los directorios ya se crearon al inicio del archivo
        print("✅ Directorios de almacenamiento ya creados")

        # Verificar conexión a base de datos PRIMERO
        print("🔄 Verificando conexión a base de datos...")
        if not test_connection():
            print("❌ Error crítico: No se puede conectar a la base de datos")
            raise Exception("Conexión a base de datos falló")

        # Crear base de datos si no existe (solo local)
        create_database_if_not_exists()

        # Inicializar tablas
        print("🔄 Inicializando estructura de base de datos...")
        init_database()

        # Inicializar servicio ML
        print("🔄 Inicializando servicios de Machine Learning...")
        try:
            ml_service = MLService()

            # Intentar cargar modelos ML si existen
            ml_service.load_models()
            if ml_service.is_trained:
                print("✅ Modelos de ML cargados exitosamente")
            else:
                print("⚠️ Modelos de ML no encontrados - se entrenarán con los primeros datos")

        except Exception as e:
            print(f"⚠️ Warning en ML service: {e}")
            print("   Se creará un servicio ML básico")
            ml_service = MLService()

        print("=" * 60)
        print("✅ SISTEMA INICIADO CORRECTAMENTE")
        print("=" * 60)
        print(f"📅 Fecha: {datetime.now().isoformat()}")

        if RAILWAY_ENVIRONMENT:
            print("🚂 Desplegado en Railway")
            print("🌐 API disponible en el dominio público de Railway")
        else:
            print(f"🌐 API local: http://localhost:{PORT}")
            print(f"📚 Docs: http://localhost:{PORT}/docs")

    except Exception as e:
        print("=" * 60)
        print("❌ ERROR CRÍTICO AL INICIAR SISTEMA")
        print("=" * 60)
        print(f"Error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Eventos de cierre de la aplicación
    """
    print("🛑 Cerrando Sistema de Reconocimiento Facial...")

    try:
        # Guardar modelos ML si están entrenados
        if ml_service and ml_service.is_trained:
            print("💾 Guardando modelos ML...")
            if hasattr(ml_service, 'eigenfaces_service'):
                ml_service.eigenfaces_service.save_model()
            if hasattr(ml_service, 'lbp_service'):
                ml_service.lbp_service.save_model()
            print("✅ Modelos ML guardados")

        print("✅ Sistema cerrado correctamente")

    except Exception as e:
        print(f"⚠️ Error al cerrar el sistema: {e}")


@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint raíz con información del sistema
    """
    return {
        "message": "🤖 Sistema de Reconocimiento Facial - API REST",
        "version": "1.0.0",
        "status": "✅ Activo",
        "environment": "Railway" if RAILWAY_ENVIRONMENT else "Local",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "documentacion": "/docs",
            "redoc": "/redoc",
            "usuarios": "/api/v1/usuarios",
            "reconocimiento": "/api/v1/reconocimiento",
            "health": "/health",
            "info": "/info/sistema"
        },
        "características": [
            "CRUD completo de usuarios",
            "Reconocimiento facial con Eigenfaces + LBP",
            "Sistema de alertas para requisitoriados",
            "Entrenamiento continuo del modelo",
            "Historial y estadísticas completas"
        ]
    }


@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Endpoint de verificación de salud del sistema
    """
    try:
        # Verificar conexión a base de datos
        db_connected = test_connection()

        # Verificar estado de los modelos ML
        ml_status = {
            "service_initialized": ml_service is not None,
            "trained": ml_service.is_trained if ml_service else False,
            "eigenfaces_ready": (ml_service.eigenfaces_service.is_trained
                                 if ml_service and hasattr(ml_service, 'eigenfaces_service')
                                 else False),
            "lbp_ready": (ml_service.lbp_service.is_trained
                          if ml_service and hasattr(ml_service, 'lbp_service')
                          else False)
        }

        # Verificar directorios
        directories_status = {}
        required_dirs = ["storage/images", "storage/temp", "storage/models", "storage/embeddings", "storage/logs"]

        for directory in required_dirs:
            directories_status[directory] = os.path.exists(directory)

        # Estado general del sistema
        system_healthy = all([
            db_connected,  # Base de datos conectada
            ml_service is not None,  # Servicio ML inicializado
            all(directories_status.values()),  # Todos los directorios existen
        ])

        return {
            "status": "✅ Saludable" if system_healthy else "⚠️ Problemas detectados",
            "environment": "Railway" if RAILWAY_ENVIRONMENT else "Local",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": "✅ Conectado" if db_connected else "❌ Desconectado",
                "ml_service": ml_status,
                "directories": directories_status,
            },
            "version": "1.0.0"
        }

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "❌ Error",
                "error": str(e),
                "environment": "Railway" if RAILWAY_ENVIRONMENT else "Local",
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/info/sistema", tags=["System Info"])
async def info_sistema():
    """
    Información detallada del sistema
    """
    try:
        # Información de los modelos ML
        system_info = {}
        if ml_service and ml_service.is_trained:
            system_info = ml_service.get_system_info()

        # Estadísticas de archivos
        file_stats = {}
        storage_dirs = ["storage/images", "storage/temp", "storage/models", "storage/embeddings"]

        for directory in storage_dirs:
            if os.path.exists(directory):
                files = os.listdir(directory)
                file_stats[directory] = {
                    "total_files": len(files),
                    "exists": True
                }
            else:
                file_stats[directory] = {
                    "total_files": 0,
                    "exists": False
                }

        return {
            "sistema": {
                "nombre": "Sistema de Reconocimiento Facial",
                "version": "1.0.0",
                "environment": "Railway" if RAILWAY_ENVIRONMENT else "Local",
                "estado": "Activo",
                "timestamp": datetime.now().isoformat()
            },
            "ml_models": system_info,
            "almacenamiento": file_stats,
            "endpoints_disponibles": [
                "/api/v1/usuarios - Gestión de usuarios",
                "/api/v1/reconocimiento - Reconocimiento facial",
                "/docs - Documentación interactiva",
                "/health - Estado del sistema"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener información del sistema: {str(e)}")


# Manejo global de errores
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint no encontrado",
            "message": f"La ruta solicitada no existe: {request.url.path}",
            "timestamp": datetime.now().isoformat(),
            "available_endpoints": [
                "/docs - Documentación",
                "/api/v1/usuarios - Gestión de usuarios",
                "/api/v1/reconocimiento - Reconocimiento facial"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Error interno del servidor",
            "message": "Ha ocurrido un error inesperado en el sistema",
            "timestamp": datetime.now().isoformat(),
            "support": "Contacte al administrador del sistema"
        }
    )


# Configuración para desarrollo y Railway
if __name__ == "__main__":
    if RAILWAY_ENVIRONMENT:
        print("🚂 Iniciando en Railway...")
    else:
        print("🔧 Iniciando en modo desarrollo...")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False if RAILWAY_ENVIRONMENT else True,
        log_level="info",
        access_log=True
    )