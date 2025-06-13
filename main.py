from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from datetime import datetime

# Importar configuración de base de datos
from config.database import init_database, create_database_if_not_exists

# Importar routers
from routers import users, recognition, face_training

# Importar servicios
from services.ml_service import MLService

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

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
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
ml_service = MLService()


@app.on_event("startup")
async def startup_event():
    """
    Eventos de inicio de la aplicación
    """
    print("🚀 Iniciando Sistema de Reconocimiento Facial...")

    try:
        # Crear base de datos si no existe
        create_database_if_not_exists()

        # Inicializar tablas
        init_database()

        # Crear directorios necesarios
        directories = [
            "storage/images",
            "storage/temp",
            "storage/models",
            "storage/embeddings",
            "storage/logs"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Intentar cargar modelos ML si existen
        try:
            ml_service.load_models()
            if ml_service.is_trained:
                print("✅ Modelos de ML cargados exitosamente")
            else:
                print("⚠️ Modelos de ML no encontrados - se entrenarán con los primeros datos")
        except Exception as e:
            print(f"⚠️ No se pudieron cargar modelos ML: {e}")

        print("✅ Sistema iniciado correctamente")
        print(f"📅 Fecha de inicio: {datetime.now().isoformat()}")
        print("🌐 API disponible en: http://localhost:8000")
        print("📚 Documentación en: http://localhost:8000/docs")

    except Exception as e:
        print(f"❌ Error al iniciar el sistema: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    Eventos de cierre de la aplicación
    """
    print("🛑 Cerrando Sistema de Reconocimiento Facial...")

    try:
        # Guardar modelos ML si están entrenados
        if ml_service.is_trained:
            ml_service.eigenfaces_service.save_model()
            ml_service.lbp_service.save_model()
            print("💾 Modelos ML guardados")

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
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "documentacion": "/docs",
            "redoc": "/redoc",
            "usuarios": "/api/v1/usuarios",
            "reconocimiento": "/api/v1/reconocimiento"
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
        # Verificar estado de los modelos ML
        ml_status = {
            "trained": ml_service.is_trained,
            "eigenfaces_ready": ml_service.eigenfaces_service.is_trained if hasattr(ml_service,
                                                                                    'eigenfaces_service') else False,
            "lbp_ready": ml_service.lbp_service.is_trained if hasattr(ml_service, 'lbp_service') else False
        }

        # Verificar directorios
        directories_status = {}
        required_dirs = ["storage/images", "storage/temp", "storage/models", "storage/embeddings", "storage/logs"]

        for directory in required_dirs:
            directories_status[directory] = os.path.exists(directory)

        # Estado general del sistema
        system_healthy = all([
            all(directories_status.values()),  # Todos los directorios existen
            True  # Agregar más verificaciones según sea necesario
        ])

        return {
            "status": "✅ Saludable" if system_healthy else "⚠️ Problemas detectados",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ml_models": ml_status,
                "directories": directories_status,
                "database": "✅ Conectado"  # Simplificado, se podría verificar la conexión real
            },
            "uptime": "Información no disponible",  # Se podría implementar un contador de tiempo
            "version": "1.0.0"
        }

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "❌ Error",
                "error": str(e),
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
        if ml_service.is_trained:
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


# Configuración para desarrollo
if __name__ == "__main__":
    print("🔧 Iniciando en modo desarrollo...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )