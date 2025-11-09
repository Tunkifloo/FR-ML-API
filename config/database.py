from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Cargar variables de entorno
load_dotenv()

# Detectar si estamos en Railway
RAILWAY_ENVIRONMENT = os.getenv('RAILWAY_ENVIRONMENT') is not None
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

print(f"🌍 Entorno detectado: {ENVIRONMENT}")
print(f"🚂 Railway: {'Sí' if RAILWAY_ENVIRONMENT else 'No'}")

# Configuración de base de datos con prioridad para variables de Railway
if RAILWAY_ENVIRONMENT or ENVIRONMENT == 'production':
    # Variables de Railway MySQL (tienen prioridad)
    DB_HOST = os.getenv('MYSQLHOST') or os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('MYSQLUSER') or os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('MYSQLPASSWORD') or os.getenv('DB_PASSWORD')
    DB_NAME = os.getenv('MYSQLDATABASE') or os.getenv('DB_NAME', 'railway')
    DB_PORT = os.getenv('MYSQLPORT') or os.getenv('DB_PORT', '3306')

    print(f"🚂 Railway MySQL Config:")
    print(f"   Host: {DB_HOST}")
    print(f"   Usuario: {DB_USER}")
    print(f"   Base de datos: {DB_NAME}")
    print(f"   Puerto: {DB_PORT}")

else:
    # Configuración local para desarrollo
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '@dmin')
    DB_NAME = os.getenv('DB_NAME', 'face_recognition_db')
    DB_PORT = os.getenv('DB_PORT', '3306')

if not DB_PASSWORD or not DB_NAME:
    print("⚠️ Advertencia: Configura DB_PASSWORD y DB_NAME en el archivo .env")

# Codificar la contraseña para URLs (maneja caracteres especiales)
encoded_password = quote_plus(DB_PASSWORD)

# Construir URL de conexión
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"🔗 Conectando a: mysql+pymysql://{DB_USER}:****@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Configuración del engine según el entorno
engine_config = {
    "echo": False,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "pool_size": 5,
    "max_overflow": 10,
    "connect_args": {
        "charset": "utf8mb4",
        "connect_timeout": 30
    }
}
print("Configuración de base de datos LOCAL aplicada")

# Crear el engine
try:
    engine = create_engine(DATABASE_URL, **engine_config)
    print("✅ Engine de base de datos creado")
except Exception as e:
    print(f"❌ Error creando engine: {e}")
    raise

# Crear sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Dependency para obtener la sesión de DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """
    Inicializa la base de datos creando todas las tablas
    """
    try:
        print("🔄 Inicializando tablas de base de datos...")
        from models.database_models import Base
        Base.metadata.create_all(bind=engine)
        print("✅ Base de datos inicializada correctamente")

        # Verificar conexión
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            print(f"✅ Conexión verificada: {result.fetchone()}")

    except Exception as e:
        print(f"❌ Error inicializando base de datos: {e}")
        raise


def drop_all_tables():
    """
    Elimina todas las tablas (útil para desarrollo)
    """
    if ENVIRONMENT == 'production':
        print("🚫 No se pueden eliminar tablas en producción")
        return

    try:
        from models.database_models import Base
        Base.metadata.drop_all(bind=engine)
        print("🗑️ Todas las tablas eliminadas")
    except Exception as e:
        print(f"❌ Error eliminando tablas: {e}")


def create_database_if_not_exists():
    """
    Crea la base de datos si no existe
    """
    print("🔄 Verificando existencia de base de datos...")

    try:
        # Crear conexión SIN especificar la base de datos
        temp_url = f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/"
        temp_engine = create_engine(temp_url, echo=False)

        # Intentar crear la base de datos
        with temp_engine.connect() as conn:
            # Verificar si existe
            result = conn.execute(text(f"SHOW DATABASES LIKE '{DB_NAME}'"))
            exists = result.fetchone() is not None

            if exists:
                print(f"✅ Base de datos '{DB_NAME}' ya existe")
            else:
                print(f"📝 Creando base de datos '{DB_NAME}'...")
                conn.execute(text(f"CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.commit()
                print(f"✅ Base de datos '{DB_NAME}' creada exitosamente")

        temp_engine.dispose()
        return True

    except Exception as e:
        print(f"❌ Error creando base de datos: {e}")
        print(f"💡 Crear manualmente con: CREATE DATABASE {DB_NAME};")
        return False


def test_connection() -> bool:
    """
    Prueba la conexión a la base de datos
    """
    print("🔄 Probando conexión a la base de datos...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.fetchone():
                print("✅ Conexión exitosa a la base de datos")
                return True
        return False
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False


# Verificación automática al importar (solo en desarrollo)
if __name__ == "__main__":
    print("🔧 Probando configuración de base de datos...")
    test_connection()