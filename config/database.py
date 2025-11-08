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
    Crea la base de datos si no existe (solo para desarrollo local)
    """
    if RAILWAY_ENVIRONMENT or ENVIRONMENT == 'production':
        print("🚂 Railway: Base de datos ya existe, saltando creación")
        return

    import pymysql

    # Configuración de conexión (sin especificar base de datos)
    db_config = {
        'host': DB_HOST,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'charset': 'utf8mb4',
        'port': int(DB_PORT)
    }

    try:
        print(f"🔌 Conectando a MySQL en {DB_HOST}:{DB_PORT} como {DB_USER}...")

        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"✅ Base de datos '{DB_NAME}' creada o ya existe")

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"❌ Error al crear la base de datos: {e}")
        print(f"💡 En Railway, la base de datos ya debe existir")
        if not RAILWAY_ENVIRONMENT:
            raise


def test_connection():
    """
    Prueba la conexión a la base de datos
    """
    try:
        print("🔄 Probando conexión a la base de datos...")

        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1 as test, DATABASE() as db_name, USER() as user"))
            row = result.fetchone()

            print(f"✅ Conexión exitosa!")
            print(f"   Test: {row[0]}")
            print(f"   Base de datos: {row[1]}")
            print(f"   Usuario: {row[2]}")

            return True

    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        if RAILWAY_ENVIRONMENT:
            print("💡 Verificar que MySQL esté activo en Railway")
            print("💡 Verificar que las variables MYSQL* estén disponibles")
        return False


# Verificación automática al importar (solo en desarrollo)
if __name__ == "__main__":
    print("🔧 Probando configuración de base de datos...")
    test_connection()