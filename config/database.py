from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Cargar variables de entorno
load_dotenv()

# Detectar si estamos en Railway
RAILWAY_ENVIRONMENT = os.getenv('RAILWAY_ENVIRONMENT') is not None
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Configuración de base de datos
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '@dmin')
DB_NAME = os.getenv('DB_NAME', 'face_recognition_db')
DB_PORT = os.getenv('DB_PORT', '3306')

print(f"🌍 Entorno detectado: {ENVIRONMENT}")
print(f"🚂 Railway: {'Sí' if RAILWAY_ENVIRONMENT else 'No'}")

# Codificar la contraseña para URLs (maneja caracteres especiales)
encoded_password = quote_plus(DB_PASSWORD)

# Construir URL de conexión
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"🔗 Conectando a: mysql+pymysql://{DB_USER}:****@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Configuración del engine según el entorno
if ENVIRONMENT == 'production' or RAILWAY_ENVIRONMENT:
    # Configuración optimizada para Railway
    engine_config = {
        "echo": False,
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 5,
        "max_overflow": 10,
        "connect_args": {
            "charset": "utf8mb4",
            "connect_timeout": 60,
            "read_timeout": 30,
            "write_timeout": 30,
        }
    }
    print("🚀 Configuración de PRODUCCIÓN aplicada")
else:
    # Configuración para desarrollo local
    engine_config = {
        "echo": False,
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 10,
        "max_overflow": 20
    }
    print("🔧 Configuración de DESARROLLO aplicada")

# Crear el engine
engine = create_engine(DATABASE_URL, **engine_config)

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
        from models.database_models import Base
        Base.metadata.create_all(bind=engine)
        print("✅ Base de datos inicializada correctamente")
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

    from models.database_models import Base
    Base.metadata.drop_all(bind=engine)
    print("🗑️ Todas las tablas eliminadas")


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
        with engine.connect() as connection:
            result = connection.execute("SELECT 1")
            print("✅ Conexión a la base de datos exitosa")

            # Test adicional para Railway
            if RAILWAY_ENVIRONMENT:
                connection.execute("SELECT DATABASE()")
                print("✅ Railway MySQL: Conexión verificada")

            return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        if RAILWAY_ENVIRONMENT:
            print("💡 Verificar variables de entorno en Railway Dashboard")
        return False


# Verificación automática al importar (solo en desarrollo)
if __name__ == "__main__" and ENVIRONMENT != 'production':
    test_connection()