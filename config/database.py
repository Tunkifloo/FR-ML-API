from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Cargar variables de entorno
load_dotenv()

# Obtener configuración individual
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '@dmin')
DB_NAME = os.getenv('DB_NAME', 'face_recognition_db')
DB_PORT = os.getenv('DB_PORT', '3306')

# Codificar la contraseña para URLs (maneja caracteres especiales como @)
encoded_password = quote_plus(DB_PASSWORD)

# Construir URL de conexión de forma segura
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"🔗 Conectando a: mysql+pymysql://{DB_USER}:****@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Crear el engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Cambié a False para menos output
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20
)

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
    from models.database_models import Base
    Base.metadata.create_all(bind=engine)
    print("✅ Base de datos inicializada correctamente")


def drop_all_tables():
    """
    Elimina todas las tablas (útil para desarrollo)
    """
    from models.database_models import Base
    Base.metadata.drop_all(bind=engine)
    print("🗑️ Todas las tablas eliminadas")


# Script para crear la base de datos si no existe
def create_database_if_not_exists():
    """
    Crea la base de datos si no existe
    """
    import pymysql

    # Configuración de conexión (sin especificar base de datos)
    db_config = {
        'host': DB_HOST,
        'user': DB_USER,
        'password': DB_PASSWORD,  # Usar la contraseña sin codificar para pymysql
        'charset': 'utf8mb4',
        'port': int(DB_PORT)
    }

    try:
        print(f"🔌 Conectando a MySQL en {DB_HOST}:{DB_PORT} como {DB_USER}...")

        # Conectar sin especificar base de datos
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()

        # Crear base de datos si no existe
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")

        print(f"✅ Base de datos '{DB_NAME}' creada o ya existe")

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"❌ Error al crear la base de datos: {e}")
        print(f"💡 Verifica que MySQL esté ejecutándose y las credenciales sean correctas")
        raise


def test_connection():
    """
    Prueba la conexión a la base de datos
    """
    try:
        with engine.connect() as connection:
            result = connection.execute("SELECT 1")
            print("✅ Conexión a la base de datos exitosa")
            return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False