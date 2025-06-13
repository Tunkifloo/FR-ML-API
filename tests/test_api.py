import pytest
import sys
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import shutil
from io import BytesIO
from PIL import Image
import numpy as np

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from config.database import get_db, Base
from models.database_models import Usuario, ImagenFacial

# Base de datos de prueba en memoria
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL,
                       connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override de la dependencia de base de datos para pruebas"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Cliente de prueba
client = TestClient(app)


@pytest.fixture(scope="module")
def setup_database():
    """Configurar base de datos de prueba"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_image():
    """Crear imagen de prueba"""
    # Crear imagen RGB simple
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Guardar en BytesIO
    img_buffer = BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)

    return img_buffer


@pytest.fixture
def sample_user_data():
    """Datos de usuario de prueba"""
    return {
        "nombre": "Juan",
        "apellido": "Pérez",
        "email": "juan.perez@test.com",
        "id_estudiante": "TEST001"
    }


class TestHealthEndpoints:
    """Pruebas para endpoints de salud del sistema"""

    def test_root_endpoint(self):
        """Probar endpoint raíz"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Sistema de Reconocimiento Facial" in data["message"]

    def test_health_check(self):
        """Probar health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data


class TestUsuariosAPI:
    """Pruebas para la API de usuarios"""

    def test_listar_usuarios_vacio(self, setup_database):
        """Probar listado de usuarios cuando no hay datos"""
        response = client.get("/api/v1/usuarios/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["total"] == 0

    def test_crear_usuario_sin_imagenes(self, setup_database, sample_user_data):
        """Probar creación de usuario sin imágenes (debe fallar)"""
        response = client.post(
            "/api/v1/usuarios/",
            data=sample_user_data
        )
        assert response.status_code == 422  # Validation error

    def test_crear_usuario_con_imagen(self, setup_database, sample_user_data, test_image):
        """Probar creación de usuario con imagen"""
        # Preparar archivos
        files = [("imagenes", ("test.jpg", test_image, "image/jpeg"))]

        response = client.post(
            "/api/v1/usuarios/",
            data=sample_user_data,
            files=files
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        assert data["data"]["email"] == sample_user_data["email"]

    def test_obtener_usuario_inexistente(self, setup_database):
        """Probar obtener usuario que no existe"""
        response = client.get("/api/v1/usuarios/999")
        assert response.status_code == 404

    def test_estadisticas_usuarios(self, setup_database):
        """Probar endpoint de estadísticas"""
        response = client.get("/api/v1/usuarios/estadisticas/resumen")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "data" in data


class TestReconocimientoAPI:
    """Pruebas para la API de reconocimiento"""

    def test_reconocimiento_sin_modelo_entrenado(self, setup_database, test_image):
        """Probar reconocimiento cuando el modelo no está entrenado"""
        files = [("imagen", ("test.jpg", test_image, "image/jpeg"))]

        response = client.post(
            "/api/v1/reconocimiento/identificar",
            files=files,
            data={"algoritmo": "hybrid"}
        )

        # Debería fallar porque no hay modelo entrenado
        assert response.status_code == 503

    def test_historial_reconocimientos_vacio(self, setup_database):
        """Probar historial cuando no hay reconocimientos"""
        response = client.get("/api/v1/reconocimiento/historial")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_estadisticas_reconocimientos(self, setup_database):
        """Probar estadísticas de reconocimientos"""
        response = client.get("/api/v1/reconocimiento/estadisticas")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True

    def test_info_modelo(self, setup_database):
        """Probar información del modelo"""
        response = client.get("/api/v1/reconocimiento/modelo/info")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True


class TestValidacionDatos:
    """Pruebas de validación de datos"""

    def test_usuario_email_invalido(self, setup_database):
        """Probar usuario con email inválido"""
        data = {
            "nombre": "Test",
            "apellido": "User",
            "email": "email-invalido",
            "id_estudiante": "TEST001"
        }

        response = client.post("/api/v1/usuarios/", data=data)
        assert response.status_code == 422

    def test_usuario_nombre_vacio(self, setup_database):
        """Probar usuario con nombre vacío"""
        data = {
            "nombre": "",
            "apellido": "User",
            "email": "test@example.com"
        }

        response = client.post("/api/v1/usuarios/", data=data)
        assert response.status_code == 422


class TestErrorHandling:
    """Pruebas de manejo de errores"""

    def test_endpoint_inexistente(self):
        """Probar endpoint que no existe"""
        response = client.get("/api/v1/endpoint-inexistente")
        assert response.status_code == 404

    def test_metodo_no_permitido(self):
        """Probar método HTTP no permitido"""
        response = client.delete("/")
        assert response.status_code == 405


class TestIntegrationFlow:
    """Pruebas de flujo de integración completo"""

    def test_flujo_completo_usuario(self, setup_database, test_image):
        """Probar flujo completo: crear, obtener, actualizar usuario"""

        # 1. Crear usuario
        user_data = {
            "nombre": "Integration",
            "apellido": "Test",
            "email": "integration@test.com",
            "id_estudiante": "INT001"
        }

        files = [("imagenes", ("test.jpg", test_image, "image/jpeg"))]

        response = client.post(
            "/api/v1/usuarios/",
            data=user_data,
            files=files
        )

        assert response.status_code == 200
        created_user = response.json()["data"]
        user_id = created_user["id"]

        # 2. Obtener usuario creado
        response = client.get(f"/api/v1/usuarios/{user_id}")
        assert response.status_code == 200
        retrieved_user = response.json()["data"]
        assert retrieved_user["email"] == user_data["email"]

        # 3. Actualizar usuario
        update_data = {
            "nombre": "Updated Integration"
        }

        response = client.put(
            f"/api/v1/usuarios/{user_id}",
            data=update_data
        )

        assert response.status_code == 200
        updated_user = response.json()["data"]
        assert updated_user["nombre"] == "Updated Integration"

        # 4. Listar usuarios (debe incluir el creado)
        response = client.get("/api/v1/usuarios/")
        assert response.status_code == 200
        users_list = response.json()
        assert users_list["total"] >= 1


# Ejecutar pruebas
if __name__ == "__main__":
    pytest.main([__file__, "-v"])