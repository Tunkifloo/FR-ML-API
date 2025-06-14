#!/usr/bin/env python3
"""
Script de inicialización de la base de datos
Ejecutar una vez para crear la base de datos y las tablas
"""

import sys
import os
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import create_database_if_not_exists, init_database, drop_all_tables
from models.database_models import Usuario, ImagenFacial, asignar_requisitoriado_aleatorio
from sqlalchemy.orm import sessionmaker
from config.database import engine


def crear_usuarios_ejemplo():
    """
    Crea algunos usuarios de ejemplo para probar el sistema
    """
    print("👥 Creando usuarios de ejemplo...")

    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        usuarios_ejemplo = [
            {
                "nombre": "Juan Carlos",
                "apellido": "Pérez García",
                "id_estudiante": "000000000",
                "email": "juan.perez@universidad.edu"
            },
            {
                "nombre": "María Elena",
                "apellido": "González López",
                "id_estudiante": "999999999",
                "email": "maria.gonzalez@universidad.edu"
            },
        ]

        for usuario_data in usuarios_ejemplo:
            # Verificar si el usuario ya existe
            existing = db.query(Usuario).filter(Usuario.email == usuario_data["email"]).first()
            if existing:
                print(f"⚠️ Usuario {usuario_data['email']} ya existe, saltando...")
                continue

            # Asignar estado de requisitoriado aleatoriamente
            es_requisitoriado, tipo_requisitoria = asignar_requisitoriado_aleatorio()

            nuevo_usuario = Usuario(
                nombre=usuario_data["nombre"],
                apellido=usuario_data["apellido"],
                id_estudiante=usuario_data["id_estudiante"],
                email=usuario_data["email"],
                requisitoriado=es_requisitoriado,
                tipo_requisitoria=tipo_requisitoria
            )

            db.add(nuevo_usuario)

            status = "🚨 REQUISITORIADO" if es_requisitoriado else "✅ Normal"
            requisitoria_info = f" ({tipo_requisitoria})" if tipo_requisitoria else ""

            print(f"➕ Creado: {usuario_data['nombre']} {usuario_data['apellido']} - {status}{requisitoria_info}")

        db.commit()

        # Mostrar resumen
        total_usuarios = db.query(Usuario).count()
        usuarios_requisitoriados = db.query(Usuario).filter(Usuario.requisitoriado == True).count()

        print(f"\n📊 Resumen de usuarios:")
        print(f"  • Total usuarios: {total_usuarios}")
        print(f"  • Usuarios requisitoriados: {usuarios_requisitoriados}")
        print(f"  • Usuarios normales: {total_usuarios - usuarios_requisitoriados}")

    except Exception as e:
        print(f"❌ Error al crear usuarios de ejemplo: {e}")
        db.rollback()
    finally:
        db.close()


def crear_directorios():
    """
    Crea los directorios necesarios para el sistema
    """
    print("📁 Creando directorios del sistema...")

    directorios = [
        "storage",
        "storage/images",
        "storage/temp",
        "storage/models",
        "storage/embeddings",
        "storage/logs"
    ]

    for directorio in directorios:
        if not os.path.exists(directorio):
            os.makedirs(directorio)
            print(f"✅ Creado: {directorio}")
        else:
            print(f"ℹ️ Ya existe: {directorio}")


def mostrar_instrucciones():
    """
    Muestra las instrucciones para usar el sistema
    """
    print("\n" + "=" * 60)
    print("🎉 ¡INICIALIZACIÓN COMPLETADA!")
    print("=" * 60)
    print()
    print("📋 PRÓXIMOS PASOS:")
    print()
    print("1️⃣ Iniciar el servidor:")
    print("   python main.py")
    print("   o")
    print("   uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print()
    print("2️⃣ Acceder a la documentación interactiva:")
    print("   🌐 http://localhost:8000/docs")
    print()
    print("3️⃣ Probar el sistema:")
    print("   • Subir imágenes de usuarios: POST /api/v1/usuarios/")
    print("   • Entrenar el modelo: POST /api/v1/usuarios/entrenar-modelo")
    print("   • Reconocer rostros: POST /api/v1/reconocimiento/identificar")
    print()
    print("🔧 CONFIGURACIÓN:")
    print("   • Editar .env para cambiar configuración de BD")
    print("   • Los modelos ML se entrenarán automáticamente")
    print("   • Las alertas se simularán para usuarios requisitoriados")
    print()
    print("📚 ENDPOINTS PRINCIPALES:")
    print("   • GET  /api/v1/usuarios/                 - Listar usuarios")
    print("   • POST /api/v1/usuarios/                 - Crear usuario con imágenes")
    print("   • POST /api/v1/reconocimiento/identificar - Reconocer rostro")
    print("   • GET  /api/v1/reconocimiento/historial   - Ver historial")
    print("   • GET  /health                           - Estado del sistema")
    print()
    print("⚠️ RECORDATORIO:")
    print("   • El sistema usa algoritmos ML implementados desde cero")
    print("   • Eigenfaces (PCA) + Local Binary Patterns (LBP)")
    print("   • NO se usan modelos pre-entrenados")
    print()
    print("=" * 60)


def main():
    """
    Función principal de inicialización
    """
    print("🚀 INICIALIZANDO SISTEMA DE RECONOCIMIENTO FACIAL")
    print("=" * 60)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Paso 1: Crear base de datos
        print("1️⃣ Configurando base de datos...")
        create_database_if_not_exists()

        # Paso 2: Crear tablas
        print("2️⃣ Creando tablas...")
        init_database()

        # Paso 3: Crear directorios
        print("3️⃣ Configurando directorios...")
        crear_directorios()

        # Paso 4: Crear usuarios de ejemplo
        print("4️⃣ Creando datos de ejemplo...")
        crear_usuarios_ejemplo()

        # Paso 5: Mostrar instrucciones
        mostrar_instrucciones()

    except Exception as e:
        print(f"\n❌ ERROR DURANTE LA INICIALIZACIÓN:")
        print(f"   {str(e)}")
        print()
        print("🔧 POSIBLES SOLUCIONES:")
        print("   • Verificar que MySQL esté ejecutándose")
        print("   • Revisar credenciales en el archivo .env")
        print("   • Asegurar que el usuario de BD tenga permisos")
        print("   • Instalar dependencias: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()