#!/usr/bin/env python3
"""
Script para importar datos desde archivo JSON al sistema de reconocimiento facial
Procesa: Nombre, Apellidos, Correo, ID_Estudiante y Foto en base64
"""

import json
import base64
import os
import sys
import uuid
import cv2
import numpy as np
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Importar modelos del sistema
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.database import get_db, engine, SessionLocal
from models.database_models import Usuario, ImagenFacial, asignar_requisitoriado_aleatorio
from services.ml_service import MLService


class JSONImporter:
    """
    Importador de datos desde archivo JSON
    """

    def __init__(self, json_file_path: str):
        """
        Inicializa el importador

        Args:
            json_file_path: Ruta del archivo JSON
        """
        self.json_file_path = json_file_path
        self.db = SessionLocal()
        self.ml_service = MLService()

        # Directorios de almacenamiento
        self.images_dir = "storage/images"
        self.temp_dir = "storage/temp"

        # Crear directorios si no existen
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Estadísticas de importación
        self.stats = {
            "total_records": 0,
            "successful_imports": 0,
            "failed_imports": 0,
            "existing_users": 0,
            "images_processed": 0,
            "errors": []
        }

    def load_json_data(self) -> list:
        """
        Carga y valida los datos del archivo JSON
        """
        try:
            print(f"📂 Cargando archivo JSON: {self.json_file_path}")

            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Determinar estructura del JSON
            if isinstance(data, dict):
                # Si es un diccionario, buscar la clave 'personas' o similar
                if 'personas' in data:
                    records = data['personas']
                elif 'users' in data:
                    records = data['users']
                elif 'data' in data:
                    records = data['data']
                else:
                    # Asumir que las claves son los registros
                    records = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
            elif isinstance(data, list):
                records = data
            else:
                raise ValueError("Formato de JSON no compatible")

            self.stats["total_records"] = len(records)
            print(f"✅ Datos cargados: {len(records)} registros encontrados")

            return records

        except FileNotFoundError:
            print(f"❌ Error: Archivo no encontrado: {self.json_file_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error al parsear JSON: {str(e)}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error inesperado al cargar JSON: {str(e)}")
            sys.exit(1)

    def validate_record(self, record: dict, index: int) -> dict:
        """
        Valida y normaliza un registro individual
        """
        try:
            # Mapear campos posibles (diferentes variaciones de nombres)
            field_mappings = {
                'nombre': ['nombre', 'name', 'first_name', 'nombres'],
                'apellidos': ['apellidos', 'apellido', 'last_name', 'surname', 'apellidos'],
                'correo': ['correo', 'email', 'mail', 'email_address'],
                'id_estudiante': ['id_estudiante', 'student_id', 'codigo', 'code', 'id'],
                'foto': ['foto', 'image', 'photo', 'foto_base64', 'image_base64']
            }

            validated_record = {}

            # Extraer y validar cada campo
            for field, possible_keys in field_mappings.items():
                value = None
                for key in possible_keys:
                    if key in record:
                        value = record[key]
                        break

                if field in ['nombre', 'apellidos', 'correo'] and not value:
                    raise ValueError(f"Campo requerido '{field}' no encontrado o vacío")

                validated_record[field] = value

            # Validaciones específicas
            if not validated_record['nombre'] or not validated_record['apellidos']:
                raise ValueError("Nombre y apellidos son requeridos")

            if not validated_record['correo'] or '@' not in validated_record['correo']:
                raise ValueError("Email válido es requerido")

            if not validated_record['foto']:
                raise ValueError("Foto en base64 es requerida")

            # Limpiar y normalizar datos
            validated_record['nombre'] = validated_record['nombre'].strip().title()
            validated_record['apellidos'] = validated_record['apellidos'].strip().title()
            validated_record['correo'] = validated_record['correo'].strip().lower()

            # Validar ID estudiante si existe
            if validated_record['id_estudiante']:
                validated_record['id_estudiante'] = str(validated_record['id_estudiante']).strip()

            return validated_record

        except Exception as e:
            error_msg = f"Error validando registro {index + 1}: {str(e)}"
            self.stats["errors"].append(error_msg)
            print(f"⚠️ {error_msg}")
            return None

    def decode_base64_image(self, base64_string: str, user_id: int) -> str:
        """
        Decodifica imagen base64 y la guarda como archivo
        """
        try:
            # Limpiar string base64
            if base64_string.startswith('data:image/'):
                base64_string = base64_string.split(',')[1]

            # Decodificar base64
            image_data = base64.b64decode(base64_string)

            # Crear nombre único para el archivo
            filename = f"imported_user_{user_id}_{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(self.images_dir, filename)

            # Guardar imagen
            with open(filepath, 'wb') as f:
                f.write(image_data)

            # Validar que se puede leer como imagen
            img = cv2.imread(filepath)
            if img is None:
                os.remove(filepath)
                raise ValueError("No se pudo leer la imagen decodificada")

            # Obtener dimensiones
            height, width = img.shape[:2]

            return filepath, width, height, len(image_data)

        except Exception as e:
            raise ValueError(f"Error decodificando imagen base64: {str(e)}")

    def import_single_record(self, record: dict, index: int) -> bool:
        """
        Importa un registro individual
        """
        try:
            # Validar registro
            validated_record = self.validate_record(record, index)
            if not validated_record:
                return False

            print(f"📝 Importando registro {index + 1}: {validated_record['nombre']} {validated_record['apellidos']}")

            # Verificar si el usuario ya existe (por email)
            existing_user = self.db.query(Usuario).filter(
                Usuario.email == validated_record['correo']
            ).first()

            if existing_user:
                print(f"⚠️ Usuario ya existe con email: {validated_record['correo']}")
                self.stats["existing_users"] += 1
                return False

            # Verificar ID estudiante si existe
            if validated_record['id_estudiante']:
                existing_student = self.db.query(Usuario).filter(
                    Usuario.id_estudiante == validated_record['id_estudiante']
                ).first()

                if existing_student:
                    print(f"⚠️ ID estudiante ya existe: {validated_record['id_estudiante']}")
                    self.stats["existing_users"] += 1
                    return False

            # Asignar estado de requisitoriado aleatoriamente
            es_requisitoriado, tipo_requisitoria = asignar_requisitoriado_aleatorio()

            # Crear usuario
            nuevo_usuario = Usuario(
                nombre=validated_record['nombre'],
                apellido=validated_record['apellidos'],
                email=validated_record['correo'],
                id_estudiante=validated_record['id_estudiante'],
                requisitoriado=es_requisitoriado,
                tipo_requisitoria=tipo_requisitoria
            )

            self.db.add(nuevo_usuario)
            self.db.commit()
            self.db.refresh(nuevo_usuario)

            # Procesar imagen
            filepath, width, height, size_bytes = self.decode_base64_image(
                validated_record['foto'],
                nuevo_usuario.id
            )

            # Crear registro de imagen
            imagen_facial = ImagenFacial(
                usuario_id=nuevo_usuario.id,
                nombre_archivo=f"imported_image_{nuevo_usuario.id}.jpg",
                ruta_archivo=filepath,
                es_principal=True,  # Primera imagen como principal
                formato="jpg",
                ancho=width,
                alto=height,
                tamano_bytes=size_bytes
            )

            self.db.add(imagen_facial)
            self.db.commit()

            # Entrenar modelo ML con la nueva imagen
            try:
                img = cv2.imread(filepath)
                if img is not None:
                    self.ml_service.add_new_person(nuevo_usuario.id, [img])
                    print(f"🤖 Modelo ML actualizado para usuario {nuevo_usuario.id}")
            except Exception as e:
                print(f"⚠️ Error actualizando modelo ML: {str(e)}")

            # Mostrar información del usuario creado
            status = "🚨 REQUISITORIADO" if es_requisitoriado else "✅ Normal"
            requisitoria_info = f" ({tipo_requisitoria})" if tipo_requisitoria else ""

            print(f"✅ Usuario creado: ID={nuevo_usuario.id}, {status}{requisitoria_info}")

            self.stats["successful_imports"] += 1
            self.stats["images_processed"] += 1

            return True

        except Exception as e:
            self.db.rollback()
            error_msg = f"Error importando registro {index + 1}: {str(e)}"
            self.stats["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            self.stats["failed_imports"] += 1
            return False

    def run_import(self):
        """
        Ejecuta el proceso completo de importación
        """
        print("🚀 Iniciando importación de datos...")
        print("=" * 60)

        start_time = datetime.now()

        try:
            # Cargar datos
            records = self.load_json_data()

            # Procesar cada registro
            for index, record in enumerate(records):
                self.import_single_record(record, index)

                # Mostrar progreso cada 10 registros
                if (index + 1) % 10 == 0:
                    print(f"📊 Progreso: {index + 1}/{len(records)} registros procesados")

            # Guardar modelos ML
            try:
                if self.ml_service.is_trained:
                    self.ml_service.eigenfaces_service.save_model()
                    self.ml_service.lbp_service.save_model()
                    print("💾 Modelos ML guardados")
            except Exception as e:
                print(f"⚠️ Error guardando modelos ML: {str(e)}")

        except Exception as e:
            print(f"❌ Error durante la importación: {str(e)}")

        finally:
            self.db.close()

        # Mostrar estadísticas finales
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("📊 RESUMEN DE IMPORTACIÓN")
        print("=" * 60)
        print(f"⏱️ Tiempo total: {duration:.2f} segundos")
        print(f"📋 Total de registros: {self.stats['total_records']}")
        print(f"✅ Importaciones exitosas: {self.stats['successful_imports']}")
        print(f"❌ Importaciones fallidas: {self.stats['failed_imports']}")
        print(f"⚠️ Usuarios existentes: {self.stats['existing_users']}")
        print(f"🖼️ Imágenes procesadas: {self.stats['images_processed']}")
        print(f"📈 Tasa de éxito: {(self.stats['successful_imports'] / self.stats['total_records'] * 100):.1f}%")

        if self.stats['errors']:
            print(f"\n⚠️ ERRORES ENCONTRADOS ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:5]:  # Mostrar solo los primeros 5
                print(f"  • {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... y {len(self.stats['errors']) - 5} errores más")

        print("\n🎉 Importación completada!")
        print("💡 Puedes probar el sistema con: python main.py")


def main():
    """
    Función principal
    """
    # Configuración
    json_file = "personas_produccion.json"  # Ajusta el nombre de tu archivo

    # Verificar que el archivo existe
    if not os.path.exists(json_file):
        print(f"❌ Error: Archivo no encontrado: {json_file}")
        print("💡 Asegúrate de que el archivo JSON esté en el directorio raíz del proyecto")
        sys.exit(1)

    # Ejecutar importación
    importer = JSONImporter(json_file)
    importer.run_import()


if __name__ == "__main__":
    main()