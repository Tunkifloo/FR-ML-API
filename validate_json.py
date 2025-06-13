#!/usr/bin/env python3
"""
Script para validar archivo JSON antes de la importación
Verifica estructura, campos requeridos y calidad de datos
"""

import json
import base64
import os
import sys
from typing import Dict, List, Any


class JSONValidator:
    """
    Validador de archivo JSON para importación
    """

    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.validation_results = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "warnings": [],
            "errors": [],
            "field_analysis": {},
            "sample_records": []
        }

    def load_and_analyze_json(self) -> List[Dict]:
        """
        Carga y analiza la estructura del JSON
        """
        print(f"🔍 Analizando archivo: {self.json_file_path}")

        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Determinar estructura
            if isinstance(data, dict):
                print("📋 Estructura detectada: Diccionario")

                # Mostrar claves principales
                print(f"🔑 Claves principales: {list(data.keys())}")

                # Buscar registros de personas
                if 'personas' in data:
                    records = data['personas']
                    print("✅ Encontrada clave 'personas'")
                elif 'users' in data:
                    records = data['users']
                    print("✅ Encontrada clave 'users'")
                elif 'data' in data:
                    records = data['data']
                    print("✅ Encontrada clave 'data'")
                else:
                    # Buscar la clave que contenga arrays de objetos
                    records = None
                    for key, value in data.items():
                        if isinstance(value, list) and value and isinstance(value[0], dict):
                            records = value
                            print(f"✅ Encontrados registros en clave '{key}'")
                            break

                    if not records:
                        print("⚠️ No se encontraron registros de personas en el JSON")
                        return []

            elif isinstance(data, list):
                records = data
                print("📋 Estructura detectada: Array de objetos")
            else:
                print("❌ Estructura no compatible")
                return []

            self.validation_results["total_records"] = len(records)
            print(f"📊 Total de registros encontrados: {len(records)}")

            return records

        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {self.json_file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error de formato JSON: {str(e)}")
            return []
        except Exception as e:
            print(f"❌ Error inesperado: {str(e)}")
            return []

    def analyze_field_patterns(self, records: List[Dict]):
        """
        Analiza patrones de campos en los registros
        """
        print("\n🔍 ANÁLISIS DE CAMPOS")
        print("-" * 40)

        field_counts = {}
        field_samples = {}

        # Analizar primeros 10 registros para obtener patrones
        sample_size = min(10, len(records))

        for i, record in enumerate(records[:sample_size]):
            for field, value in record.items():
                if field not in field_counts:
                    field_counts[field] = 0
                    field_samples[field] = []

                field_counts[field] += 1

                # Guardar muestra del valor
                if len(field_samples[field]) < 3:
                    sample_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    field_samples[field].append(sample_value)

        # Mostrar análisis
        print(f"📋 Campos encontrados (en los primeros {sample_size} registros):")
        for field, count in sorted(field_counts.items()):
            percentage = (count / sample_size) * 100
            print(f"  • {field}: {count}/{sample_size} ({percentage:.0f}%)")
            print(f"    Ejemplos: {', '.join(field_samples[field])}")

        self.validation_results["field_analysis"] = {
            "field_counts": field_counts,
            "field_samples": field_samples,
            "sample_size": sample_size
        }

    def validate_record(self, record: Dict, index: int) -> Dict[str, Any]:
        """
        Valida un registro individual
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "fields_found": {}
        }

        # Campos requeridos y sus posibles variaciones
        required_fields = {
            'nombre': ['nombre', 'name', 'first_name', 'nombres'],
            'apellidos': ['apellidos', 'apellido', 'last_name', 'surname'],
            'correo': ['correo', 'email', 'mail', 'email_address'],
            'foto': ['foto', 'image', 'photo', 'foto_base64', 'image_base64']
        }

        optional_fields = {
            'id_estudiante': ['id_estudiante', 'student_id', 'codigo', 'code', 'id']
        }

        # Validar campos requeridos
        for field_name, possible_keys in required_fields.items():
            found_value = None
            found_key = None

            for key in possible_keys:
                if key in record and record[key]:
                    found_value = record[key]
                    found_key = key
                    break

            if found_value:
                validation["fields_found"][field_name] = {
                    "key": found_key,
                    "value_preview": str(found_value)[:30] + "..." if len(str(found_value)) > 30 else str(found_value)
                }

                # Validaciones específicas
                if field_name == 'correo' and '@' not in str(found_value):
                    validation["errors"].append(f"Email inválido: {found_value}")
                    validation["valid"] = False

                elif field_name == 'foto':
                    # Validar que parece ser base64
                    foto_str = str(found_value)
                    if len(foto_str) < 100:
                        validation["errors"].append("Foto muy pequeña, posiblemente no sea base64 válido")
                        validation["valid"] = False
                    elif not (foto_str.startswith('/9j/') or foto_str.startswith('iVBOR') or foto_str.startswith(
                            'data:image/')):
                        validation["warnings"].append("Foto podría no ser formato base64 válido")

            else:
                validation["errors"].append(f"Campo requerido '{field_name}' no encontrado")
                validation["valid"] = False

        # Validar campos opcionales
        for field_name, possible_keys in optional_fields.items():
            for key in possible_keys:
                if key in record and record[key]:
                    validation["fields_found"][field_name] = {
                        "key": key,
                        "value_preview": str(record[key])
                    }
                    break

        return validation

    def run_validation(self):
        """
        Ejecuta validación completa
        """
        print("🚀 INICIANDO VALIDACIÓN DE ARCHIVO JSON")
        print("=" * 60)

        # Cargar y analizar estructura
        records = self.load_and_analyze_json()

        if not records:
            print("❌ No se pudieron cargar registros para validar")
            return

        # Analizar patrones de campos
        self.analyze_field_patterns(records)

        # Validar registros individuales
        print(f"\n🔍 VALIDANDO {len(records)} REGISTROS")
        print("-" * 40)

        valid_count = 0

        for i, record in enumerate(records):
            validation = self.validate_record(record, i)

            if validation["valid"]:
                valid_count += 1
                if i < 3:  # Mostrar detalles de los primeros 3 registros válidos
                    print(f"✅ Registro {i + 1}: VÁLIDO")
                    for field, info in validation["fields_found"].items():
                        print(f"   {field}: {info['value_preview']}")
            else:
                print(f"❌ Registro {i + 1}: INVÁLIDO")
                for error in validation["errors"]:
                    print(f"   Error: {error}")
                    self.validation_results["errors"].append(f"Registro {i + 1}: {error}")

            # Añadir warnings
            for warning in validation["warnings"]:
                print(f"⚠️ Registro {i + 1}: {warning}")
                self.validation_results["warnings"].append(f"Registro {i + 1}: {warning}")

            # Guardar muestra de registros válidos
            if validation["valid"] and len(self.validation_results["sample_records"]) < 3:
                self.validation_results["sample_records"].append({
                    "index": i + 1,
                    "fields": validation["fields_found"]
                })

        self.validation_results["valid_records"] = valid_count
        self.validation_results["invalid_records"] = len(records) - valid_count

        # Mostrar resumen final
        self.print_final_summary()

    def print_final_summary(self):
        """
        Muestra resumen final de la validación
        """
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE VALIDACIÓN")
        print("=" * 60)

        results = self.validation_results

        print(f"📋 Total de registros: {results['total_records']}")
        print(f"✅ Registros válidos: {results['valid_records']}")
        print(f"❌ Registros inválidos: {results['invalid_records']}")

        if results['total_records'] > 0:
            success_rate = (results['valid_records'] / results['total_records']) * 100
            print(f"📈 Tasa de éxito: {success_rate:.1f}%")

        # Mostrar errores más comunes
        if results['errors']:
            print(f"\n❌ ERRORES ENCONTRADOS ({len(results['errors'])}):")
            for error in results['errors'][:5]:
                print(f"  • {error}")
            if len(results['errors']) > 5:
                print(f"  ... y {len(results['errors']) - 5} errores más")

        # Mostrar warnings
        if results['warnings']:
            print(f"\n⚠️ ADVERTENCIAS ({len(results['warnings'])}):")
            for warning in results['warnings'][:3]:
                print(f"  • {warning}")
            if len(results['warnings']) > 3:
                print(f"  ... y {len(results['warnings']) - 3} advertencias más")

        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")

        if results['valid_records'] == results['total_records']:
            print("  ✅ El archivo está listo para importar")
            print("  🚀 Ejecuta: python import_json_data.py")
        elif results['valid_records'] > 0:
            print(f"  ⚠️ Solo {results['valid_records']} de {results['total_records']} registros son válidos")
            print("  🔧 Revisa y corrige los errores antes de importar")
            print("  📝 Los registros válidos se importarán correctamente")
        else:
            print("  ❌ Ningún registro es válido para importar")
            print("  🔧 Revisa la estructura del archivo JSON")

        print(f"\n📄 Estructura esperada para cada registro:")
        print("  {")
        print('    "nombre": "Juan",')
        print('    "apellidos": "Pérez García",')
        print('    "correo": "juan@ejemplo.com",')
        print('    "id_estudiante": "000243425",  // Opcional')
        print('    "foto": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAY..."  // Base64')
        print("  }")


def main():
    """
    Función principal
    """
    # Configuración
    json_file = "personas_produccion.json"  # Cambia por el nombre de tu archivo

    if len(sys.argv) > 1:
        json_file = sys.argv[1]

    print(f"🔍 Validando archivo: {json_file}")

    if not os.path.exists(json_file):
        print(f"❌ Archivo no encontrado: {json_file}")
        print("💡 Uso: python validate_json.py [nombre_archivo.json]")
        sys.exit(1)

    # Ejecutar validación
    validator = JSONValidator(json_file)
    validator.run_validation()


if __name__ == "__main__":
    main()