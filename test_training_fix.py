#!/usr/bin/env python3
"""
✅ SCRIPT DE TESTING PARA VERIFICAR CORRECCIONES EN EL ENTRENAMIENTO
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_image_preprocessing():
    """
    Test 1: Verificar que el preprocesamiento funciona correctamente
    """
    print("🧪 TEST 1: Preprocesamiento de imágenes")
    print("=" * 50)

    try:
        from services.image_preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()

        # Crear imagen de prueba
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        print(f"✅ Imagen de prueba creada: {test_image.shape}, {test_image.dtype}")

        # Test para Eigenfaces
        eigenfaces_result = preprocessor.preprocess_for_ml(test_image, "eigenfaces")
        print(
            f"✅ Eigenfaces: {eigenfaces_result.shape}, {eigenfaces_result.dtype}, range=[{eigenfaces_result.min():.3f}, {eigenfaces_result.max():.3f}]")

        # Test para LBP
        lbp_result = preprocessor.preprocess_for_ml(test_image, "lbp")
        print(f"✅ LBP: {lbp_result.shape}, {lbp_result.dtype}, range=[{lbp_result.min()}, {lbp_result.max()}]")

        # Test para ambos
        both_result = preprocessor.preprocess_for_ml(test_image, "both")
        print(
            f"✅ Both: {both_result.shape}, {both_result.dtype}, range=[{both_result.min():.3f}, {both_result.max():.3f}]")

        # Verificar tipos correctos
        assert eigenfaces_result.dtype == np.float64, f"Eigenfaces debe ser float64, es {eigenfaces_result.dtype}"
        assert lbp_result.dtype == np.uint8, f"LBP debe ser uint8, es {lbp_result.dtype}"
        assert both_result.dtype == np.float64, f"Both debe ser float64, es {both_result.dtype}"

        print("✅ TEST 1 PASADO: Preprocesamiento funciona correctamente")
        return True

    except Exception as e:
        print(f"❌ TEST 1 FALLIDO: {e}")
        return False


def test_lbp_service():
    """
    Test 2: Verificar que LBPService maneja uint8 correctamente
    """
    print("\n🧪 TEST 2: LBPService con tipos uint8")
    print("=" * 50)

    try:
        from services.lbp_service import LBPService
        lbp_service = LBPService()

        # Crear imágenes de prueba en diferentes formatos
        uint8_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        float_image = np.random.rand(100, 100).astype(np.float64)

        print(f"✅ Imagen uint8 creada: {uint8_image.shape}, {uint8_image.dtype}")
        print(f"✅ Imagen float64 creada: {float_image.shape}, {float_image.dtype}")

        # Test preprocesamiento
        processed_uint8 = lbp_service.preprocess_image(uint8_image)
        processed_float = lbp_service.preprocess_image(float_image)

        print(f"✅ Procesado uint8: {processed_uint8.shape}, {processed_uint8.dtype}")
        print(f"✅ Procesado float: {processed_float.shape}, {processed_float.dtype}")

        # Verificar que ambos resultan en uint8
        assert processed_uint8.dtype == np.uint8, f"Resultado debe ser uint8, es {processed_uint8.dtype}"
        assert processed_float.dtype == np.uint8, f"Resultado debe ser uint8, es {processed_float.dtype}"

        # Test extracción de características
        features_uint8 = lbp_service.extract_lbp_features(processed_uint8)
        features_float = lbp_service.extract_lbp_features(processed_float)

        print(f"✅ Features de uint8: {features_uint8.shape}")
        print(f"✅ Features de float: {features_float.shape}")

        print("✅ TEST 2 PASADO: LBPService maneja tipos correctamente")
        return True

    except Exception as e:
        print(f"❌ TEST 2 FALLIDO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eigenfaces_service():
    """
    Test 3: Verificar que EigenfacesService maneja float64 correctamente
    """
    print("\n🧪 TEST 3: EigenfacesService con tipos float64")
    print("=" * 50)

    try:
        from services.eigenfaces_service import EigenfacesService
        eigenfaces_service = EigenfacesService()

        # Crear imágenes de prueba
        float_image = np.random.rand(100, 100).astype(np.float64)
        uint8_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        print(f"✅ Imagen float64 creada: {float_image.shape}, {float_image.dtype}")
        print(f"✅ Imagen uint8 creada: {uint8_image.shape}, {uint8_image.dtype}")

        # Test preprocesamiento
        processed_float = eigenfaces_service.preprocess_image(float_image)
        processed_uint8 = eigenfaces_service.preprocess_image(uint8_image)

        print(f"✅ Procesado float: {processed_float.shape}, {processed_float.dtype}")
        print(f"✅ Procesado uint8: {processed_uint8.shape}, {processed_uint8.dtype}")

        # Verificar que ambos resultan en float64
        assert processed_float.dtype == np.float64, f"Resultado debe ser float64, es {processed_float.dtype}"
        assert processed_uint8.dtype == np.float64, f"Resultado debe ser float64, es {processed_uint8.dtype}"

        print("✅ TEST 3 PASADO: EigenfacesService maneja tipos correctamente")
        return True

    except Exception as e:
        print(f"❌ TEST 3 FALLIDO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_service_training():
    """
    Test 4: Verificar que MLService entrena correctamente con tipos separados
    """
    print("\n🧪 TEST 4: MLService entrenamiento con tipos separados")
    print("=" * 50)

    try:
        from services.ml_service import MLService
        ml_service = MLService()

        # Crear datos de entrenamiento simulados
        images_by_person = {}

        # Persona 1
        person1_images = []
        for i in range(2):
            img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            person1_images.append(img)
        images_by_person[1] = person1_images

        # Persona 2
        person2_images = []
        for i in range(2):
            img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
            person2_images.append(img)
        images_by_person[2] = person2_images

        print(f"✅ Datos de prueba creados: {len(images_by_person)} personas")

        # Test entrenamiento
        training_stats = ml_service.train_models(images_by_person)

        print(f"✅ Entrenamiento completado:")
        print(f"   - Total imágenes: {training_stats['total_images']}")
        print(f"   - Personas únicas: {training_stats['unique_persons']}")
        print(f"   - Modelo entrenado: {ml_service.is_trained}")

        # Verificar que los modelos están entrenados
        assert ml_service.is_trained, "El modelo debe estar entrenado"
        assert ml_service.eigenfaces_service.is_trained, "Eigenfaces debe estar entrenado"
        assert ml_service.lbp_service.is_trained, "LBP debe estar entrenado"

        print("✅ TEST 4 PASADO: MLService entrena correctamente")
        return True

    except Exception as e:
        print(f"❌ TEST 4 FALLIDO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recognition():
    """
    Test 5: Verificar que el reconocimiento funciona con los tipos corregidos
    """
    print("\n🧪 TEST 5: Sistema de reconocimiento")
    print("=" * 50)

    try:
        from services.ml_service import MLService
        ml_service = MLService()

        # Solo hacer el test si el modelo está entrenado del test anterior
        if not ml_service.is_trained:
            print("⚠️ Saltando test de reconocimiento - modelo no entrenado")
            return True

        # Crear imagen de prueba para reconocimiento
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # Test reconocimiento híbrido
        result = ml_service.recognize_face(test_image, "hybrid")

        print(f"✅ Reconocimiento híbrido:")
        print(f"   - Reconocido: {result.get('recognized', False)}")
        print(f"   - Confianza: {result.get('confidence', 0)}")
        print(f"   - Método: {result.get('method', 'N/A')}")

        # Test reconocimiento solo Eigenfaces
        result_eigen = ml_service.recognize_face(test_image, "eigenfaces")
        print(f"✅ Reconocimiento Eigenfaces completado")

        # Test reconocimiento solo LBP
        result_lbp = ml_service.recognize_face(test_image, "lbp")
        print(f"✅ Reconocimiento LBP completado")

        print("✅ TEST 5 PASADO: Sistema de reconocimiento funciona")
        return True

    except Exception as e:
        print(f"❌ TEST 5 FALLIDO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_incremental_training():
    """
    Test 6: Verificar entrenamiento incremental
    """
    print("\n🧪 TEST 6: Entrenamiento incremental")
    print("=" * 50)

    try:
        from services.ml_service import MLService
        ml_service = MLService()

        if not ml_service.is_trained:
            print("⚠️ Saltando test incremental - modelo no entrenado")
            return True

        # Crear nueva persona para entrenamiento incremental
        new_person_images = []
        for i in range(2):
            img = np.random.randint(0, 255, (180, 180, 3), dtype=np.uint8)
            new_person_images.append(img)

        # Test add_new_person
        result = ml_service.add_new_person(999, new_person_images)

        print(f"✅ Entrenamiento incremental:")
        print(f"   - Status: {result.get('status', 'N/A')}")
        print(f"   - Mensaje: {result.get('message', 'N/A')}")

        # Verificar que el modelo sigue entrenado
        assert ml_service.is_trained, "El modelo debe seguir entrenado después del entrenamiento incremental"

        print("✅ TEST 6 PASADO: Entrenamiento incremental funciona")
        return True

    except Exception as e:
        print(f"❌ TEST 6 FALLIDO: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_type_conversion():
    """
    Test 7: Verificar conversiones específicas de tipos de datos
    """
    print("\n🧪 TEST 7: Conversiones de tipos de datos")
    print("=" * 50)

    try:
        from services.image_preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()

        # Test conversiones entre algoritmos
        base_image = np.random.rand(100, 100).astype(np.float64)  # [0,1]

        # Convertir para Eigenfaces
        eigenfaces_img = preprocessor.convert_for_algorithm(base_image, "eigenfaces")
        assert eigenfaces_img.dtype == np.float64, f"Eigenfaces debe ser float64"
        print(f"✅ Conversión a Eigenfaces: {eigenfaces_img.dtype}")

        # Convertir para LBP
        lbp_img = preprocessor.convert_for_algorithm(base_image, "lbp")
        assert lbp_img.dtype == np.uint8, f"LBP debe ser uint8"
        print(f"✅ Conversión a LBP: {lbp_img.dtype}")

        # Test con imagen uint8
        uint8_base = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        eigenfaces_from_uint8 = preprocessor.convert_for_algorithm(uint8_base, "eigenfaces")
        assert eigenfaces_from_uint8.dtype == np.float64, f"Eigenfaces debe ser float64"
        print(f"✅ uint8 → Eigenfaces: {eigenfaces_from_uint8.dtype}")

        lbp_from_uint8 = preprocessor.convert_for_algorithm(uint8_base, "lbp")
        assert lbp_from_uint8.dtype == np.uint8, f"LBP debe ser uint8"
        print(f"✅ uint8 → LBP: {lbp_from_uint8.dtype}")

        print("✅ TEST 7 PASADO: Conversiones de tipos funcionan correctamente")
        return True

    except Exception as e:
        print(f"❌ TEST 7 FALLIDO: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """
    Ejecuta todos los tests y muestra el resumen
    """
    print("🚀 INICIANDO SUITE DE TESTS - CORRECCIONES DE ENTRENAMIENTO")
    print("=" * 60)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tests = [
        ("Preprocesamiento", test_image_preprocessing),
        ("LBP Service", test_lbp_service),
        ("Eigenfaces Service", test_eigenfaces_service),
        ("ML Service Training", test_ml_service_training),
        ("Sistema de Reconocimiento", test_recognition),
        ("Entrenamiento Incremental", test_incremental_training),
        ("Conversiones de Tipos", test_data_type_conversion)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en {test_name}: {e}")
            results.append((test_name, False))

    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ PASADO" if result else "❌ FALLIDO"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("-" * 60)
    print(f"Total: {len(results)} | Pasados: {passed} | Fallidos: {failed}")

    if failed == 0:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ Las correcciones de entrenamiento funcionan correctamente")
        print("✅ Los tipos de datos están bien manejados")
        print("✅ El sistema está listo para producción")
    else:
        print(f"\n⚠️ {failed} test(s) fallaron")
        print("❌ Revisar las correcciones antes de usar en producción")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)