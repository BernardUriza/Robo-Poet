#!/usr/bin/env python3
"""
Test comprehensivo para Strategy 4: Configuración GPU Profesional
Prueba todos los componentes de optimización GPU para RTX 2000 Ada
"""

import os
import sys
import logging
import time
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gpu_availability():
    """Prueba disponibilidad y configuración básica de GPU."""
    print("🔍 PRUEBA 1: Disponibilidad de GPU")
    print("=" * 50)
    
    # Verificar GPU física
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs físicas detectadas: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            
            try:
                # Obtener detalles del dispositivo
                details = tf.config.experimental.get_device_details(gpu)
                if details:
                    print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
                    print(f"    Device Name: {details.get('device_name', 'N/A')}")
            except Exception as e:
                print(f"    No se pudieron obtener detalles: {e}")
        
        # Probar operación simple en GPU
        try:
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                result = tf.reduce_sum(c).numpy()
            print(f"✅ Operación matricial exitosa: suma = {result:.2f}")
            return True
        except Exception as e:
            print(f"❌ Error en operación GPU: {e}")
            return False
    else:
        print("❌ No se detectó GPU")
        return False


def test_mixed_precision():
    """Prueba sistema de mixed precision."""
    print("\n🔍 PRUEBA 2: Mixed Precision Training")
    print("=" * 50)
    
    try:
        from gpu_optimization.mixed_precision import MixedPrecisionManager, MixedPrecisionPolicy
        
        # Crear política personalizada
        policy = MixedPrecisionPolicy(
            name='mixed_float16',
            use_tensor_cores=True,
            optimize_matmul=True
        )
        
        # Crear manager
        manager = MixedPrecisionManager(policy)
        
        if not manager.gpu_available:
            print("❌ GPU no disponible para mixed precision")
            return False
        
        # Habilitar mixed precision
        success = manager.enable()
        print(f"Mixed precision habilitado: {success}")
        
        if success:
            # Verificar política activa
            active_policy = tf.keras.mixed_precision.global_policy()
            print(f"Política activa: {active_policy.name}")
            print(f"Compute dtype: {active_policy.compute_dtype}")
            print(f"Variable dtype: {active_policy.variable_dtype}")
            
            # Crear modelo simple para prueba
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax', dtype='float32')  # Salida en float32
            ])
            
            # Compilar con optimizador mixed precision
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            loss_scale_optimizer = manager.create_loss_scale_optimizer(optimizer)
            
            model.compile(
                optimizer=loss_scale_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Modelo compilado con loss scale optimizer")
            print(f"Initial loss scale: {policy.initial_scale}")
            
            # Obtener métricas de rendimiento
            metrics = manager.get_performance_metrics()
            print("Métricas de rendimiento:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            
            # Deshabilitar mixed precision
            manager.disable()
            print("Mixed precision deshabilitado correctamente")
            
            return True
        else:
            print("❌ No se pudo habilitar mixed precision")
            return False
            
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en mixed precision: {e}")
        return False


def test_tensor_cores():
    """Prueba optimizaciones de Tensor Cores."""
    print("\n🔍 PRUEBA 3: Tensor Cores Optimization")
    print("=" * 50)
    
    try:
        from gpu_optimization.tensor_cores import TensorCoreOptimizer, TensorCoreConfig, verify_tensor_core_support
        
        # Verificar soporte
        has_support = verify_tensor_core_support()
        print(f"Soporte de Tensor Cores: {has_support}")
        
        if not has_support:
            print("❌ Tensor Cores no disponibles")
            return False
        
        # Crear configuración
        config = TensorCoreConfig(
            dimension_alignment=8,
            use_tf32=True,
            use_fp16=True,
            optimize_matmul=True
        )
        
        # Crear optimizador
        optimizer = TensorCoreOptimizer(config)
        
        # Crear modelo no optimizado
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 128, input_length=100),
            tf.keras.layers.LSTM(250, return_sequences=True),  # No alineado
            tf.keras.layers.LSTM(250),  # No alineado
            tf.keras.layers.Dense(500, activation='relu'),  # No alineado
            tf.keras.layers.Dense(10000, activation='softmax')
        ], name="original_model")
        
        print(f"Modelo original: {model.count_params():,} parámetros")
        
        # Optimizar modelo
        optimized_model = optimizer.optimize_model(model)
        print(f"Modelo optimizado: {optimized_model.count_params():,} parámetros")
        
        # Benchmark de multiplicación matricial
        print("\nEjecutando benchmark de matmul...")
        benchmark_results = optimizer.benchmark_matmul(
            size=512,
            dtype='float16',
            iterations=50
        )
        
        if 'error' not in benchmark_results:
            print(f"  Tamaño matriz: {benchmark_results['matrix_size']}x{benchmark_results['matrix_size']}")
            print(f"  Tipo de dato: {benchmark_results['dtype']}")
            print(f"  Iteraciones: {benchmark_results['iterations']}")
            print(f"  Tiempo total: {benchmark_results['total_time']:.3f}s")
            print(f"  Tiempo por operación: {benchmark_results['time_per_op']:.3f}s")
            print(f"  Rendimiento: {benchmark_results['tflops']:.2f} TFLOPS")
            print("✅ Benchmark de Tensor Cores exitoso")
        else:
            print(f"❌ Error en benchmark: {benchmark_results['error']}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en Tensor Cores: {e}")
        return False


def test_memory_manager():
    """Prueba gestión de memoria GPU."""
    print("\n🔍 PRUEBA 4: Memory Management")
    print("=" * 50)
    
    try:
        from gpu_optimization.memory_manager import GPUMemoryManager, MemoryConfig, get_gpu_memory_info
        
        # Configuración para RTX 2000 Ada (8GB)
        config = MemoryConfig(
            total_memory_mb=8192,
            reserved_memory_mb=512,
            memory_limit_mb=7680,
            enable_growth=True,
            log_memory_usage=True
        )
        
        print(f"Configuración de memoria:")
        print(f"  Total: {config.total_memory_mb}MB")
        print(f"  Reservada: {config.reserved_memory_mb}MB")
        print(f"  Límite: {config.memory_limit_mb}MB")
        print(f"  Growth habilitado: {config.enable_growth}")
        
        # Crear manager
        manager = GPUMemoryManager(config)
        print(f"Manager configurado: {manager.is_configured}")
        
        # Obtener información de memoria
        print("\nInformación de memoria:")
        memory_info = get_gpu_memory_info()
        for key, value in memory_info.items():
            if isinstance(value, (int, float)):
                if 'mb' in key.lower():
                    print(f"  {key}: {value}MB")
                elif 'percent' in key.lower():
                    print(f"  {key}: {value:.1f}%")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        # Verificar disponibilidad de memoria
        required_memory = 2048  # 2GB
        has_memory = manager.check_memory_availability(required_memory)
        print(f"\nMemoria disponible para {required_memory}MB: {has_memory}")
        
        # Crear modelo simple para optimización
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(100,)),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(32)
        ])
        
        # Optimizar batch size
        optimal_batch = manager.optimizer.optimize_batch_size(
            model, base_batch_size=32, sequence_length=100
        )
        print(f"Batch size óptimo: {optimal_batch}")
        
        # Obtener recomendaciones de optimización
        recommendations = manager.optimizer.optimize_model_for_memory(model)
        print("\nRecomendaciones de optimización:")
        print(f"  Gradient checkpointing: {recommendations['gradient_checkpointing']}")
        print(f"  Mixed precision: {recommendations['mixed_precision']}")
        print(f"  Reducir tamaño: {recommendations['reduce_model_size']}")
        print(f"  Batch size recomendado: {recommendations['batch_size_recommendation']}")
        
        if recommendations['optimizations']:
            print("  Optimizaciones sugeridas:")
            for opt in recommendations['optimizations']:
                print(f"    - {opt}")
        
        # Obtener perfil completo
        profile = manager.get_memory_profile()
        print(f"\nPerfil de memoria:")
        print(f"  Configurado: {profile['is_configured']}")
        if 'current' in profile:
            current = profile['current']
            if 'error' not in current:
                print(f"  Memoria actual: {current.get('used_mb', 0)}MB / {current.get('total_mb', 0)}MB")
        
        print("✅ Memory management probado exitosamente")
        
        # Cleanup
        manager.cleanup()
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en memory management: {e}")
        return False


def test_adaptive_batch():
    """Prueba gestión adaptiva de batch size."""
    print("\n🔍 PRUEBA 5: Adaptive Batch Size")
    print("=" * 50)
    
    try:
        from gpu_optimization.adaptive_batch import AdaptiveBatchSizeManager, BatchSizeConfig
        
        # Configuración
        config = BatchSizeConfig(
            initial_batch_size=32,
            min_batch_size=8,
            max_batch_size=128,
            target_memory_usage=0.8,
            safety_margin=0.1
        )
        
        print(f"Configuración de batch adaptivo:")
        print(f"  Inicial: {config.initial_batch_size}")
        print(f"  Rango: {config.min_batch_size} - {config.max_batch_size}")
        print(f"  Target memoria: {config.target_memory_usage*100}%")
        print(f"  Margen seguridad: {config.safety_margin*100}%")
        
        # Crear manager
        manager = AdaptiveBatchSizeManager(config)
        
        # Crear modelo simple
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 128, input_length=50),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.LSTM(256),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10000, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy'
        )
        
        print(f"Modelo creado: {model.count_params():,} parámetros")
        
        # Crear datos sintéticos
        x_sample = tf.random.uniform([100, 50], minval=0, maxval=10000, dtype=tf.int32)
        y_sample = tf.random.uniform([100], minval=0, maxval=10000, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((x_sample, y_sample))
        
        # Buscar batch size óptimo con diferentes estrategias
        strategies = ['binary', 'exponential', 'linear']
        
        for strategy in strategies:
            print(f"\nEstrategia {strategy}:")
            try:
                optimal_batch = manager.find_optimal_batch_size(
                    model=model,
                    sample_data=dataset,
                    search_strategy=strategy,
                    max_search_time=60  # 1 minuto máximo
                )
                print(f"  Batch size óptimo: {optimal_batch}")
            except Exception as e:
                print(f"  ❌ Error en estrategia {strategy}: {e}")
        
        # Probar adaptación durante entrenamiento
        print("\nProbando adaptación durante entrenamiento...")
        
        # Crear callback de adaptación
        callback = manager.create_training_callback(
            check_interval=5,
            adaptation_threshold=0.1
        )
        
        print(f"✅ Callback creado: intervalo={callback.check_interval}")
        
        # Obtener métricas actuales
        current_metrics = manager.get_current_metrics()
        print("Métricas actuales:")
        for key, value in current_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print("✅ Adaptive batch size probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en adaptive batch: {e}")
        return False


def test_benchmark():
    """Prueba sistema de benchmark comprehensivo."""
    print("\n🔍 PRUEBA 6: Comprehensive Benchmarking")
    print("=" * 50)
    
    try:
        from gpu_optimization.benchmark import GPUBenchmark, BenchmarkConfig
        
        # Configuración de benchmark
        config = BenchmarkConfig(
            test_mixed_precision=True,
            test_tensor_cores=True,
            test_batch_sizes=[8, 16, 32],
            test_sequence_lengths=[50, 100],
            iterations_per_test=10,
            warmup_iterations=5
        )
        
        print("Configuración de benchmark:")
        print(f"  Mixed precision: {config.test_mixed_precision}")
        print(f"  Tensor cores: {config.test_tensor_cores}")
        print(f"  Batch sizes: {config.test_batch_sizes}")
        print(f"  Sequence lengths: {config.test_sequence_lengths}")
        print(f"  Iteraciones por test: {config.iterations_per_test}")
        
        # Crear benchmark
        benchmark = GPUBenchmark(config)
        
        # Verificar configuración del sistema
        print("\nVerificando configuración del sistema...")
        system_info = benchmark.get_system_info()
        
        print("Información del sistema:")
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        
        # Ejecutar benchmark rápido (versión reducida para test)
        print("\nEjecutando benchmark rápido...")
        
        # Configuración reducida para test
        quick_config = BenchmarkConfig(
            test_mixed_precision=True,
            test_tensor_cores=True,
            test_batch_sizes=[16, 32],
            test_sequence_lengths=[50],
            iterations_per_test=5,
            warmup_iterations=2
        )
        benchmark.config = quick_config
        
        try:
            results = benchmark.run_comprehensive_benchmark()
            
            print(f"\nResultados del benchmark ({len(results)} configuraciones):")
            
            best_result = None
            best_throughput = 0
            
            for i, result in enumerate(results[:3]):  # Mostrar solo primeros 3
                print(f"\nConfiguración {i+1}:")
                print(f"  Batch size: {result.batch_size}")
                print(f"  Sequence length: {result.sequence_length}")
                print(f"  Mixed precision: {result.mixed_precision}")
                print(f"  Tensor cores: {result.tensor_cores}")
                print(f"  Throughput: {result.throughput:.2f} samples/sec")
                print(f"  Memoria usada: {result.memory_used_mb:.1f}MB")
                print(f"  Tiempo por batch: {result.time_per_batch:.3f}s")
                
                if result.throughput > best_throughput:
                    best_throughput = result.throughput
                    best_result = result
            
            if best_result:
                print(f"\n🏆 Mejor configuración:")
                print(f"  Throughput: {best_result.throughput:.2f} samples/sec")
                print(f"  Batch size: {best_result.batch_size}")
                print(f"  Mixed precision: {best_result.mixed_precision}")
                print(f"  Tensor cores: {best_result.tensor_cores}")
            
            print("✅ Benchmark comprehensivo exitoso")
            return True
            
        except Exception as e:
            print(f"❌ Error ejecutando benchmark: {e}")
            # Continuar con tests básicos
            
            # Test de benchmark individual
            print("\nProbando benchmark básico...")
            result = benchmark.benchmark_configuration(
                batch_size=16,
                sequence_length=50,
                mixed_precision=True,
                tensor_cores=True
            )
            
            if result:
                print(f"  Throughput: {result.throughput:.2f} samples/sec")
                print(f"  Memoria: {result.memory_used_mb:.1f}MB")
                print("✅ Benchmark básico exitoso")
                return True
            else:
                print("❌ Benchmark básico falló")
                return False
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en benchmark: {e}")
        return False


def main():
    """Función principal de testing."""
    print("🚀 TESTING STRATEGY 4: CONFIGURACIÓN GPU PROFESIONAL")
    print("=" * 70)
    print("Sistema: RTX 2000 Ada Generation (8GB GDDR6)")
    print("TensorFlow:", tf.__version__)
    print("=" * 70)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Mixed Precision", test_mixed_precision),
        ("Tensor Cores", test_tensor_cores),
        ("Memory Manager", test_memory_manager),
        ("Adaptive Batch", test_adaptive_batch),
        ("Benchmarking", test_benchmark)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print(f"\n" + "="*70)
            success = test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLÓ")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Resumen final
    total_time = time.time() - start_time
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "="*70)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 70)
    
    for test_name, success in results.items():
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nTests exitosos: {successful_tests}/{total_tests}")
    print(f"Tiempo total: {total_time:.2f} segundos")
    
    if successful_tests == total_tests:
        print("\n🎉 TODOS LOS TESTS EXITOSOS")
        print("Strategy 4: Configuración GPU Profesional - COMPLETADA")
        return True
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests fallaron")
        print("Revisar configuración del sistema GPU")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)