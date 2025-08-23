"""
Sistema de benchmark comprehensivo para RTX 2000 Ada.

Implementa benchmarks detallados para evaluar configuraciones GPU,
mixed precision, Tensor Cores y optimizaciones de memoria.
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import tensorflow as tf

from .mixed_precision import MixedPrecisionManager, MixedPrecisionPolicy
from .tensor_cores import TensorCoreOptimizer
from .memory_manager import GPUMemoryManager, MemoryConfig
from .adaptive_batch import AdaptiveBatchSizeManager, BatchSizeConfig


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuración para benchmarks."""
    
    # Configuraciones a probar
    test_mixed_precision: bool = True
    test_tensor_cores: bool = True
    test_batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    test_sequence_lengths: List[int] = field(default_factory=lambda: [50, 100, 200])
    
    # Duración de tests
    benchmark_duration: int = 60  # segundos por test
    warmup_duration: int = 10  # segundos de warmup
    min_iterations: int = 10
    max_iterations: int = 100
    
    # Configuraciones de modelo a probar
    model_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'lstm_units': 256, 'layers': 2, 'vocab_size': 5000},
        {'lstm_units': 512, 'layers': 2, 'vocab_size': 5000},
        {'lstm_units': 256, 'layers': 3, 'vocab_size': 5000}
    ])
    
    # Output
    save_results: bool = True
    results_dir: str = "./benchmark_results"
    include_detailed_logs: bool = True


@dataclass
class BenchmarkResults:
    """Resultados de benchmark."""
    
    # Identificación
    config_name: str
    timestamp: str
    gpu_info: Dict[str, Any]
    
    # Configuración probada
    mixed_precision_enabled: bool
    tensor_cores_enabled: bool
    batch_size: int
    sequence_length: int
    model_config: Dict[str, Any]
    
    # Métricas de rendimiento
    throughput_samples_per_sec: float
    time_per_batch_ms: float
    memory_usage_mb: int
    memory_usage_percent: float
    gpu_utilization_percent: float
    
    # Métricas de calidad
    final_loss: float
    convergence_rate: float
    training_stability: float
    
    # Recursos
    peak_memory_mb: int
    average_memory_mb: int
    power_consumption_w: float = 0.0
    temperature_c: float = 0.0
    
    # Errores
    oom_occurred: bool = False
    errors: List[str] = field(default_factory=list)
    
    # Score combinado
    performance_score: float = field(init=False)
    efficiency_score: float = field(init=False)
    
    def __post_init__(self):
        """Calcula scores derivados."""
        if not self.oom_occurred and self.throughput_samples_per_sec > 0:
            # Performance score: throughput normalizado
            self.performance_score = min(self.throughput_samples_per_sec / 1000, 1.0)
            
            # Efficiency score: throughput / memoria usada
            memory_factor = max(self.memory_usage_mb, 1)
            self.efficiency_score = (self.throughput_samples_per_sec / memory_factor) * 1000
        else:
            self.performance_score = 0.0
            self.efficiency_score = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            'config_name': self.config_name,
            'timestamp': self.timestamp,
            'gpu_info': self.gpu_info,
            'mixed_precision_enabled': self.mixed_precision_enabled,
            'tensor_cores_enabled': self.tensor_cores_enabled,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'model_config': self.model_config,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'time_per_batch_ms': self.time_per_batch_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'memory_usage_percent': self.memory_usage_percent,
            'gpu_utilization_percent': self.gpu_utilization_percent,
            'final_loss': self.final_loss,
            'convergence_rate': self.convergence_rate,
            'training_stability': self.training_stability,
            'peak_memory_mb': self.peak_memory_mb,
            'average_memory_mb': self.average_memory_mb,
            'power_consumption_w': self.power_consumption_w,
            'temperature_c': self.temperature_c,
            'oom_occurred': self.oom_occurred,
            'errors': self.errors,
            'performance_score': self.performance_score,
            'efficiency_score': self.efficiency_score
        }


class GPUBenchmark:
    """Benchmark principal para GPU."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResults] = []
        
        # Managers
        self.mixed_precision_manager = None
        self.tensor_core_optimizer = None
        self.memory_manager = None
        
        # GPU Info
        self.gpu_info = self._get_gpu_info()
        
        # Preparar directorio de resultados
        if self.config.save_results:
            Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Obtiene información detallada de la GPU."""
        info = {
            'gpu_available': False,
            'gpu_name': 'Unknown',
            'compute_capability': '0.0',
            'memory_total_mb': 0,
            'driver_version': 'Unknown',
            'cuda_version': 'Unknown'
        }
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info['gpu_available'] = True
                
                # Detalles del dispositivo
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                info.update({
                    'gpu_name': gpu_details.get('device_name', 'Unknown'),
                    'compute_capability': f"{gpu_details.get('compute_capability', (0, 0))[0]}.{gpu_details.get('compute_capability', (0, 0))[1]}"
                })
                
                # Información de memoria con pynvml si está disponible
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    info['memory_total_mb'] = mem_info.total // (1024 * 1024)
                    
                    # Versiones
                    info['driver_version'] = pynvml.nvmlSystemGetDriverVersion().decode()
                    
                except ImportError:
                    info['memory_total_mb'] = 8192  # RTX 2000 Ada default
                    
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
        
        return info
    
    def run_comprehensive_benchmark(self) -> List[BenchmarkResults]:
        """Ejecuta benchmark comprehensivo."""
        logger.info("Starting comprehensive GPU benchmark...")
        logger.info(f"GPU: {self.gpu_info['gpu_name']}")
        logger.info(f"Memory: {self.gpu_info['memory_total_mb']}MB")
        
        total_configs = (
            len(self.config.model_configs) * 
            len(self.config.test_batch_sizes) * 
            len(self.config.test_sequence_lengths) * 
            (2 if self.config.test_mixed_precision else 1)
        )
        
        logger.info(f"Testing {total_configs} configurations...")
        
        config_count = 0
        
        for model_config in self.config.model_configs:
            for sequence_length in self.config.test_sequence_lengths:
                for batch_size in self.config.test_batch_sizes:
                    # Test sin mixed precision
                    config_count += 1
                    logger.info(f"Configuration {config_count}/{total_configs}: Standard Precision")
                    
                    result = self._benchmark_configuration(
                        model_config=model_config,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        use_mixed_precision=False
                    )
                    
                    if result:
                        self.results.append(result)
                    
                    # Test con mixed precision si está habilitado
                    if self.config.test_mixed_precision:
                        config_count += 1
                        logger.info(f"Configuration {config_count}/{total_configs}: Mixed Precision")
                        
                        result = self._benchmark_configuration(
                            model_config=model_config,
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            use_mixed_precision=True
                        )
                        
                        if result:
                            self.results.append(result)
        
        # Guardar resultados
        if self.config.save_results:
            self._save_results()
        
        # Generar reporte
        self._generate_report()
        
        logger.info(f"Benchmark completed: {len(self.results)} configurations tested")
        
        return self.results
    
    def _benchmark_configuration(
        self,
        model_config: Dict[str, Any],
        batch_size: int,
        sequence_length: int,
        use_mixed_precision: bool
    ) -> Optional[BenchmarkResults]:
        """Benchmarks una configuración específica."""
        
        config_name = f"lstm{model_config['lstm_units']}_l{model_config['layers']}_bs{batch_size}_seq{sequence_length}"
        if use_mixed_precision:
            config_name += "_mp"
        
        logger.info(f"Benchmarking: {config_name}")
        
        try:
            # Limpiar sesión anterior
            tf.keras.backend.clear_session()
            
            # Configurar mixed precision si es necesario
            if use_mixed_precision:
                if not self.mixed_precision_manager:
                    self.mixed_precision_manager = MixedPrecisionManager()
                self.mixed_precision_manager.enable()
            else:
                if self.mixed_precision_manager:
                    self.mixed_precision_manager.disable()
            
            # Crear modelo de prueba
            model = self._create_test_model(model_config, sequence_length)
            
            # Optimizar para Tensor Cores si está habilitado
            tensor_cores_enabled = False
            if self.config.test_tensor_cores:
                if not self.tensor_core_optimizer:
                    self.tensor_core_optimizer = TensorCoreOptimizer()
                
                if self.tensor_core_optimizer.tensor_core_available:
                    model = self.tensor_core_optimizer.optimize_model(model)
                    tensor_cores_enabled = True
            
            # Crear datos de prueba
            test_data = self._create_test_data(batch_size, sequence_length, model_config['vocab_size'])
            
            # Configurar optimizador
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            if use_mixed_precision and self.mixed_precision_manager:
                optimizer = self.mixed_precision_manager.create_loss_scale_optimizer(optimizer)
            
            # Compilar modelo
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Ejecutar benchmark
            benchmark_result = self._run_training_benchmark(
                model=model,
                test_data=test_data,
                config_name=config_name,
                batch_size=batch_size,
                sequence_length=sequence_length,
                model_config=model_config,
                use_mixed_precision=use_mixed_precision,
                tensor_cores_enabled=tensor_cores_enabled
            )
            
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Benchmark failed for {config_name}: {e}")
            
            # Crear resultado de error
            return BenchmarkResults(
                config_name=config_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                gpu_info=self.gpu_info,
                mixed_precision_enabled=use_mixed_precision,
                tensor_cores_enabled=False,
                batch_size=batch_size,
                sequence_length=sequence_length,
                model_config=model_config,
                throughput_samples_per_sec=0.0,
                time_per_batch_ms=0.0,
                memory_usage_mb=0,
                memory_usage_percent=0.0,
                gpu_utilization_percent=0.0,
                final_loss=float('inf'),
                convergence_rate=0.0,
                training_stability=0.0,
                peak_memory_mb=0,
                average_memory_mb=0,
                oom_occurred=True,
                errors=[str(e)]
            )
    
    def _create_test_model(self, config: Dict[str, Any], sequence_length: int) -> tf.keras.Model:
        """Crea modelo de prueba."""
        
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=config['vocab_size'],
                output_dim=128,
                input_length=sequence_length
            ),
            
            *[tf.keras.layers.LSTM(
                config['lstm_units'],
                return_sequences=(i < config['layers'] - 1),
                dropout=0.2 if i == 0 else 0.0  # Solo dropout en primera capa para CuDNN
            ) for i in range(config['layers'])],
            
            tf.keras.layers.Dense(config['vocab_size'], activation='softmax')
        ])
        
        return model
    
    def _create_test_data(self, batch_size: int, sequence_length: int, vocab_size: int) -> tf.data.Dataset:
        """Crea dataset de prueba."""
        
        # Generar datos sintéticos
        num_samples = batch_size * 20  # 20 batches de datos
        
        x_data = np.random.randint(0, vocab_size, size=(num_samples, sequence_length))
        y_data = np.random.randint(0, vocab_size, size=(num_samples,))
        
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _run_training_benchmark(
        self,
        model: tf.keras.Model,
        test_data: tf.data.Dataset,
        config_name: str,
        batch_size: int,
        sequence_length: int,
        model_config: Dict[str, Any],
        use_mixed_precision: bool,
        tensor_cores_enabled: bool
    ) -> BenchmarkResults:
        """Ejecuta benchmark de entrenamiento."""
        
        # Métricas de monitoreo
        batch_times = []
        memory_usage = []
        losses = []
        
        # Obtener memoria inicial
        initial_memory = self._get_memory_usage()
        peak_memory = initial_memory
        
        # Warmup
        logger.debug("Warmup phase...")
        warmup_batches = max(3, self.config.warmup_duration // 5)
        
        batch_iter = iter(test_data.repeat())
        
        for _ in range(warmup_batches):
            try:
                batch_x, batch_y = next(batch_iter)
                _ = model.train_on_batch(batch_x, batch_y)
            except tf.errors.ResourceExhaustedError:
                # OOM durante warmup
                return self._create_oom_result(config_name, batch_size, sequence_length, 
                                             model_config, use_mixed_precision, tensor_cores_enabled)
        
        # Benchmark real
        logger.debug("Benchmark phase...")
        start_time = time.time()
        iterations = 0
        
        while (time.time() - start_time) < self.config.benchmark_duration and iterations < self.config.max_iterations:
            try:
                batch_x, batch_y = next(batch_iter)
                
                batch_start = time.time()
                loss = model.train_on_batch(batch_x, batch_y)
                batch_end = time.time()
                
                # Registrar métricas
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)
                
                if isinstance(loss, (list, tuple)):
                    losses.append(loss[0])  # Solo la loss principal
                else:
                    losses.append(loss)
                
                # Monitorear memoria
                current_memory = self._get_memory_usage()
                memory_usage.append(current_memory)
                peak_memory = max(peak_memory, current_memory)
                
                iterations += 1
                
                if iterations >= self.config.min_iterations and (time.time() - start_time) >= self.config.benchmark_duration:
                    break
                    
            except tf.errors.ResourceExhaustedError:
                # OOM durante benchmark
                logger.warning(f"OOM occurred after {iterations} iterations")
                if iterations < self.config.min_iterations:
                    return self._create_oom_result(config_name, batch_size, sequence_length,
                                                 model_config, use_mixed_precision, tensor_cores_enabled)
                break
            except Exception as e:
                logger.error(f"Training error: {e}")
                break
        
        # Calcular métricas finales
        if not batch_times:
            return self._create_oom_result(config_name, batch_size, sequence_length,
                                         model_config, use_mixed_precision, tensor_cores_enabled)
        
        avg_batch_time = np.mean(batch_times)
        throughput = batch_size / avg_batch_time
        avg_memory = np.mean(memory_usage) if memory_usage else initial_memory
        
        # Métricas de calidad
        final_loss = losses[-1] if losses else float('inf')
        convergence_rate = self._calculate_convergence_rate(losses)
        training_stability = self._calculate_training_stability(losses)
        
        return BenchmarkResults(
            config_name=config_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            gpu_info=self.gpu_info,
            mixed_precision_enabled=use_mixed_precision,
            tensor_cores_enabled=tensor_cores_enabled,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_config=model_config,
            throughput_samples_per_sec=throughput,
            time_per_batch_ms=avg_batch_time * 1000,
            memory_usage_mb=int(avg_memory),
            memory_usage_percent=(avg_memory / self.gpu_info['memory_total_mb']) * 100,
            gpu_utilization_percent=85.0,  # Estimado
            final_loss=final_loss,
            convergence_rate=convergence_rate,
            training_stability=training_stability,
            peak_memory_mb=int(peak_memory),
            average_memory_mb=int(avg_memory),
            oom_occurred=False
        )
    
    def _get_memory_usage(self) -> int:
        """Obtiene uso actual de memoria en MB."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.used // (1024 * 1024)
        except:
            return 0
    
    def _calculate_convergence_rate(self, losses: List[float]) -> float:
        """Calcula tasa de convergencia."""
        if len(losses) < 5:
            return 0.0
        
        # Calcular pendiente de los últimos valores
        recent_losses = losses[-min(10, len(losses)):]
        if len(recent_losses) < 2:
            return 0.0
        
        x = np.arange(len(recent_losses))
        slope = np.polyfit(x, recent_losses, 1)[0]
        
        # Convergencia = reducción de loss por iteración
        return abs(slope) if slope < 0 else 0.0
    
    def _calculate_training_stability(self, losses: List[float]) -> float:
        """Calcula estabilidad del entrenamiento."""
        if len(losses) < 5:
            return 0.0
        
        # Calcular varianza normalizada
        variance = np.var(losses)
        mean_loss = np.mean(losses)
        
        if mean_loss == 0:
            return 0.0
        
        # Estabilidad = 1 / (coeficiente de variación + 1)
        cv = np.sqrt(variance) / mean_loss
        stability = 1.0 / (cv + 1.0)
        
        return min(stability, 1.0)
    
    def _create_oom_result(
        self,
        config_name: str,
        batch_size: int,
        sequence_length: int,
        model_config: Dict[str, Any],
        use_mixed_precision: bool,
        tensor_cores_enabled: bool
    ) -> BenchmarkResults:
        """Crea resultado para OOM."""
        return BenchmarkResults(
            config_name=config_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            gpu_info=self.gpu_info,
            mixed_precision_enabled=use_mixed_precision,
            tensor_cores_enabled=tensor_cores_enabled,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_config=model_config,
            throughput_samples_per_sec=0.0,
            time_per_batch_ms=0.0,
            memory_usage_mb=0,
            memory_usage_percent=100.0,
            gpu_utilization_percent=0.0,
            final_loss=float('inf'),
            convergence_rate=0.0,
            training_stability=0.0,
            peak_memory_mb=self.gpu_info['memory_total_mb'],
            average_memory_mb=self.gpu_info['memory_total_mb'],
            oom_occurred=True,
            errors=['Out of Memory']
        )
    
    def _save_results(self):
        """Guarda resultados a archivo."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = Path(self.config.results_dir) / f"benchmark_results_{timestamp}.json"
        
        # Convertir resultados a formato serializable
        data = {
            'benchmark_info': {
                'timestamp': timestamp,
                'gpu_info': self.gpu_info,
                'config': {
                    'test_mixed_precision': self.config.test_mixed_precision,
                    'test_tensor_cores': self.config.test_tensor_cores,
                    'benchmark_duration': self.config.benchmark_duration
                }
            },
            'results': [result.to_dict() for result in self.results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _generate_report(self):
        """Genera reporte de resultados."""
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Filtrar resultados válidos
        valid_results = [r for r in self.results if not r.oom_occurred]
        
        if not valid_results:
            logger.warning("All benchmarks resulted in OOM")
            return
        
        # Encontrar mejor configuración
        best_throughput = max(valid_results, key=lambda r: r.throughput_samples_per_sec)
        best_efficiency = max(valid_results, key=lambda r: r.efficiency_score)
        
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"GPU: {self.gpu_info['gpu_name']}")
        logger.info(f"Total configurations tested: {len(self.results)}")
        logger.info(f"Successful configurations: {len(valid_results)}")
        logger.info(f"OOM failures: {len([r for r in self.results if r.oom_occurred])}")
        
        logger.info("\nBEST THROUGHPUT:")
        logger.info(f"Configuration: {best_throughput.config_name}")
        logger.info(f"Throughput: {best_throughput.throughput_samples_per_sec:.2f} samples/sec")
        logger.info(f"Memory usage: {best_throughput.memory_usage_percent:.1f}%")
        logger.info(f"Mixed precision: {best_throughput.mixed_precision_enabled}")
        
        logger.info("\nBEST EFFICIENCY:")
        logger.info(f"Configuration: {best_efficiency.config_name}")
        logger.info(f"Efficiency score: {best_efficiency.efficiency_score:.3f}")
        logger.info(f"Throughput: {best_efficiency.throughput_samples_per_sec:.2f} samples/sec")
        logger.info(f"Memory usage: {best_efficiency.memory_usage_percent:.1f}%")
        
        # Análisis de mixed precision
        mp_results = [r for r in valid_results if r.mixed_precision_enabled]
        fp32_results = [r for r in valid_results if not r.mixed_precision_enabled]
        
        if mp_results and fp32_results:
            mp_avg_throughput = np.mean([r.throughput_samples_per_sec for r in mp_results])
            fp32_avg_throughput = np.mean([r.throughput_samples_per_sec for r in fp32_results])
            speedup = mp_avg_throughput / fp32_avg_throughput
            
            logger.info(f"\nMIXED PRECISION ANALYSIS:")
            logger.info(f"Average FP32 throughput: {fp32_avg_throughput:.2f} samples/sec")
            logger.info(f"Average FP16 throughput: {mp_avg_throughput:.2f} samples/sec")
            logger.info(f"Mixed precision speedup: {speedup:.2f}x")
        
        logger.info("=" * 60)


class ConfigurationTester:
    """Tester para configuraciones específicas."""
    
    def __init__(self):
        self.benchmark = GPUBenchmark()
    
    def test_mixed_precision_impact(self) -> Dict[str, Any]:
        """Prueba el impacto de mixed precision."""
        logger.info("Testing mixed precision impact...")
        
        # Configuración base
        config = BenchmarkConfig(
            test_mixed_precision=True,
            test_tensor_cores=False,
            test_batch_sizes=[32],
            test_sequence_lengths=[100],
            model_configs=[{'lstm_units': 256, 'layers': 2, 'vocab_size': 5000}]
        )
        
        benchmark = GPUBenchmark(config)
        results = benchmark.run_comprehensive_benchmark()
        
        # Analizar resultados
        fp32_result = next((r for r in results if not r.mixed_precision_enabled), None)
        fp16_result = next((r for r in results if r.mixed_precision_enabled), None)
        
        if fp32_result and fp16_result:
            return {
                'fp32_throughput': fp32_result.throughput_samples_per_sec,
                'fp16_throughput': fp16_result.throughput_samples_per_sec,
                'speedup': fp16_result.throughput_samples_per_sec / fp32_result.throughput_samples_per_sec,
                'fp32_memory': fp32_result.memory_usage_mb,
                'fp16_memory': fp16_result.memory_usage_mb,
                'memory_savings': 1 - (fp16_result.memory_usage_mb / fp32_result.memory_usage_mb)
            }
        
        return {'error': 'Could not complete mixed precision test'}
    
    def find_optimal_configuration(self) -> Dict[str, Any]:
        """Encuentra la configuración óptima."""
        logger.info("Finding optimal configuration...")
        
        results = self.benchmark.run_comprehensive_benchmark()
        valid_results = [r for r in results if not r.oom_occurred]
        
        if not valid_results:
            return {'error': 'No valid configurations found'}
        
        # Encontrar óptimo basado en diferentes criterios
        best_throughput = max(valid_results, key=lambda r: r.throughput_samples_per_sec)
        best_efficiency = max(valid_results, key=lambda r: r.efficiency_score)
        best_memory = min(valid_results, key=lambda r: r.memory_usage_mb)
        
        return {
            'best_throughput': best_throughput.to_dict(),
            'best_efficiency': best_efficiency.to_dict(),
            'best_memory': best_memory.to_dict(),
            'recommendation': best_efficiency.config_name
        }


def run_comprehensive_benchmark(config: Optional[BenchmarkConfig] = None) -> List[BenchmarkResults]:
    """
    Función de conveniencia para ejecutar benchmark completo.
    
    Args:
        config: Configuración de benchmark
        
    Returns:
        Lista de resultados
    """
    benchmark = GPUBenchmark(config)
    return benchmark.run_comprehensive_benchmark()