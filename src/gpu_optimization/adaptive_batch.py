"""
Gestión adaptativa de batch size para RTX 2000 Ada.

Implementa ajuste dinámico del batch size basado en memoria disponible,
rendimiento y características del modelo.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)


@dataclass
class BatchSizeConfig:
    """Configuración para gestión adaptativa de batch size."""
    
    # Límites
    min_batch_size: int = 1
    max_batch_size: int = 256
    initial_batch_size: int = 32
    
    # Estrategia de búsqueda
    search_strategy: str = 'binary'  # 'binary', 'exponential', 'linear'
    growth_factor: float = 1.5
    
    # Criterios de optimización
    optimize_for: str = 'memory'  # 'memory', 'speed', 'balanced'
    memory_utilization_target: float = 0.80  # 80% de memoria GPU
    speed_tolerance: float = 0.05  # 5% tolerancia en velocidad
    
    # Alineación para Tensor Cores
    align_to_tensor_cores: bool = True
    tensor_core_alignment: int = 8
    
    # Monitoreo
    benchmark_iterations: int = 5
    warmup_iterations: int = 3
    
    # Adaptación dinámica
    enable_dynamic_adjustment: bool = True
    adjustment_frequency: int = 100  # cada N batches
    performance_window: int = 20  # ventana para calcular promedios


@dataclass
class BatchPerformance:
    """Información de rendimiento para un batch size."""
    
    batch_size: int
    throughput_samples_per_sec: float
    memory_usage_mb: int
    memory_usage_percent: float
    time_per_batch: float
    gpu_utilization: float = 0.0
    oom_error: bool = False
    
    # Métricas derivadas
    efficiency_score: float = field(init=False)
    memory_efficiency: float = field(init=False)
    
    def __post_init__(self):
        # Score de eficiencia combinado
        self.memory_efficiency = self.batch_size / max(self.memory_usage_mb, 1)
        
        # Score que combina throughput y uso de memoria
        if self.oom_error:
            self.efficiency_score = 0.0
        else:
            # Normalizar métricas (0-1) y combinar
            throughput_norm = min(self.throughput_samples_per_sec / 1000, 1.0)
            memory_norm = min(self.memory_usage_percent / 100, 1.0)
            
            # Favorecer throughput pero penalizar uso excesivo de memoria
            self.efficiency_score = throughput_norm * (1.0 - memory_norm * 0.5)


class AdaptiveBatchSizeManager:
    """Manager para ajuste adaptativo de batch size."""
    
    def __init__(self, config: Optional[BatchSizeConfig] = None):
        self.config = config or BatchSizeConfig()
        self.current_batch_size = self.config.initial_batch_size
        self.optimal_batch_size = self.config.initial_batch_size
        
        # Historial de rendimiento
        self.performance_history: List[BatchPerformance] = []
        self.batch_history: List[int] = []
        
        # Estado de adaptación
        self.adaptation_enabled = False
        self.batch_count = 0
        self.last_adjustment_batch = 0
        
        # Callbacks
        self.adjustment_callbacks: List[Callable] = []
        
    def find_optimal_batch_size(
        self, 
        model: tf.keras.Model, 
        sample_data: tf.data.Dataset,
        max_search_time: int = 300  # 5 minutos máximo
    ) -> int:
        """
        Encuentra el batch size óptimo para el modelo.
        
        Args:
            model: Modelo a optimizar
            sample_data: Dataset de prueba
            max_search_time: Tiempo máximo de búsqueda en segundos
            
        Returns:
            Batch size óptimo
        """
        logger.info("Finding optimal batch size...")
        start_time = time.time()
        
        # Estrategias de búsqueda
        if self.config.search_strategy == 'binary':
            return self._binary_search(model, sample_data, start_time, max_search_time)
        elif self.config.search_strategy == 'exponential':
            return self._exponential_search(model, sample_data, start_time, max_search_time)
        else:
            return self._linear_search(model, sample_data, start_time, max_search_time)
    
    def _binary_search(
        self, 
        model: tf.keras.Model, 
        sample_data: tf.data.Dataset,
        start_time: float,
        max_search_time: int
    ) -> int:
        """Búsqueda binaria del batch size óptimo."""
        low = self.config.min_batch_size
        high = self.config.max_batch_size
        best_batch_size = self.config.initial_batch_size
        best_performance = None
        
        logger.info(f"Binary search between {low} and {high}")
        
        while low <= high and (time.time() - start_time) < max_search_time:
            mid = (low + high) // 2
            
            # Alinear a Tensor Cores si está habilitado
            if self.config.align_to_tensor_cores:
                mid = self._align_batch_size(mid)
            
            logger.info(f"Testing batch size: {mid}")
            
            try:
                performance = self._benchmark_batch_size(model, sample_data, mid)
                self.performance_history.append(performance)
                
                if performance.oom_error:
                    high = mid - 1
                    logger.info(f"OOM at batch size {mid}, reducing search space")
                else:
                    # Comparar con mejor performance actual
                    if (best_performance is None or 
                        performance.efficiency_score > best_performance.efficiency_score):
                        best_performance = performance
                        best_batch_size = mid
                    
                    # Decidir dirección de búsqueda basada en criterio
                    if self.config.optimize_for == 'memory':
                        if performance.memory_usage_percent < self.config.memory_utilization_target * 100:
                            low = mid + 1  # Puede usar más memoria
                        else:
                            high = mid - 1  # Demasiada memoria
                    elif self.config.optimize_for == 'speed':
                        # Siempre intentar batch size mayor para mejor throughput
                        if performance.memory_usage_percent < 95:  # Límite de seguridad
                            low = mid + 1
                        else:
                            high = mid - 1
                    else:  # balanced
                        if performance.efficiency_score > 0.7:  # Buen balance
                            low = mid + 1
                        else:
                            high = mid - 1
                
            except Exception as e:
                logger.warning(f"Error testing batch size {mid}: {e}")
                high = mid - 1
        
        self.optimal_batch_size = best_batch_size
        logger.info(f"Optimal batch size found: {best_batch_size}")
        
        if best_performance:
            logger.info(f"Performance: {best_performance.throughput_samples_per_sec:.2f} samples/sec")
            logger.info(f"Memory usage: {best_performance.memory_usage_percent:.1f}%")
            logger.info(f"Efficiency score: {best_performance.efficiency_score:.3f}")
        
        return best_batch_size
    
    def _exponential_search(
        self, 
        model: tf.keras.Model, 
        sample_data: tf.data.Dataset,
        start_time: float,
        max_search_time: int
    ) -> int:
        """Búsqueda exponencial del batch size óptimo."""
        batch_size = self.config.initial_batch_size
        best_batch_size = batch_size
        best_performance = None
        
        logger.info(f"Exponential search starting from {batch_size}")
        
        # Fase 1: Incrementar exponencialmente hasta OOM
        while batch_size <= self.config.max_batch_size and (time.time() - start_time) < max_search_time:
            aligned_batch = self._align_batch_size(batch_size)
            
            try:
                performance = self._benchmark_batch_size(model, sample_data, aligned_batch)
                self.performance_history.append(performance)
                
                if performance.oom_error:
                    logger.info(f"OOM reached at batch size {aligned_batch}")
                    break
                
                # Actualizar mejor performance
                if (best_performance is None or 
                    performance.efficiency_score > best_performance.efficiency_score):
                    best_performance = performance
                    best_batch_size = aligned_batch
                
                # Incrementar exponencialmente
                batch_size = int(batch_size * self.config.growth_factor)
                
            except Exception as e:
                logger.warning(f"Error at batch size {aligned_batch}: {e}")
                break
        
        # Fase 2: Búsqueda fina alrededor del mejor encontrado
        if best_batch_size > self.config.min_batch_size:
            fine_search_range = max(16, best_batch_size // 4)
            start_fine = max(self.config.min_batch_size, best_batch_size - fine_search_range)
            end_fine = min(self.config.max_batch_size, best_batch_size + fine_search_range)
            
            for test_batch in range(start_fine, end_fine + 1, 8):  # Pasos de 8
                if (time.time() - start_time) >= max_search_time:
                    break
                
                if test_batch not in [p.batch_size for p in self.performance_history]:
                    try:
                        performance = self._benchmark_batch_size(model, sample_data, test_batch)
                        self.performance_history.append(performance)
                        
                        if not performance.oom_error and performance.efficiency_score > best_performance.efficiency_score:
                            best_performance = performance
                            best_batch_size = test_batch
                            
                    except Exception as e:
                        logger.debug(f"Fine search error at {test_batch}: {e}")
        
        self.optimal_batch_size = best_batch_size
        logger.info(f"Exponential search completed: optimal batch size = {best_batch_size}")
        
        return best_batch_size
    
    def _linear_search(
        self, 
        model: tf.keras.Model, 
        sample_data: tf.data.Dataset,
        start_time: float,
        max_search_time: int
    ) -> int:
        """Búsqueda lineal del batch size óptimo."""
        best_batch_size = self.config.initial_batch_size
        best_performance = None
        
        step = 8 if self.config.align_to_tensor_cores else 4
        
        for batch_size in range(self.config.min_batch_size, self.config.max_batch_size + 1, step):
            if (time.time() - start_time) >= max_search_time:
                break
            
            try:
                performance = self._benchmark_batch_size(model, sample_data, batch_size)
                self.performance_history.append(performance)
                
                if performance.oom_error:
                    logger.info(f"OOM reached at batch size {batch_size}, stopping search")
                    break
                
                if (best_performance is None or 
                    performance.efficiency_score > best_performance.efficiency_score):
                    best_performance = performance
                    best_batch_size = batch_size
                
            except Exception as e:
                logger.warning(f"Error at batch size {batch_size}: {e}")
        
        self.optimal_batch_size = best_batch_size
        return best_batch_size
    
    def _benchmark_batch_size(
        self, 
        model: tf.keras.Model, 
        sample_data: tf.data.Dataset, 
        batch_size: int
    ) -> BatchPerformance:
        """
        Hace benchmark de un batch size específico.
        
        Args:
            model: Modelo a probar
            sample_data: Dataset de muestra
            batch_size: Batch size a probar
            
        Returns:
            Información de rendimiento
        """
        try:
            # Preparar dataset con el batch size
            batched_data = sample_data.batch(batch_size).take(self.config.benchmark_iterations + self.config.warmup_iterations)
            
            # Obtener memoria inicial
            initial_memory = self._get_memory_usage()
            
            # Compilar función de entrenamiento
            @tf.function
            def train_step(batch_x, batch_y):
                with tf.GradientTape() as tape:
                    predictions = model(batch_x, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(batch_y, predictions)
                    loss = tf.reduce_mean(loss)
                
                # Simular backward pass
                grads = tape.gradient(loss, model.trainable_variables)
                return loss
            
            # Warmup
            batch_iter = iter(batched_data)
            for _ in range(self.config.warmup_iterations):
                try:
                    batch_x, batch_y = next(batch_iter)
                    _ = train_step(batch_x, batch_y)
                except StopIteration:
                    break
                except tf.errors.ResourceExhaustedError:
                    return BatchPerformance(
                        batch_size=batch_size,
                        throughput_samples_per_sec=0.0,
                        memory_usage_mb=0,
                        memory_usage_percent=100.0,
                        time_per_batch=float('inf'),
                        oom_error=True
                    )
            
            # Benchmark real
            times = []
            start_time = time.time()
            
            for i in range(self.config.benchmark_iterations):
                try:
                    batch_x, batch_y = next(batch_iter)
                    
                    batch_start = time.time()
                    loss = train_step(batch_x, batch_y)
                    batch_end = time.time()
                    
                    times.append(batch_end - batch_start)
                    
                except StopIteration:
                    # Recrear iterator si se agotó
                    batch_iter = iter(batched_data.skip(self.config.warmup_iterations))
                    batch_x, batch_y = next(batch_iter)
                    
                    batch_start = time.time()
                    loss = train_step(batch_x, batch_y)
                    batch_end = time.time()
                    
                    times.append(batch_end - batch_start)
                    
                except tf.errors.ResourceExhaustedError:
                    return BatchPerformance(
                        batch_size=batch_size,
                        throughput_samples_per_sec=0.0,
                        memory_usage_mb=0,
                        memory_usage_percent=100.0,
                        time_per_batch=float('inf'),
                        oom_error=True
                    )
            
            total_time = time.time() - start_time
            current_memory = self._get_memory_usage()
            
            # Calcular métricas
            avg_time_per_batch = np.mean(times)
            throughput = batch_size / avg_time_per_batch
            memory_used = max(current_memory - initial_memory, current_memory)
            
            return BatchPerformance(
                batch_size=batch_size,
                throughput_samples_per_sec=throughput,
                memory_usage_mb=memory_used,
                memory_usage_percent=(memory_used / 8192) * 100,  # RTX 2000 Ada = 8GB
                time_per_batch=avg_time_per_batch,
                oom_error=False
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed for batch size {batch_size}: {e}")
            return BatchPerformance(
                batch_size=batch_size,
                throughput_samples_per_sec=0.0,
                memory_usage_mb=0,
                memory_usage_percent=100.0,
                time_per_batch=float('inf'),
                oom_error=True
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
    
    def _align_batch_size(self, batch_size: int) -> int:
        """Alinea batch size para Tensor Cores."""
        if not self.config.align_to_tensor_cores:
            return batch_size
        
        alignment = self.config.tensor_core_alignment
        aligned = ((batch_size + alignment - 1) // alignment) * alignment
        return max(aligned, alignment)  # Mínimo = alignment
    
    def enable_dynamic_adjustment(self):
        """Habilita ajuste dinámico durante entrenamiento."""
        self.adaptation_enabled = True
        logger.info("Dynamic batch size adjustment enabled")
    
    def disable_dynamic_adjustment(self):
        """Deshabilita ajuste dinámico."""
        self.adaptation_enabled = False
        logger.info("Dynamic batch size adjustment disabled")
    
    def should_adjust_batch_size(self, current_batch: int) -> bool:
        """Determina si se debe ajustar el batch size."""
        if not self.adaptation_enabled:
            return False
        
        # Ajustar según frecuencia configurada
        batches_since_adjustment = current_batch - self.last_adjustment_batch
        return batches_since_adjustment >= self.config.adjustment_frequency
    
    def suggest_batch_size_adjustment(
        self, 
        current_performance: Dict[str, float]
    ) -> Tuple[int, str]:
        """
        Sugiere ajuste de batch size basado en rendimiento actual.
        
        Args:
            current_performance: Métricas de rendimiento actuales
            
        Returns:
            Tuple de (nuevo_batch_size, razón)
        """
        current_batch = self.current_batch_size
        memory_usage = current_performance.get('memory_usage_percent', 50)
        throughput = current_performance.get('throughput', 100)
        
        # Lógica de ajuste
        if memory_usage < 60:  # Poca memoria usada
            new_batch = min(current_batch + 8, self.config.max_batch_size)
            reason = f"Low memory usage ({memory_usage:.1f}%) - increasing batch size"
            
        elif memory_usage > 85:  # Memoria alta
            new_batch = max(current_batch - 8, self.config.min_batch_size)
            reason = f"High memory usage ({memory_usage:.1f}%) - decreasing batch size"
            
        else:
            new_batch = current_batch
            reason = "No adjustment needed"
        
        # Alinear si es necesario
        if new_batch != current_batch:
            new_batch = self._align_batch_size(new_batch)
        
        return new_batch, reason
    
    def add_adjustment_callback(self, callback: Callable):
        """Añade callback para cuando se ajusta batch size."""
        self.adjustment_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de rendimiento."""
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        # Filtrar resultados válidos (sin OOM)
        valid_results = [p for p in self.performance_history if not p.oom_error]
        
        if not valid_results:
            return {'error': 'No valid performance data'}
        
        # Encontrar mejor resultado
        best = max(valid_results, key=lambda p: p.efficiency_score)
        
        return {
            'optimal_batch_size': best.batch_size,
            'max_throughput': max(p.throughput_samples_per_sec for p in valid_results),
            'min_memory_usage': min(p.memory_usage_percent for p in valid_results),
            'best_efficiency_score': best.efficiency_score,
            'total_tests': len(self.performance_history),
            'valid_tests': len(valid_results),
            'oom_count': len([p for p in self.performance_history if p.oom_error])
        }


class BatchSizeOptimizer:
    """Optimizador de alto nivel para batch size."""
    
    def __init__(self, config: Optional[BatchSizeConfig] = None):
        self.config = config or BatchSizeConfig()
        self.manager = AdaptiveBatchSizeManager(config)
    
    def optimize_for_model(
        self,
        model: tf.keras.Model,
        sample_data: tf.data.Dataset,
        target_metric: str = 'efficiency'
    ) -> Dict[str, Any]:
        """
        Optimiza batch size para un modelo específico.
        
        Args:
            model: Modelo a optimizar
            sample_data: Dataset de muestra
            target_metric: Métrica objetivo ('efficiency', 'speed', 'memory')
            
        Returns:
            Resultados de optimización
        """
        logger.info(f"Optimizing batch size for {target_metric}")
        
        # Ajustar configuración según métrica objetivo
        if target_metric == 'speed':
            self.config.optimize_for = 'speed'
        elif target_metric == 'memory':
            self.config.optimize_for = 'memory'
        else:
            self.config.optimize_for = 'balanced'
        
        # Encontrar batch size óptimo
        optimal_batch = self.manager.find_optimal_batch_size(model, sample_data)
        
        # Generar reporte
        performance_summary = self.manager.get_performance_summary()
        
        return {
            'optimal_batch_size': optimal_batch,
            'target_metric': target_metric,
            'performance_summary': performance_summary,
            'recommendations': self._generate_recommendations(performance_summary)
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en resultados."""
        recommendations = []
        
        if 'error' in summary:
            recommendations.append("Could not determine optimal batch size")
            return recommendations
        
        oom_count = summary.get('oom_count', 0)
        if oom_count > 0:
            recommendations.append(f"Encountered {oom_count} OOM errors - consider using memory optimization")
        
        optimal_batch = summary.get('optimal_batch_size', 32)
        if optimal_batch <= 8:
            recommendations.append("Small optimal batch size suggests memory constraints")
        elif optimal_batch >= 128:
            recommendations.append("Large optimal batch size suggests good memory capacity")
        
        max_throughput = summary.get('max_throughput', 0)
        if max_throughput > 1000:
            recommendations.append("High throughput achieved - good GPU utilization")
        elif max_throughput < 100:
            recommendations.append("Low throughput - consider model or data pipeline optimization")
        
        return recommendations


class DynamicBatchScheduler:
    """Scheduler para ajuste dinámico durante entrenamiento."""
    
    def __init__(self, manager: AdaptiveBatchSizeManager):
        self.manager = manager
        self.performance_window = []
        self.adjustment_history = []
    
    def on_batch_end(self, batch: int, logs: Dict[str, float]) -> Optional[int]:
        """
        Callback para final de batch - puede sugerir nuevo batch size.
        
        Args:
            batch: Número de batch
            logs: Métricas del batch
            
        Returns:
            Nuevo batch size si se sugiere cambio, None en caso contrario
        """
        # Registrar performance
        self.performance_window.append({
            'batch': batch,
            'throughput': logs.get('samples_per_sec', 100),
            'memory_usage_percent': logs.get('gpu_memory_percent', 50),
            'loss': logs.get('loss', 0)
        })
        
        # Mantener ventana de tamaño fijo
        if len(self.performance_window) > self.manager.config.performance_window:
            self.performance_window.pop(0)
        
        # Verificar si se debe ajustar
        if self.manager.should_adjust_batch_size(batch):
            # Calcular performance promedio
            if len(self.performance_window) >= 5:  # Mínimo de datos
                avg_performance = {
                    'throughput': np.mean([p['throughput'] for p in self.performance_window]),
                    'memory_usage_percent': np.mean([p['memory_usage_percent'] for p in self.performance_window])
                }
                
                new_batch, reason = self.manager.suggest_batch_size_adjustment(avg_performance)
                
                if new_batch != self.manager.current_batch_size:
                    logger.info(f"Suggesting batch size change: {self.manager.current_batch_size} -> {new_batch}")
                    logger.info(f"Reason: {reason}")
                    
                    self.adjustment_history.append({
                        'batch': batch,
                        'old_batch_size': self.manager.current_batch_size,
                        'new_batch_size': new_batch,
                        'reason': reason
                    })
                    
                    self.manager.current_batch_size = new_batch
                    self.manager.last_adjustment_batch = batch
                    
                    return new_batch
        
        return None


def find_optimal_batch_size(
    model: tf.keras.Model,
    sample_data: tf.data.Dataset,
    config: Optional[BatchSizeConfig] = None
) -> int:
    """
    Función de conveniencia para encontrar batch size óptimo.
    
    Args:
        model: Modelo a optimizar
        sample_data: Dataset de muestra
        config: Configuración opcional
        
    Returns:
        Batch size óptimo
    """
    manager = AdaptiveBatchSizeManager(config)
    return manager.find_optimal_batch_size(model, sample_data)