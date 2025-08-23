"""
Gestión dinámica de memoria GPU para RTX 2000 Ada.

Implementa gestión eficiente de memoria VRAM con growth dinámico,
monitoreo y optimización para 8GB GDDR6.
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import threading
import tensorflow as tf
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuración de gestión de memoria."""
    
    # RTX 2000 Ada tiene 8GB GDDR6
    total_memory_mb: int = 8192
    
    # Reservar memoria para el sistema
    reserved_memory_mb: int = 512
    
    # Límite de memoria para el modelo
    max_model_memory_mb: Optional[int] = None
    
    # Growth configuration
    enable_growth: bool = True
    memory_limit_mb: Optional[int] = 7680  # 7.5GB de 8GB
    
    # Fragmentación
    allow_fragmentation: bool = False
    defragmentation_threshold: float = 0.3  # 30% fragmentación
    
    # Monitoreo
    monitor_interval: float = 5.0  # segundos
    log_memory_usage: bool = True
    
    # Optimizaciones
    use_unified_memory: bool = False  # Para GPUs con soporte
    prefetch_data: bool = True
    
    def get_available_memory(self) -> int:
        """Calcula memoria disponible para el modelo."""
        available = self.total_memory_mb - self.reserved_memory_mb
        if self.max_model_memory_mb:
            available = min(available, self.max_model_memory_mb)
        return available


class DynamicMemoryGrowth:
    """Gestor de crecimiento dinámico de memoria."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.initial_memory = 0
        self.current_memory = 0
        self.peak_memory = 0
        self.enabled = False
        
        self._setup_memory_growth()
    
    def _setup_memory_growth(self):
        """Configura crecimiento dinámico de memoria."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            
            if not gpus:
                logger.warning("No GPU found for memory management")
                return
            
            for gpu in gpus:
                try:
                    if self.config.enable_growth:
                        # Habilitar memory growth
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Memory growth enabled for {gpu.name}")
                    
                    if self.config.memory_limit_mb:
                        # Establecer límite de memoria
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=self.config.memory_limit_mb
                            )]
                        )
                        logger.info(f"Memory limit set to {self.config.memory_limit_mb}MB")
                    
                    self.enabled = True
                    
                except RuntimeError as e:
                    logger.error(f"Failed to configure GPU memory: {e}")
                    
        except Exception as e:
            logger.error(f"GPU memory configuration failed: {e}")
    
    def get_memory_info(self) -> Dict[str, int]:
        """Obtiene información actual de memoria."""
        try:
            # Intentar usar nvidia-ml-py si está disponible
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            info = {
                'total_mb': mem_info.total // (1024 * 1024),
                'used_mb': mem_info.used // (1024 * 1024),
                'free_mb': mem_info.free // (1024 * 1024),
                'usage_percent': (mem_info.used / mem_info.total) * 100
            }
            
            # Actualizar estadísticas
            self.current_memory = info['used_mb']
            self.peak_memory = max(self.peak_memory, self.current_memory)
            
            return info
            
        except ImportError:
            # Fallback sin pynvml
            return self._get_memory_info_tensorflow()
        except Exception as e:
            logger.debug(f"Could not get GPU memory info: {e}")
            return {'error': str(e)}
    
    def _get_memory_info_tensorflow(self) -> Dict[str, int]:
        """Obtiene información de memoria usando TensorFlow."""
        # TensorFlow no proporciona API directa, usar estimaciones
        return {
            'total_mb': self.config.total_memory_mb,
            'used_mb': self.current_memory,
            'free_mb': self.config.total_memory_mb - self.current_memory,
            'usage_percent': (self.current_memory / self.config.total_memory_mb) * 100
        }


class MemoryOptimizer:
    """Optimizador de uso de memoria GPU."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.memory_growth = DynamicMemoryGrowth(config)
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.memory_history = []
        
    def optimize_batch_size(
        self, 
        model: tf.keras.Model,
        base_batch_size: int = 32,
        sequence_length: int = 100
    ) -> int:
        """
        Encuentra el batch size óptimo para la memoria disponible.
        
        Args:
            model: Modelo a optimizar
            base_batch_size: Batch size inicial
            sequence_length: Longitud de secuencia
            
        Returns:
            Batch size óptimo
        """
        logger.info("Finding optimal batch size for available memory...")
        
        # Obtener memoria disponible
        mem_info = self.memory_growth.get_memory_info()
        available_mb = mem_info.get('free_mb', 4000)
        
        # Estimar memoria por batch
        params = model.count_params()
        
        # Estimación: params * 4 bytes (float32) + activaciones
        model_memory_mb = (params * 4) / (1024 * 1024)
        
        # Memoria para activaciones (aproximada)
        # Para LSTM: batch_size * sequence_length * hidden_size * num_layers * factor
        activation_memory_per_batch = (base_batch_size * sequence_length * 256 * 2 * 8) / (1024 * 1024)
        
        # Calcular batch size máximo
        overhead_factor = 1.5  # Factor de seguridad
        usable_memory = available_mb * 0.8  # Usar solo 80% de memoria disponible
        
        max_batches = int((usable_memory - model_memory_mb * overhead_factor) / activation_memory_per_batch)
        optimal_batch_size = min(max(max_batches, 1), 256)  # Limitar a 256 máximo
        
        # Asegurar que es múltiplo de 8 para Tensor Cores
        optimal_batch_size = (optimal_batch_size // 8) * 8
        optimal_batch_size = max(optimal_batch_size, 8)
        
        logger.info(f"Model memory: {model_memory_mb:.1f}MB")
        logger.info(f"Available memory: {available_mb}MB")
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def start_monitoring(self):
        """Inicia monitoreo de memoria en segundo plano."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor_memory)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Detiene monitoreo de memoria."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self):
        """Thread de monitoreo de memoria."""
        while not self.stop_monitoring.is_set():
            try:
                mem_info = self.memory_growth.get_memory_info()
                
                if 'error' not in mem_info:
                    # Registrar en historial
                    mem_info['timestamp'] = time.time()
                    self.memory_history.append(mem_info)
                    
                    # Mantener solo últimas 1000 entradas
                    if len(self.memory_history) > 1000:
                        self.memory_history = self.memory_history[-1000:]
                    
                    # Log si está habilitado
                    if self.config.log_memory_usage:
                        logger.debug(
                            f"GPU Memory: {mem_info['used_mb']}MB / {mem_info['total_mb']}MB "
                            f"({mem_info['usage_percent']:.1f}%)"
                        )
                    
                    # Alertar si memoria está muy alta
                    if mem_info['usage_percent'] > 90:
                        logger.warning(f"High GPU memory usage: {mem_info['usage_percent']:.1f}%")
                    
            except Exception as e:
                logger.debug(f"Memory monitoring error: {e}")
            
            time.sleep(self.config.monitor_interval)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de uso de memoria."""
        if not self.memory_history:
            return {'error': 'No memory history available'}
        
        used_values = [h['used_mb'] for h in self.memory_history if 'used_mb' in h]
        
        if not used_values:
            return {'error': 'No valid memory data'}
        
        return {
            'current_mb': used_values[-1] if used_values else 0,
            'peak_mb': max(used_values),
            'average_mb': np.mean(used_values),
            'min_mb': min(used_values),
            'samples': len(used_values),
            'monitoring_duration': len(self.memory_history) * self.config.monitor_interval
        }
    
    def optimize_model_for_memory(self, model: tf.keras.Model) -> Dict[str, Any]:
        """
        Optimiza modelo para uso eficiente de memoria.
        
        Args:
            model: Modelo a optimizar
            
        Returns:
            Recomendaciones de optimización
        """
        recommendations = {
            'gradient_checkpointing': False,
            'mixed_precision': True,
            'reduce_model_size': False,
            'batch_size_recommendation': 32,
            'optimizations': []
        }
        
        # Calcular uso de memoria del modelo
        params = model.count_params()
        model_size_mb = (params * 4) / (1024 * 1024)  # float32
        
        logger.info(f"Model size: {model_size_mb:.1f}MB ({params:,} parameters)")
        
        # Recomendaciones basadas en tamaño
        if model_size_mb > 2000:  # Modelo grande (>2GB)
            recommendations['gradient_checkpointing'] = True
            recommendations['optimizations'].append("Enable gradient checkpointing to save memory")
        
        if model_size_mb > 1000:  # Modelo mediano-grande (>1GB)
            recommendations['mixed_precision'] = True
            recommendations['optimizations'].append("Use mixed precision training (FP16)")
        
        # Calcular batch size recomendado
        mem_info = self.memory_growth.get_memory_info()
        available_mb = mem_info.get('free_mb', 4000)
        
        if available_mb < 2000:
            recommendations['batch_size_recommendation'] = 8
            recommendations['optimizations'].append("Use small batch size due to limited memory")
        elif available_mb < 4000:
            recommendations['batch_size_recommendation'] = 16
        else:
            recommendations['batch_size_recommendation'] = 32
        
        # Verificar si el modelo es muy grande para la GPU
        if model_size_mb > available_mb * 0.5:
            recommendations['reduce_model_size'] = True
            recommendations['optimizations'].append("Consider reducing model size or using model parallelism")
        
        return recommendations


class GPUMemoryManager:
    """Manager principal de memoria GPU."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.memory_growth = DynamicMemoryGrowth(config)
        self.optimizer = MemoryOptimizer(config)
        
        # Estado
        self.is_configured = False
        self.memory_callbacks = []
        
        self._configure()
    
    def _configure(self):
        """Configura gestión de memoria."""
        try:
            # Configurar TensorFlow
            self._configure_tensorflow()
            
            # Iniciar monitoreo si está habilitado
            if self.config.log_memory_usage:
                self.optimizer.start_monitoring()
            
            self.is_configured = True
            logger.info("GPU memory manager configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure GPU memory manager: {e}")
    
    def _configure_tensorflow(self):
        """Configura opciones de memoria de TensorFlow."""
        # Opciones de GPU
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Allocator asíncrono
        os.environ['TF_GPU_HOST_MEM_LIMIT_IN_MB'] = '2048'  # Límite de memoria host
        
        # Prefetching
        if self.config.prefetch_data:
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = '2'
    
    def add_memory_callback(self, callback: callable):
        """Añade callback para eventos de memoria."""
        self.memory_callbacks.append(callback)
    
    def check_memory_availability(self, required_mb: int) -> bool:
        """
        Verifica si hay suficiente memoria disponible.
        
        Args:
            required_mb: Memoria requerida en MB
            
        Returns:
            True si hay suficiente memoria
        """
        mem_info = self.memory_growth.get_memory_info()
        available = mem_info.get('free_mb', 0)
        
        has_enough = available >= required_mb
        
        if not has_enough:
            logger.warning(f"Insufficient GPU memory: {available}MB available, {required_mb}MB required")
            
            # Ejecutar callbacks
            for callback in self.memory_callbacks:
                try:
                    callback('insufficient_memory', mem_info)
                except Exception as e:
                    logger.error(f"Memory callback failed: {e}")
        
        return has_enough
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """Obtiene perfil completo de memoria."""
        return {
            'current': self.memory_growth.get_memory_info(),
            'summary': self.optimizer.get_memory_summary(),
            'config': {
                'total_memory_mb': self.config.total_memory_mb,
                'reserved_mb': self.config.reserved_memory_mb,
                'memory_limit_mb': self.config.memory_limit_mb,
                'growth_enabled': self.config.enable_growth
            },
            'peak_memory_mb': self.memory_growth.peak_memory,
            'is_configured': self.is_configured
        }
    
    def cleanup(self):
        """Limpia recursos."""
        try:
            self.optimizer.stop_monitoring()
            
            # Limpiar caché de GPU si es posible
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
                logger.info("GPU memory cache cleared")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def get_gpu_memory_info() -> Dict[str, int]:
    """
    Función de utilidad para obtener información de memoria GPU.
    
    Returns:
        Diccionario con información de memoria
    """
    manager = DynamicMemoryGrowth()
    return manager.get_memory_info()