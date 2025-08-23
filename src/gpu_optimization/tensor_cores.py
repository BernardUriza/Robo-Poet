"""
Optimización de Tensor Cores para RTX 2000 Ada Generation.

Implementa optimizaciones específicas para aprovechar los Tensor Cores
de 4ta generación de la arquitectura Ada Lovelace.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import tensorflow as tf
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TensorCoreConfig:
    """Configuración para optimización de Tensor Cores."""
    
    # Alineación de dimensiones (múltiplos de 8 para Ada)
    dimension_alignment: int = 8
    
    # Tipos de datos soportados por Ada Lovelace
    supported_dtypes: List[str] = None
    
    # Configuración de operaciones
    use_tf32: bool = True  # TensorFloat-32 para mejor precisión
    use_fp16: bool = True  # FP16 para máxima velocidad
    use_fp8: bool = False  # FP8 (experimental en Ada)
    
    # Optimizaciones específicas
    optimize_matmul: bool = True
    optimize_conv: bool = True
    optimize_attention: bool = True
    
    # Tamaños de tile óptimos para Ada
    matmul_tile_size: int = 256
    conv_tile_size: int = 128
    
    def __post_init__(self):
        if self.supported_dtypes is None:
            self.supported_dtypes = ['float16', 'float32', 'bfloat16']


class TensorCoreOptimizer:
    """Optimizador para aprovechar Tensor Cores de RTX 2000 Ada."""
    
    def __init__(self, config: Optional[TensorCoreConfig] = None):
        self.config = config or TensorCoreConfig()
        self.gpu_info = self._get_gpu_info()
        self.tensor_core_available = self._verify_tensor_cores()
        
        if self.tensor_core_available:
            logger.info("Tensor Cores optimization available for RTX 2000 Ada")
            self._configure_tensor_cores()
        else:
            logger.warning("Tensor Cores not available or not detected")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Obtiene información de la GPU."""
        info = {
            'gpu_available': False,
            'gpu_name': 'Unknown',
            'compute_capability': (0, 0),
            'memory_mb': 0,
            'tensor_cores': False
        }
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info['gpu_available'] = True
                
                # Obtener detalles del dispositivo
                for gpu in gpus:
                    try:
                        details = tf.config.experimental.get_device_details(gpu)
                        
                        # RTX 2000 Ada tiene compute capability 8.9
                        compute_cap = details.get('compute_capability', (0, 0))
                        info['compute_capability'] = compute_cap
                        
                        # Ada Lovelace (8.9) tiene Tensor Cores de 4ta gen
                        if compute_cap[0] >= 7:  # Tensor Cores desde Volta (7.0)
                            info['tensor_cores'] = True
                        
                        # Nombre del dispositivo
                        device_name = details.get('device_name', 'Unknown')
                        info['gpu_name'] = device_name
                        
                        logger.info(f"GPU detected: {device_name}")
                        logger.info(f"Compute capability: {compute_cap[0]}.{compute_cap[1]}")
                        
                    except Exception as e:
                        logger.debug(f"Could not get GPU details: {e}")
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
        
        return info
    
    def _verify_tensor_cores(self) -> bool:
        """Verifica disponibilidad de Tensor Cores."""
        if not self.gpu_info['gpu_available']:
            return False
        
        compute_cap = self.gpu_info['compute_capability']
        
        # Tensor Cores disponibles desde compute capability 7.0
        # RTX 2000 Ada tiene 8.9
        if compute_cap[0] >= 7:
            logger.info(f"Tensor Cores Gen {self._get_tensor_core_generation(compute_cap)} detected")
            return True
        
        return False
    
    def _get_tensor_core_generation(self, compute_cap: Tuple[int, int]) -> int:
        """Determina generación de Tensor Cores."""
        major, minor = compute_cap
        
        if major == 7 and minor in [0, 2]:
            return 1  # Volta
        elif major == 7 and minor == 5:
            return 2  # Turing
        elif major == 8 and minor in [0, 6, 7]:
            return 3  # Ampere
        elif major == 8 and minor == 9:
            return 4  # Ada Lovelace (RTX 2000 Ada)
        elif major == 9:
            return 5  # Hopper
        else:
            return 0
    
    def _configure_tensor_cores(self):
        """Configura optimizaciones para Tensor Cores."""
        import os
        
        # TensorFloat-32 para mejor balance precisión/velocidad
        if self.config.use_tf32:
            os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
            os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
            os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
            logger.info("TensorFloat-32 enabled for Tensor Cores")
        
        # Optimizaciones para cuDNN
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
        os.environ['TF_ENABLE_CUDNN_FRONTEND'] = '1'
        os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '512'  # Para RTX 2000 con 8GB
        
        # Deshabilitar determinismo para máximo rendimiento
        os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
        os.environ['TF_DETERMINISTIC_OPS'] = '0'
        
        logger.info("Tensor Core optimizations configured for Ada Lovelace")
    
    def optimize_layer(self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        """
        Optimiza una capa para Tensor Cores.
        
        Args:
            layer: Capa a optimizar
            
        Returns:
            Capa optimizada
        """
        if not self.tensor_core_available:
            return layer
        
        layer_type = type(layer).__name__
        
        if isinstance(layer, tf.keras.layers.Dense):
            return self._optimize_dense_layer(layer)
        elif isinstance(layer, tf.keras.layers.LSTM):
            return self._optimize_lstm_layer(layer)
        elif isinstance(layer, tf.keras.layers.Conv1D):
            return self._optimize_conv_layer(layer)
        elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
            return self._optimize_attention_layer(layer)
        else:
            return layer
    
    def _optimize_dense_layer(self, layer: tf.keras.layers.Dense) -> tf.keras.layers.Dense:
        """Optimiza capa Dense para Tensor Cores."""
        units = layer.units
        aligned_units = self._align_dimension(units)
        
        if units != aligned_units:
            logger.info(f"Aligning Dense layer from {units} to {aligned_units} units")
            
            # Crear nueva capa con dimensiones alineadas
            new_layer = tf.keras.layers.Dense(
                units=aligned_units,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                name=f"{layer.name}_tc_optimized"
            )
            
            return new_layer
        
        return layer
    
    def _optimize_lstm_layer(self, layer: tf.keras.layers.LSTM) -> tf.keras.layers.LSTM:
        """Optimiza capa LSTM para Tensor Cores."""
        units = layer.units
        aligned_units = self._align_dimension(units)
        
        if units != aligned_units:
            logger.info(f"Aligning LSTM layer from {units} to {aligned_units} units")
            
            # LSTM con CuDNN para máximo rendimiento
            new_layer = tf.keras.layers.LSTM(
                units=aligned_units,
                return_sequences=layer.return_sequences,
                return_state=layer.return_state,
                go_backwards=layer.go_backwards,
                stateful=layer.stateful,
                unroll=False,  # No unroll para usar CuDNN
                time_major=False,
                dropout=layer.dropout,
                recurrent_dropout=0.0,  # CuDNN no soporta recurrent dropout
                implementation=2,  # Usar implementación optimizada
                name=f"{layer.name}_tc_optimized"
            )
            
            return new_layer
        
        # Asegurar que use CuDNN
        if layer.recurrent_dropout > 0:
            logger.warning(f"LSTM {layer.name} has recurrent_dropout - cannot use CuDNN acceleration")
        
        return layer
    
    def _optimize_conv_layer(self, layer: tf.keras.layers.Conv1D) -> tf.keras.layers.Conv1D:
        """Optimiza capa convolucional para Tensor Cores."""
        filters = layer.filters
        aligned_filters = self._align_dimension(filters)
        
        if filters != aligned_filters:
            logger.info(f"Aligning Conv1D from {filters} to {aligned_filters} filters")
            
            new_layer = tf.keras.layers.Conv1D(
                filters=aligned_filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                groups=layer.groups,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                name=f"{layer.name}_tc_optimized"
            )
            
            return new_layer
        
        return layer
    
    def _optimize_attention_layer(
        self, 
        layer: tf.keras.layers.MultiHeadAttention
    ) -> tf.keras.layers.MultiHeadAttention:
        """Optimiza capa de atención para Tensor Cores."""
        # Multi-head attention beneficia mucho de Tensor Cores
        key_dim = layer.key_dim
        aligned_key_dim = self._align_dimension(key_dim)
        
        num_heads = layer.num_heads
        # Para Ada, mejor rendimiento con múltiplos de 8 heads
        aligned_heads = self._align_dimension(num_heads)
        
        if key_dim != aligned_key_dim or num_heads != aligned_heads:
            logger.info(
                f"Aligning attention: key_dim {key_dim}->{aligned_key_dim}, "
                f"heads {num_heads}->{aligned_heads}"
            )
            
            new_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=aligned_heads,
                key_dim=aligned_key_dim,
                value_dim=layer.value_dim,
                dropout=layer.dropout,
                use_bias=layer.use_bias,
                output_shape=layer.output_shape,
                attention_axes=layer.attention_axes,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint,
                name=f"{layer.name}_tc_optimized"
            )
            
            return new_layer
        
        return layer
    
    def _align_dimension(self, dim: int) -> int:
        """Alinea dimensión a múltiplo óptimo para Tensor Cores."""
        alignment = self.config.dimension_alignment
        return ((dim + alignment - 1) // alignment) * alignment
    
    def optimize_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Optimiza modelo completo para Tensor Cores.
        
        Args:
            model: Modelo a optimizar
            
        Returns:
            Modelo optimizado
        """
        if not self.tensor_core_available:
            logger.warning("Tensor Cores not available - returning original model")
            return model
        
        logger.info("Optimizing model for Tensor Cores...")
        
        # Para modelos Sequential
        if isinstance(model, tf.keras.Sequential):
            optimized_layers = []
            for layer in model.layers:
                optimized_layer = self.optimize_layer(layer)
                optimized_layers.append(optimized_layer)
            
            optimized_model = tf.keras.Sequential(optimized_layers, name=f"{model.name}_tc")
            
        # Para modelos funcionales o subclassed
        else:
            # Por ahora, solo advertir sobre optimizaciones manuales
            logger.warning(
                "Model optimization for non-Sequential models requires manual layer replacement. "
                "Consider using optimize_layer() on individual layers."
            )
            optimized_model = model
        
        # Log optimizaciones aplicadas
        self._log_optimization_summary(model, optimized_model)
        
        return optimized_model
    
    def _log_optimization_summary(self, original_model: tf.keras.Model, optimized_model: tf.keras.Model):
        """Registra resumen de optimizaciones."""
        original_params = original_model.count_params()
        optimized_params = optimized_model.count_params()
        
        logger.info("=" * 50)
        logger.info("Tensor Core Optimization Summary")
        logger.info("=" * 50)
        logger.info(f"Original parameters: {original_params:,}")
        logger.info(f"Optimized parameters: {optimized_params:,}")
        logger.info(f"Parameter increase: {(optimized_params - original_params):,}")
        logger.info(f"Compute capability: {self.gpu_info['compute_capability']}")
        logger.info(f"Tensor Core generation: {self._get_tensor_core_generation(self.gpu_info['compute_capability'])}")
        logger.info("=" * 50)
    
    def benchmark_matmul(self, size: int = 1024, dtype: str = 'float16', iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark de multiplicación matricial con Tensor Cores.
        
        Args:
            size: Tamaño de las matrices
            dtype: Tipo de dato
            iterations: Número de iteraciones
            
        Returns:
            Resultados del benchmark
        """
        if not self.tensor_core_available:
            return {'error': 'Tensor Cores not available'}
        
        # Alinear tamaño
        aligned_size = self._align_dimension(size)
        
        logger.info(f"Running Tensor Core matmul benchmark: {aligned_size}x{aligned_size} {dtype}")
        
        # Crear matrices aleatorias
        a = tf.random.normal([aligned_size, aligned_size], dtype=dtype)
        b = tf.random.normal([aligned_size, aligned_size], dtype=dtype)
        
        # Warmup
        for _ in range(10):
            _ = tf.matmul(a, b)
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(iterations):
            result = tf.matmul(a, b)
        
        # Sincronizar GPU
        if tf.config.list_physical_devices('GPU'):
            tf.debugging.assert_all_finite(result, "Result check")
        
        elapsed_time = time.time() - start_time
        
        # Calcular TFLOPS
        ops_per_matmul = 2 * aligned_size ** 3  # 2*N^3 operaciones
        total_ops = ops_per_matmul * iterations
        tflops = (total_ops / elapsed_time) / 1e12
        
        results = {
            'matrix_size': aligned_size,
            'dtype': dtype,
            'iterations': iterations,
            'total_time': elapsed_time,
            'time_per_op': elapsed_time / iterations,
            'tflops': tflops
        }
        
        logger.info(f"Benchmark results: {tflops:.2f} TFLOPS")
        
        return results


def verify_tensor_core_support() -> bool:
    """
    Verifica soporte de Tensor Cores en el sistema.
    
    Returns:
        True si Tensor Cores están disponibles
    """
    optimizer = TensorCoreOptimizer()
    return optimizer.tensor_core_available


def optimize_layer_for_tensor_cores(
    layer: tf.keras.layers.Layer,
    config: Optional[TensorCoreConfig] = None
) -> tf.keras.layers.Layer:
    """
    Optimiza una capa para Tensor Cores.
    
    Args:
        layer: Capa a optimizar
        config: Configuración de optimización
        
    Returns:
        Capa optimizada
    """
    optimizer = TensorCoreOptimizer(config)
    return optimizer.optimize_layer(layer)