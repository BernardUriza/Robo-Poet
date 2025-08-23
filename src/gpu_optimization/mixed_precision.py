"""
Mixed Precision Training para RTX 2000 Ada Generation.

Implementa entrenamiento con precisión mixta FP16/FP32 para
acelerar el entrenamiento y reducir el uso de memoria.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import mixed_precision


logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionPolicy:
    """Política de precisión mixta."""
    
    name: str = 'mixed_float16'
    compute_dtype: str = 'float16'
    variable_dtype: str = 'float32'
    
    # Loss scaling
    initial_scale: float = 2**15
    growth_steps: int = 2000
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    
    # Configuración específica RTX 2000 Ada
    use_tensor_cores: bool = True
    optimize_matmul: bool = True
    use_xla: bool = False  # XLA puede causar problemas en WSL2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            'name': self.name,
            'compute_dtype': self.compute_dtype,
            'variable_dtype': self.variable_dtype,
            'initial_scale': self.initial_scale,
            'growth_steps': self.growth_steps,
            'use_tensor_cores': self.use_tensor_cores
        }


class MixedPrecisionManager:
    """Gestor de precisión mixta para entrenamiento."""
    
    def __init__(self, policy: Optional[MixedPrecisionPolicy] = None):
        self.policy = policy or MixedPrecisionPolicy()
        self.original_policy = None
        self.loss_scale_optimizer = None
        self.is_enabled = False
        
        # Verificar soporte de GPU
        self.gpu_available = self._check_gpu_support()
        
        if self.gpu_available:
            logger.info("Mixed precision training available for RTX 2000 Ada")
        else:
            logger.warning("GPU not available - mixed precision disabled")
    
    def _check_gpu_support(self) -> bool:
        """Verifica soporte de GPU y capacidades."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return False
            
            # Verificar compute capability (RTX 2000 Ada tiene 8.9)
            # Ada Lovelace soporta FP8, FP16, TF32
            logger.info(f"Found {len(gpus)} GPU(s)")
            
            # Configurar opciones de GPU
            for gpu in gpus:
                try:
                    # Habilitar memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Obtener información del dispositivo
                    device_details = tf.config.experimental.get_device_details(gpu)
                    compute_capability = device_details.get('compute_capability', (0, 0))
                    
                    logger.info(
                        f"GPU: {gpu.name}, "
                        f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}"
                    )
                    
                    # RTX 2000 Ada tiene compute capability 8.9
                    if compute_capability[0] >= 7:  # Tensor Cores desde 7.0
                        logger.info("Tensor Cores support detected")
                    
                except Exception as e:
                    logger.warning(f"Could not configure GPU {gpu.name}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            return False
    
    def enable(self) -> bool:
        """Habilita precisión mixta."""
        if not self.gpu_available:
            logger.warning("Cannot enable mixed precision without GPU")
            return False
        
        try:
            # Guardar política original
            self.original_policy = mixed_precision.global_policy()
            
            # Establecer nueva política
            policy = mixed_precision.Policy(self.policy.name)
            mixed_precision.set_global_policy(policy)
            
            # Configurar opciones adicionales
            self._configure_matmul_precision()
            self._configure_tensor_core_usage()
            
            self.is_enabled = True
            
            logger.info(f"Mixed precision enabled: {policy.name}")
            logger.info(f"Compute dtype: {policy.compute_dtype}")
            logger.info(f"Variable dtype: {policy.variable_dtype}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable mixed precision: {e}")
            return False
    
    def disable(self) -> bool:
        """Deshabilita precisión mixta."""
        if self.original_policy:
            try:
                mixed_precision.set_global_policy(self.original_policy)
                self.is_enabled = False
                logger.info("Mixed precision disabled")
                return True
            except Exception as e:
                logger.error(f"Failed to disable mixed precision: {e}")
                return False
        return True
    
    def _configure_matmul_precision(self):
        """Configura precisión para operaciones matriciales."""
        if self.policy.optimize_matmul:
            # TF32 para multiplicaciones matriciales (mejor precisión que FP16)
            # Específico para GPUs Ampere y Ada Lovelace
            os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
            os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
            
            # Para RTX 2000 Ada - habilitar todas las optimizaciones
            os.environ['TF_ENABLE_CUDNN_FRONTEND'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '0'  # Más rápido, menos determinista
    
    def _configure_tensor_core_usage(self):
        """Configura uso de Tensor Cores."""
        if self.policy.use_tensor_cores:
            # Asegurar que las dimensiones sean múltiplos de 8 para Tensor Cores
            os.environ['TF_ENABLE_TENSOR_FLOAT_32_EXECUTION'] = '1'
            
            # Optimizaciones específicas para Ada Lovelace
            os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
    
    def create_loss_scale_optimizer(
        self, 
        optimizer: tf.keras.optimizers.Optimizer
    ) -> tf.keras.optimizers.Optimizer:
        """
        Crea optimizador con loss scaling para mixed precision.
        
        Args:
            optimizer: Optimizador base
            
        Returns:
            Optimizador con loss scaling
        """
        if not self.is_enabled:
            return optimizer
        
        # Para mixed precision, usar LossScaleOptimizer
        from tensorflow.keras.mixed_precision import LossScaleOptimizer
        
        self.loss_scale_optimizer = LossScaleOptimizer(
            optimizer,
            initial_scale=self.policy.initial_scale,
            dynamic_growth_steps=self.policy.growth_steps
        )
        
        logger.info(f"Created loss scale optimizer with initial scale: {self.policy.initial_scale}")
        
        return self.loss_scale_optimizer
    
    def wrap_model_for_mixed_precision(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Prepara modelo para mixed precision.
        
        Args:
            model: Modelo a preparar
            
        Returns:
            Modelo preparado
        """
        if not self.is_enabled:
            return model
        
        # Añadir capa de cast final si es necesario
        if model.output.dtype != tf.float32:
            from tensorflow.keras.layers import Activation
            
            # Asegurar que la salida sea float32
            outputs = Activation('linear', dtype='float32')(model.output)
            model = tf.keras.Model(inputs=model.input, outputs=outputs)
            
            logger.info("Added float32 cast layer to model output")
        
        return model
    
    def get_training_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """Obtiene callbacks para entrenamiento con mixed precision."""
        callbacks = []
        
        if self.is_enabled:
            # Callback para monitorear loss scale
            callbacks.append(LossScaleMonitorCallback(self))
            
            # Callback para ajustar batch size dinámicamente
            if self.policy.use_tensor_cores:
                callbacks.append(TensorCoreAlignmentCallback())
        
        return callbacks
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de rendimiento."""
        metrics = {
            'mixed_precision_enabled': self.is_enabled,
            'policy_name': self.policy.name if self.is_enabled else 'float32',
            'compute_dtype': self.policy.compute_dtype if self.is_enabled else 'float32',
            'tensor_cores_enabled': self.policy.use_tensor_cores and self.is_enabled
        }
        
        if self.loss_scale_optimizer and hasattr(self.loss_scale_optimizer, 'loss_scale'):
            metrics['current_loss_scale'] = float(self.loss_scale_optimizer.loss_scale)
        
        return metrics


class LossScaleMonitorCallback(tf.keras.callbacks.Callback):
    """Callback para monitorear loss scale durante entrenamiento."""
    
    def __init__(self, manager: MixedPrecisionManager):
        super().__init__()
        self.manager = manager
        self.loss_scale_history = []
    
    def on_batch_end(self, batch, logs=None):
        """Registra loss scale al final de cada batch."""
        if self.manager.loss_scale_optimizer:
            try:
                current_scale = float(self.manager.loss_scale_optimizer.loss_scale)
                self.loss_scale_history.append(current_scale)
                
                # Alertar si el scale es muy bajo
                if current_scale < 1.0:
                    logger.warning(f"Loss scale very low: {current_scale}")
                
                # Log cada 100 batches
                if batch % 100 == 0:
                    logger.debug(f"Current loss scale: {current_scale}")
                    
            except Exception as e:
                logger.debug(f"Could not get loss scale: {e}")


class TensorCoreAlignmentCallback(tf.keras.callbacks.Callback):
    """Callback para verificar alineación con Tensor Cores."""
    
    def __init__(self, alignment: int = 8):
        super().__init__()
        self.alignment = alignment
        self.warnings_issued = set()
    
    def on_train_begin(self, logs=None):
        """Verifica configuración del modelo para Tensor Cores."""
        if not self.model:
            return
        
        for layer in self.model.layers:
            self._check_layer_alignment(layer)
    
    def _check_layer_alignment(self, layer):
        """Verifica que las dimensiones sean óptimas para Tensor Cores."""
        layer_name = layer.name
        
        # Verificar Dense layers
        if isinstance(layer, tf.keras.layers.Dense):
            units = layer.units
            if units % self.alignment != 0:
                warning = f"Layer {layer_name}: units={units} not aligned to {self.alignment}"
                if warning not in self.warnings_issued:
                    logger.warning(warning)
                    logger.info(f"Consider using units={self._next_aligned(units)} for better performance")
                    self.warnings_issued.add(warning)
        
        # Verificar LSTM layers
        elif isinstance(layer, tf.keras.layers.LSTM):
            units = layer.units
            if units % self.alignment != 0:
                warning = f"LSTM {layer_name}: units={units} not aligned to {self.alignment}"
                if warning not in self.warnings_issued:
                    logger.warning(warning)
                    logger.info(f"Consider using units={self._next_aligned(units)} for Tensor Cores")
                    self.warnings_issued.add(warning)
    
    def _next_aligned(self, value: int) -> int:
        """Encuentra el siguiente valor alineado."""
        return ((value + self.alignment - 1) // self.alignment) * self.alignment


def configure_mixed_precision(
    use_mixed_precision: bool = True,
    policy_name: str = 'mixed_float16',
    use_xla: bool = False
) -> MixedPrecisionManager:
    """
    Configura mixed precision training de forma global.
    
    Args:
        use_mixed_precision: Si habilitar mixed precision
        policy_name: Nombre de la política ('mixed_float16', 'mixed_bfloat16')
        use_xla: Si habilitar compilación XLA
        
    Returns:
        Manager configurado
    """
    # Configurar XLA si se solicita
    if use_xla:
        tf.config.optimizer.set_jit(True)
        logger.info("XLA JIT compilation enabled")
    
    # Crear política
    policy = MixedPrecisionPolicy(
        name=policy_name,
        use_xla=use_xla
    )
    
    # Crear y configurar manager
    manager = MixedPrecisionManager(policy)
    
    if use_mixed_precision:
        success = manager.enable()
        if not success:
            logger.warning("Mixed precision could not be enabled")
    
    return manager


def create_mixed_precision_model(
    model_fn: callable,
    input_shape: Tuple[int, ...],
    **model_kwargs
) -> Tuple[tf.keras.Model, MixedPrecisionManager]:
    """
    Crea modelo con mixed precision configurado.
    
    Args:
        model_fn: Función que crea el modelo
        input_shape: Shape de entrada
        **model_kwargs: Argumentos para model_fn
        
    Returns:
        Tuple de (modelo, manager)
    """
    # Configurar mixed precision
    manager = configure_mixed_precision()
    
    # Crear modelo
    model = model_fn(input_shape=input_shape, **model_kwargs)
    
    # Preparar para mixed precision
    model = manager.wrap_model_for_mixed_precision(model)
    
    # Log configuración
    logger.info(f"Created mixed precision model with policy: {manager.policy.name}")
    logger.info(f"Model parameters: {model.count_params():,}")
    
    return model, manager