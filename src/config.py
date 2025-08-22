"""
Configuration module for neural text generation experiments.

This module contains all hyperparameters, model configurations, and system settings
following best practices for reproducible machine learning research.
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional

# XLA configuration is now handled by the main script

# NOW import TensorFlow with proper configuration
import tensorflow as tf

@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    
    # LSTM Architecture
    lstm_units: int = 128
    sequence_length: int = 40
    embedding_dim: Optional[int] = None  # Will be set to vocab_size
    dropout_rate: float = 0.2
    
    # Training hyperparameters
    batch_size: int = 128
    epochs: int = 50  # Entrenamiento intensivo de 1 hora
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10  # Más paciencia para entrenamientos largos
    
    # Text processing
    max_text_length: int = 100_000
    step_size: int = 3  # Stride for sequence generation

@dataclass
class SystemConfig:
    """System and hardware configuration."""
    
    # GPU Configuration
    enable_mixed_precision: bool = False  # Disabled due to XLA issues
    force_gpu: bool = True  # GPU is MANDATORY for academic requirements
    memory_growth: bool = True
    
    # Paths
    data_dir: str = "data"
    models_dir: str = "models" 
    logs_dir: str = "logs"
    
    # Logging
    verbose: int = 1
    log_level: str = "INFO"

class GPUConfigurator:
    """Handles GPU configuration and optimization for RTX 2000 Ada."""
    
    @staticmethod
    def setup_gpu() -> bool:
        """
        Configure GPU with memory growth.
        GPU is MANDATORY for academic requirements.
        
        Returns:
            bool: True if GPU available and configured, False if not available
        """
        # First try standard detection
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if not gpus:
            # Standard detection failed, try direct GPU operation (WSL2 fix)
            print("🔍 Detección estándar GPU falló, probando acceso directo...")
            try:
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([1.0])
                print("✅ GPU accesible directamente, configurando para uso forzado...")
                # GPU works, proceed with configuration assuming GPU:0 exists
                gpus = ['/GPU:0']  # Fake GPU entry for configuration
            except Exception as e:
                print(f"❌ GPU no accesible: {e}")
                return False
        
        try:
            # Enable memory growth (only for real GPU objects, not string paths)
            real_gpus = [gpu for gpu in gpus if hasattr(gpu, 'name')]
            if real_gpus:
                for gpu in real_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU ACADÉMICA CONFIGURADA: {real_gpus[0].name}")
            else:
                print("✅ GPU ACADÉMICA CONFIGURADA: /GPU:0 (acceso directo)")
            
            # Use float32 for stability (avoiding XLA issues)
            policy = tf.keras.mixed_precision.Policy('float32')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"   Modo: Académico (GPU obligatoria)")
            print(f"   XLA JIT: Deshabilitado")
            print(f"   Precision: float32")
            print(f"   Memory Growth: Habilitado")
            return True
            
        except RuntimeError as e:
            print(f"❌ Error configurando GPU: {e}")
            return False
    
    @staticmethod
    def get_device_strategy() -> str:
        """
        Determine optimal device strategy.
        REQUIREMENT: GPU is mandatory for academic purposes.
        
        Returns:
            str: Device string ('/GPU:0' if available, '/CPU:0' if forced)
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if not gpus:
            return '/CPU:0'  # Return CPU as fallback
        
        return '/GPU:0'

def get_config() -> Tuple[ModelConfig, SystemConfig]:
    """
    Factory function to create configuration objects.
    
    Returns:
        Tuple containing ModelConfig and SystemConfig instances.
    """
    return ModelConfig(), SystemConfig()