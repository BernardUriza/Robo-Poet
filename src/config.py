"""
Configuration module for neural text generation experiments.

This module contains all hyperparameters, model configurations, and system settings
following best practices for reproducible machine learning research.
"""

import os
from dataclasses import dataclass
from typing import Tuple, Optional
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
    epochs: int = 5
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Text processing
    max_text_length: int = 100_000
    step_size: int = 3  # Stride for sequence generation

@dataclass
class SystemConfig:
    """System and hardware configuration."""
    
    # GPU Configuration
    enable_mixed_precision: bool = False  # Disabled due to XLA issues
    force_cpu: bool = True  # Force CPU to avoid libdevice problems
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
        Configure GPU with memory growth and mixed precision.
        
        Returns:
            bool: True if GPU available and configured, False otherwise.
        """
        try:
            # Disable XLA to avoid libdevice issues
            os.environ['TF_XLA_FLAGS'] = ''
            os.environ['XLA_FLAGS'] = ''
            tf.config.optimizer.set_jit(False)
            
            gpus = tf.config.experimental.list_physical_devices('GPU')
            
            if gpus:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✅ GPU configured: {gpus[0].name}")
                return True
            else:
                print("💻 No GPU detected, using CPU")
                return False
                
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
            print("   Falling back to CPU")
            return False
    
    @staticmethod
    def get_device_strategy() -> str:
        """
        Determine optimal device strategy.
        
        Returns:
            str: Device string ('/GPU:0' or '/CPU:0')
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        # Force CPU due to XLA libdevice issues
        return '/CPU:0'

def get_config() -> Tuple[ModelConfig, SystemConfig]:
    """
    Factory function to create configuration objects.
    
    Returns:
        Tuple containing ModelConfig and SystemConfig instances.
    """
    return ModelConfig(), SystemConfig()