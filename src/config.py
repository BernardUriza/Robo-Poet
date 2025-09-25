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

@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    
    # LSTM Architecture - CORRECTED per CLAUDE.md specs
    lstm_units: int = 256  # FIXED: Was 128, must be 256 per academic specs
    sequence_length: int = 40
    embedding_dim: Optional[int] = None  # Will be set to vocab_size
    dropout_rate: float = 0.2
    
    # Training hyperparameters - STRATEGY 1.3: Intensive training
    batch_size: int = 64  # Strategy 1.4: Reduced from 128 to 64 for 5000-vocab + 8GB VRAM optimization
    epochs: int = 100  # Strategy 1.3: Increased from 50 to 100+ for intensive training
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 15  # Increased patience for longer training
    
    # Text processing - STRATEGY 1.1: Expanded vocabulary
    max_text_length: int = 50_000_000  # Strategy 1.5: Increased to 50MB for larger corpus
    step_size: int = 3  # Stride for sequence generation
    vocab_size: int = 5000  # Expanded from 44 chars to 5000 tokens
    tokenization: str = 'word'  # 'word' or 'char' level tokenization

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
        
        if not gpus:
            # Standard detection failed, try direct GPU operation (WSL2 fix)
            print("[SEARCH] Detección estándar GPU falló, probando acceso directo...")
            try:
                print("[OK] GPU accesible directamente, configurando para uso forzado...")
                # GPU works, proceed with configuration assuming GPU:0 exists
                gpus = ['/GPU:0']  # Fake GPU entry for configuration
            except Exception as e:
                print(f"[X] GPU no accesible: {e}")
                return False
        
        try:
            # Enable memory growth (only for real GPU objects, not string paths)
            real_gpus = [gpu for gpu in gpus if hasattr(gpu, 'name')]
            if real_gpus:
                for gpu in real_gpus:
                print(f"[OK] GPU ACADÉMICA CONFIGURADA: {real_gpus[0].name}")
            else:
                print("[OK] GPU ACADÉMICA CONFIGURADA: /GPU:0 (acceso directo)")
            
            # Use float32 for stability (avoiding XLA issues)
            
            print(f"   Modo: Académico (GPU obligatoria)")
            print(f"   XLA JIT: Deshabilitado")
            print(f"   Precision: float32")
            print(f"   Memory Growth: Habilitado")
            return True
            
        except RuntimeError as e:
            print(f"[X] Error configurando GPU: {e}")
            return False
    
    @staticmethod
    def get_device_strategy() -> str:
        """
        Obtiene string del dispositivo GPU.
        Sistema termina si GPU no está disponible.
        
        Returns:
            str: Always '/GPU:0' - terminates if GPU unavailable
        """
        
        if not gpus:
            # Verificar que CUDA_VISIBLE_DEVICES no esté vacío
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible == '':
                # CUDA_VISIBLE_DEVICES vacío = no GPU forzado
                print(" CUDA_VISIBLE_DEVICES=\"\" - GPU forzadamente deshabilitada")
                # TERMINAR SISTEMA - NO HAY FALLBACK
                print("\n" + "="*60)
                print(" ERROR CRÍTICO: GPU DESHABILITADA")
                print("="*60)
                print("\nEste proyecto REQUIERE GPU para cumplir requisitos académicos.")
                print("CUDA_VISIBLE_DEVICES está vacío - GPU forzadamente deshabilitada.")
                print("\n" + "="*60)
                import sys
                sys.exit(1)
            
            # Intentar detección WSL2 solo si GPU podría estar disponible
            try:
                print("[OK] GPU detectada via workaround WSL2")
                return '/GPU:0'
            except Exception as e:
                # TERMINAR SISTEMA - NO HAY FALLBACK
                print("\n" + "="*60)
                print(" ERROR CRÍTICO: GPU NO DISPONIBLE")
                print("="*60)
                print("\nEste proyecto REQUIERE GPU para cumplir requisitos académicos.")
                print("\nSoluciones:")
                print("1. Verificar driver NVIDIA: nvidia-smi")
                print("2. Activar entorno conda: conda activate robo-poet-gpu")
                print("\nSi estás en WSL2, asegúrate de:")
                print("- Tener Windows 11 o Windows 10 build 21H2+")
                print("- Driver NVIDIA actualizado en Windows (no en WSL2)")
                print("- nvidia-smi funciona desde WSL2")
                print("\n" + "="*60)
                
                # TERMINAR EJECUCIÓN
                import sys
                sys.exit(1)
        
        # GPU detectada exitosamente
        return '/GPU:0'

@dataclass
class Config:
    """Combined configuration object."""
    model: ModelConfig
    system: SystemConfig
    training: ModelConfig  # Alias for backward compatibility

def get_config() -> Config:
    """
    Factory function to create configuration objects.
    
    Returns:
        Config object with model and system configurations.
    """
    model_config = ModelConfig()
    system_config = SystemConfig()
    return Config(
        model=model_config,
        system=system_config,
        training=model_config  # Alias for training config
    )