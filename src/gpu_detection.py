#!/usr/bin/env python3
"""
GPU Detection and Configuration Module for WSL2 Environments

Advanced GPU detection specifically designed for WSL2 environments where
standard TensorFlow GPU detection often fails but direct operations work.

Author: ML Academic Framework
Version: 2.1 - WSL2 Optimization
"""

import os
import time
from typing import Tuple, Any


def configure_gpu_environment():
    """Configure optimal GPU environment variables before TensorFlow import."""
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        os.environ['CUDA_HOME'] = conda_prefix
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Force correct library paths, don't append potentially wrong existing ones
        lib_paths = [f'{conda_prefix}/lib', f'{conda_prefix}/lib64']
        # Only append system paths, not potentially wrong conda paths
        system_paths = ['/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu']
        lib_paths.extend(system_paths)
        clean_ld = ':'.join(lib_paths)
        os.environ['LD_LIBRARY_PATH'] = clean_ld
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        print(f"🔧 GPU environment FORZADO para: {conda_prefix}")
        print(f"🔧 CUDA_HOME forzado a: {os.environ.get('CUDA_HOME', 'NO_SET')}")
        print(f"🔧 LD_LIBRARY_PATH limpiado y reconfigurado")
        print(f"🔧 Nuevas rutas: {os.environ.get('LD_LIBRARY_PATH', '')[:120]}...")
    else:
        print("⚠️ CONDA_PREFIX no encontrado")


def detect_gpu_for_wsl2() -> Tuple[bool, Any]:
    """
    Advanced GPU detection specifically designed for WSL2 environments.
    
    WSL2 has known issues with standard TensorFlow GPU detection where:
    1. tf.config.list_physical_devices('GPU') returns empty list
    2. But direct GPU operations work perfectly
    3. This affects NVIDIA drivers in WSL2 environments
    
    Returns:
        tuple: (gpu_available: bool, tf_module: module)
    """
    print("🔧 Iniciando detección de GPU optimizada para WSL2...")
    
    # Ensure optimal logging for diagnosis
    original_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    
    try:
        import tensorflow as tf
        
        # Strategy 1: Standard TensorFlow detection
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            print(f"✅ GPU detectada vía método estándar: {len(tf_gpus)} GPU(s)")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_log_level
            return True, tf
        
        # Strategy 2: Direct GPU operation test (WSL2 fix)
        print("🔍 Método estándar falló, probando acceso directo GPU (fix WSL2)...")
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                result = tf.reduce_sum(test_tensor)
            
            print("🎯 ¡GPU funciona perfectamente via acceso directo!")
            print("💡 Aplicando workaround WSL2 para usar GPU")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_log_level
            return True, tf
        
        except Exception as gpu_error:
            print(f"🔴 Test directo GPU falló: {gpu_error}")
            print("🔄 Iniciando en modo CPU")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_log_level
            return False, tf
    
    except ImportError:
        print("❌ TensorFlow no está instalado")
        return False, None
    except Exception as e:
        print(f"❌ Error crítico en detección GPU: {e}")
        return False, None


def setup_mixed_precision_if_available(tf_module) -> bool:
    """Setup mixed precision training if GPU is available."""
    try:
        # Check if GPU is truly available for mixed precision
        gpus = tf_module.config.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth
            for gpu in gpus:
                tf_module.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision policy
            policy = tf_module.keras.mixed_precision.Policy('mixed_float16')
            tf_module.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision (FP16) activado para RTX 2000 Ada")
            return True
        else:
            print("⚠️ Mixed precision no disponible (no GPU detectada)")
            return False
    except Exception as e:
        print(f"⚠️ No se pudo activar mixed precision: {e}")
        return False


def validate_gpu_libraries() -> bool:
    """Validate that all required CUDA libraries are available."""
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if not conda_prefix:
        return False
    
    required_libs = [
        'libcudnn.so',
        'libcublas.so', 
        'libcufft.so',
        'libcurand.so',
        'libcusolver.so',
        'libcusparse.so'
    ]
    
    lib_dir = f"{conda_prefix}/lib"
    missing_libs = []
    
    for lib in required_libs:
        lib_path = f"{lib_dir}/{lib}"
        if not os.path.exists(lib_path):
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"⚠️ Librerías CUDA faltantes: {missing_libs}")
        print("💡 Ejecuta: conda install -c conda-forge cudnn libcublas libcufft libcurand libcusolver libcusparse -y")
        return False
    
    print("✅ Todas las librerías CUDA están disponibles")
    return True


def get_gpu_info() -> dict:
    """Get detailed GPU information for academic display."""
    info = {
        'gpu_available': False,
        'gpu_name': 'N/A',
        'driver_version': 'N/A',
        'cuda_version': 'N/A',
        'memory_total': 'N/A',
        'tf_version': 'N/A'
    }
    
    try:
        import subprocess
        import tensorflow as tf
        
        info['tf_version'] = tf.__version__
        
        # Try to get GPU info via nvidia-smi
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(', ')
            if len(gpu_data) >= 3:
                info['gpu_name'] = gpu_data[0]
                info['driver_version'] = gpu_data[1]
                info['memory_total'] = gpu_data[2]
                info['gpu_available'] = True
        
        # Get CUDA version
        cuda_result = subprocess.run(['nvcc', '--version'], 
                                   capture_output=True, text=True, timeout=5)
        if cuda_result.returncode == 0:
            for line in cuda_result.stdout.split('\n'):
                if 'release' in line.lower():
                    info['cuda_version'] = line.strip()
                    break
    
    except Exception as e:
        print(f"⚠️ No se pudo obtener información detallada de GPU: {e}")
    
    return info