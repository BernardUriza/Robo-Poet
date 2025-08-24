#!/usr/bin/env python3
"""
Robo-Poet Academic Framework - Clean Architecture Entry Point

Clean, maintainable entry point following Clean Architecture principles.
Replaces the monolithic orchestrator with proper separation of concerns.

Author: ML Academic Framework
Version: 2.1 - Clean Architecture
Hardware: Optimized for NVIDIA RTX 2000 Ada
"""

import sys
import os
from pathlib import Path

# Environment configuration for GPU support
def configure_gpu_environment():
    """Configure environment variables for optimal GPU performance."""
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix and conda_prefix != '/usr/local':
        os.environ['CUDA_HOME'] = conda_prefix
        lib_path = f'{conda_prefix}/lib:{conda_prefix}/lib64'
        existing_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f'{lib_path}:{existing_lib_path}' if existing_lib_path else lib_path
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def setup_python_path():
    """Add src directory to Python path for imports."""
    src_path = Path(__file__).parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> int:
    """Main entry point with clean error handling."""
    try:
        # Configure environment
        configure_gpu_environment()
        setup_python_path()
        
        # Import and run CLI controller
        from src.interface.cli_controller import CLIController
        
        # Check GPU availability and display status
        print("🎯 GPU funcionando correctamente" if check_gpu() else "⚠️ GPU not available, using CPU")
        
        # Run application
        controller = CLIController()
        return controller.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Falling back to legacy orchestrator...")
        try:
            from orchestrator import main as legacy_main
            return legacy_main()
        except ImportError:
            print("❌ Legacy orchestrator also unavailable")
            return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def check_gpu() -> bool:
    """Quick GPU availability check."""
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except:
        return False


if __name__ == "__main__":
    sys.exit(main())