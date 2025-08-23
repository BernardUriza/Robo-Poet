"""
Optimización GPU Profesional para RTX 2000 Ada Generation.

Módulo que implementa mixed precision training, Tensor Cores,
gestión de memoria dinámica y batch size adaptativo.
"""

from .mixed_precision import (
    MixedPrecisionManager, configure_mixed_precision,
    create_mixed_precision_model, MixedPrecisionPolicy
)
from .tensor_cores import (
    TensorCoreOptimizer, verify_tensor_core_support,
    optimize_layer_for_tensor_cores, TensorCoreConfig
)
from .memory_manager import (
    GPUMemoryManager, DynamicMemoryGrowth, 
    MemoryOptimizer, get_gpu_memory_info
)
from .adaptive_batch import (
    AdaptiveBatchSizeManager, BatchSizeOptimizer,
    DynamicBatchScheduler, find_optimal_batch_size
)
from .benchmark import (
    GPUBenchmark, BenchmarkResults, ConfigurationTester,
    run_comprehensive_benchmark
)

__all__ = [
    # Mixed Precision
    'MixedPrecisionManager', 'configure_mixed_precision',
    'create_mixed_precision_model', 'MixedPrecisionPolicy',
    
    # Tensor Cores
    'TensorCoreOptimizer', 'verify_tensor_core_support',
    'optimize_layer_for_tensor_cores', 'TensorCoreConfig',
    
    # Memory Management
    'GPUMemoryManager', 'DynamicMemoryGrowth',
    'MemoryOptimizer', 'get_gpu_memory_info',
    
    # Adaptive Batch
    'AdaptiveBatchSizeManager', 'BatchSizeOptimizer',
    'DynamicBatchScheduler', 'find_optimal_batch_size',
    
    # Benchmark
    'GPUBenchmark', 'BenchmarkResults', 'ConfigurationTester',
    'run_comprehensive_benchmark'
]