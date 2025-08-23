"""
Data processing pipeline for professional text generation.

Strategy 6: Pipeline de Datos Profesional
- Streaming data with tf.data
- Prefetching and parallelization
- Data augmentation for text
- K-fold cross validation
- Optimized memory management
"""

from .streaming import (
    TextDataset,
    StreamingDataLoader,
    DatasetConfig,
    create_streaming_dataset
)

from .augmentation import (
    TextAugmenter,
    AugmentationConfig,
    AugmentationType,
    create_augmented_dataset
)

from .validation import (
    CrossValidator,
    ValidationConfig,
    ValidationResult,
    create_kfold_splits
)

from .preprocessing import (
    AdvancedPreprocessor,
    PreprocessingConfig,
    TokenizationStrategy,
    create_preprocessing_pipeline
)

from .memory_optimizer import (
    DataMemoryOptimizer,
    MemoryConfig,
    BatchOptimizer,
    optimize_dataset_memory
)

__all__ = [
    'TextDataset',
    'StreamingDataLoader', 
    'DatasetConfig',
    'create_streaming_dataset',
    'TextAugmenter',
    'AugmentationConfig',
    'AugmentationType', 
    'create_augmented_dataset',
    'CrossValidator',
    'ValidationConfig',
    'ValidationResult',
    'create_kfold_splits',
    'AdvancedPreprocessor',
    'PreprocessingConfig',
    'TokenizationStrategy',
    'create_preprocessing_pipeline',
    'DataMemoryOptimizer',
    'MemoryConfig',
    'BatchOptimizer',
    'optimize_dataset_memory'
]