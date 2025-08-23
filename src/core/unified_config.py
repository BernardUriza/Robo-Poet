"""
Unified configuration system for Robo-Poet.

Consolidates all configuration management into a single, type-safe,
hierarchical configuration system with environment variable support.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Available logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class GPUBackend(str, Enum):
    """Available GPU backends."""
    CUDA = "cuda"
    ROCM = "rocm"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class SystemConfig:
    """System-level configuration."""
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Environment
    environment: str = "development"  # development, production, testing
    debug: bool = True
    seed: Optional[int] = 42
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_cpu_threads: int = 8
    temp_dir: str = "/tmp/robo-poet"
    
    def __post_init__(self):
        # Override with environment variables
        if env_log := os.getenv("ROBO_POET_LOG_LEVEL"):
            self.log_level = LogLevel(env_log.upper())
        if env_debug := os.getenv("ROBO_POET_DEBUG"):
            self.debug = env_debug.lower() in ("true", "1", "yes")
        if env_seed := os.getenv("ROBO_POET_SEED"):
            self.seed = int(env_seed)


@dataclass
class GPUConfig:
    """GPU optimization configuration."""
    
    # Hardware detection
    backend: GPUBackend = GPUBackend.AUTO
    device_id: int = 0
    require_gpu: bool = True
    
    # Memory management
    memory_growth: bool = True
    memory_limit_mb: Optional[int] = 7680  # 7.5GB for RTX 2000 Ada
    memory_fraction: float = 0.95
    
    # Mixed precision
    mixed_precision: bool = True
    precision_policy: str = "mixed_float16"
    loss_scale: str = "dynamic"
    
    # Tensor Cores
    use_tensor_cores: bool = True
    dimension_alignment: int = 8
    use_tf32: bool = True
    
    # Performance
    allow_soft_placement: bool = True
    log_device_placement: bool = False
    enable_xla: bool = False  # Disabled by default for WSL2 compatibility
    
    def __post_init__(self):
        # Override with environment variables
        if env_backend := os.getenv("ROBO_POET_GPU_BACKEND"):
            self.backend = GPUBackend(env_backend.lower())
        if env_require := os.getenv("ROBO_POET_REQUIRE_GPU"):
            self.require_gpu = env_require.lower() in ("true", "1", "yes")
        if env_memory := os.getenv("ROBO_POET_GPU_MEMORY"):
            self.memory_limit_mb = int(env_memory)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    model_type: str = "lstm"
    vocab_size: int = 10000
    embedding_dim: int = 128
    
    # LSTM specific
    lstm_units: List[int] = field(default_factory=lambda: [256, 256])
    dropout_rate: float = 0.3
    recurrent_dropout_rate: float = 0.0  # Disabled for CuDNN
    
    # Training
    sequence_length: int = 128
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    
    # Regularization
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Advanced features
    use_attention: bool = False
    use_layer_norm: bool = True
    
    def __post_init__(self):
        # Validate configuration
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert 0.0 <= self.dropout_rate <= 1.0, "dropout_rate must be in [0, 1]"
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert self.batch_size > 0, "batch_size must be positive"


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Input/Output
    corpus_path: str = ""
    output_dir: str = "output"
    cache_dir: str = "cache"
    
    # Preprocessing
    tokenization_strategy: str = "word_based"
    lowercase: bool = True
    remove_punctuation: bool = False
    min_token_frequency: int = 2
    
    # Data pipeline
    streaming: bool = True
    prefetch_buffer: int = 4
    shuffle_buffer: int = 10000
    num_parallel_calls: int = 4
    
    # Augmentation
    augmentation_probability: float = 0.1
    augmentation_techniques: List[str] = field(default_factory=lambda: [
        "synonym_replacement", "random_insertion", "random_swap"
    ])
    
    # Validation
    validation_split: float = 0.1
    test_split: float = 0.1
    cross_validation_folds: int = 5
    
    def __post_init__(self):
        # Create directories
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.cache_dir).mkdir(exist_ok=True)


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    
    # Generation parameters
    max_length: int = 200
    min_length: int = 10
    temperature: float = 1.0
    
    # Sampling methods
    sampling_method: str = "nucleus"  # greedy, temperature, top_k, nucleus, beam_search
    top_k: int = 50
    top_p: float = 0.9
    beam_width: int = 5
    
    # Quality control
    repetition_penalty: float = 1.1
    diversity_threshold: float = 0.7
    no_repeat_ngram_size: int = 3
    
    # Advanced features
    use_temperature_scheduling: bool = False
    scheduler_type: str = "linear"
    style_conditioning: Optional[str] = None
    
    def __post_init__(self):
        # Validate parameters
        assert self.max_length > self.min_length, "max_length must be > min_length"
        assert self.temperature > 0, "temperature must be positive"
        assert 0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.top_k > 0, "top_k must be positive"


@dataclass
class EvaluationConfig:
    """Evaluation and monitoring configuration."""
    
    # Metrics
    calculate_bleu: bool = True
    calculate_rouge: bool = True
    calculate_perplexity: bool = True
    calculate_diversity: bool = True
    
    # Monitoring
    real_time_evaluation: bool = True
    evaluation_frequency: int = 100  # batches
    save_samples: bool = True
    max_samples: int = 100
    
    # Dashboard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "logs/tensorboard"
    dashboard_port: int = 6006
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_metric: str = "loss"
    min_delta: float = 0.001


@dataclass
class UnifiedConfig:
    """Unified configuration for the entire Robo-Poet system."""
    
    system: SystemConfig = field(default_factory=SystemConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Metadata
    version: str = "2.1.0"
    config_source: str = "default"
    
    @classmethod
    def from_file(cls, config_path: str) -> "UnifiedConfig":
        """Load configuration from JSON file."""
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config instance
        config = cls()
        config.config_source = config_path
        
        # Update with file contents
        if "system" in config_dict:
            config.system = SystemConfig(**config_dict["system"])
        if "gpu" in config_dict:
            config.gpu = GPUConfig(**config_dict["gpu"])
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "generation" in config_dict:
            config.generation = GenerationConfig(**config_dict["generation"])
        if "evaluation" in config_dict:
            config.evaluation = EvaluationConfig(**config_dict["evaluation"])
        
        return config
    
    @classmethod
    def from_env(cls) -> "UnifiedConfig":
        """Load configuration from environment variables."""
        logger.info("Loading configuration from environment variables")
        
        config = cls()
        config.config_source = "environment"
        
        # Environment variables are handled in __post_init__ methods
        return config
    
    @classmethod
    def for_strategy(cls, strategy_name: str) -> "UnifiedConfig":
        """Create optimized configuration for specific strategy."""
        logger.info(f"Creating configuration for strategy: {strategy_name}")
        
        config = cls()
        config.config_source = f"strategy_{strategy_name}"
        
        if strategy_name == "gpu_optimization":
            # Optimize for GPU performance
            config.gpu.mixed_precision = True
            config.gpu.use_tensor_cores = True
            config.gpu.enable_xla = False  # WSL2 compatibility
            config.model.batch_size = 64
            config.data.prefetch_buffer = 8
            
        elif strategy_name == "generation_quality":
            # Optimize for generation quality
            config.generation.sampling_method = "nucleus"
            config.generation.top_p = 0.9
            config.generation.repetition_penalty = 1.2
            config.generation.use_temperature_scheduling = True
            
        elif strategy_name == "data_efficiency":
            # Optimize for data processing
            config.data.streaming = True
            config.data.augmentation_probability = 0.2
            config.data.prefetch_buffer = 16
            config.data.num_parallel_calls = 8
        
        elif strategy_name == "educational":
            # Simplified configuration for learning
            config.system.debug = True
            config.system.log_level = LogLevel.INFO
            config.model.epochs = 5
            config.model.batch_size = 16
            config.evaluation.save_samples = True
            config.evaluation.use_tensorboard = True
        
        return config
    
    def save(self, config_path: str):
        """Save configuration to JSON file."""
        logger.info(f"Saving configuration to {config_path}")
        
        config_dict = {
            "system": self.system.__dict__,
            "gpu": self.gpu.__dict__,
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "generation": self.generation.__dict__,
            "evaluation": self.evaluation.__dict__,
            "version": self.version,
            "config_source": self.config_source
        }
        
        # Convert enums to strings
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            else:
                return obj
        
        config_dict = convert_enums(config_dict)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate entire configuration and return any issues."""
        issues = []
        
        try:
            # System validation
            if not os.path.exists(self.system.temp_dir):
                issues.append(f"Temp directory does not exist: {self.system.temp_dir}")
            
            # GPU validation
            if self.gpu.require_gpu and self.gpu.backend == GPUBackend.CPU:
                issues.append("GPU required but backend is set to CPU")
            
            # Data validation
            if self.data.corpus_path and not os.path.exists(self.data.corpus_path):
                issues.append(f"Corpus path does not exist: {self.data.corpus_path}")
            
            # Model validation
            if self.model.batch_size > 256:
                issues.append("Batch size may be too large for RTX 2000 Ada (8GB VRAM)")
            
            # Generation validation
            if self.generation.beam_width > 10:
                issues.append("Large beam width may cause memory issues")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "version": self.version,
            "source": self.config_source,
            "system": {
                "environment": self.system.environment,
                "debug": self.system.debug,
                "log_level": self.system.log_level.value
            },
            "gpu": {
                "backend": self.gpu.backend.value,
                "mixed_precision": self.gpu.mixed_precision,
                "memory_limit_mb": self.gpu.memory_limit_mb
            },
            "model": {
                "type": self.model.model_type,
                "vocab_size": self.model.vocab_size,
                "batch_size": self.model.batch_size,
                "epochs": self.model.epochs
            },
            "generation": {
                "method": self.generation.sampling_method,
                "max_length": self.generation.max_length
            }
        }


# Global configuration instance
_config: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = UnifiedConfig()
    return _config


def set_config(config: UnifiedConfig):
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(config_path: Optional[str] = None) -> UnifiedConfig:
    """Load configuration from file or environment."""
    global _config
    
    if config_path and os.path.exists(config_path):
        _config = UnifiedConfig.from_file(config_path)
    else:
        _config = UnifiedConfig.from_env()
    
    # Validate configuration
    issues = _config.validate()
    if issues:
        logger.warning(f"Configuration issues found: {issues}")
    
    return _config


def create_strategy_config(strategy_name: str) -> UnifiedConfig:
    """Create and set configuration optimized for specific strategy."""
    global _config
    _config = UnifiedConfig.for_strategy(strategy_name)
    return _config


# Example usage and configuration presets
def demo_configurations():
    """Demonstrate different configuration setups."""
    
    # Educational configuration
    edu_config = UnifiedConfig.for_strategy("educational")
    print("Educational Config:")
    print(json.dumps(edu_config.get_summary(), indent=2))
    
    # Performance configuration
    perf_config = UnifiedConfig.for_strategy("gpu_optimization")
    print("\nGPU Optimization Config:")
    print(json.dumps(perf_config.get_summary(), indent=2))
    
    # Quality configuration
    quality_config = UnifiedConfig.for_strategy("generation_quality")
    print("\nGeneration Quality Config:")
    print(json.dumps(quality_config.get_summary(), indent=2))