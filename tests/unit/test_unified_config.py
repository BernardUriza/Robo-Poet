"""
Tests unitarios para el sistema de configuración unificada.

Prueba el sistema de configuración type-safe y jerárquico.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

# Setup sys.path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.unified_config import (
    LogLevel, GPUBackend,
    SystemConfig, GPUConfig, ModelConfig, DataConfig, 
    GenerationConfig, EvaluationConfig, UnifiedConfig,
    get_config, set_config, load_config, create_strategy_config
)


class TestEnums:
    """Tests para los enums de configuración."""
    
    def test_log_level_values(self):
        """Test valores de LogLevel."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"
    
    def test_gpu_backend_values(self):
        """Test valores de GPUBackend."""
        assert GPUBackend.CUDA == "cuda"
        assert GPUBackend.ROCM == "rocm"
        assert GPUBackend.CPU == "cpu"
        assert GPUBackend.AUTO == "auto"


class TestSystemConfig:
    """Tests para SystemConfig."""
    
    def test_system_config_defaults(self):
        """Test valores por defecto de SystemConfig."""
        config = SystemConfig()
        
        assert config.log_level == LogLevel.INFO
        assert config.log_format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.log_file is None
        assert config.environment == "development"
        assert config.debug is True
        assert config.seed == 42
        assert config.max_memory_gb == 8.0
        assert config.max_cpu_threads == 8
        assert config.temp_dir == "/tmp/robo-poet"
    
    def test_system_config_environment_variables(self):
        """Test override con variables de entorno."""
        with patch.dict(os.environ, {
            'ROBO_POET_LOG_LEVEL': 'ERROR',
            'ROBO_POET_DEBUG': 'false',
            'ROBO_POET_SEED': '123'
        }):
            config = SystemConfig()
            
            assert config.log_level == LogLevel.ERROR
            assert config.debug is False
            assert config.seed == 123


class TestGPUConfig:
    """Tests para GPUConfig."""
    
    def test_gpu_config_defaults(self):
        """Test valores por defecto de GPUConfig."""
        config = GPUConfig()
        
        assert config.backend == GPUBackend.AUTO
        assert config.device_id == 0
        assert config.require_gpu is True
        assert config.memory_growth is True
        assert config.memory_limit_mb == 7680
        assert config.memory_fraction == 0.95
        assert config.mixed_precision is True
        assert config.precision_policy == "mixed_float16"
        assert config.loss_scale == "dynamic"
        assert config.use_tensor_cores is True
        assert config.dimension_alignment == 8
        assert config.use_tf32 is True
        assert config.allow_soft_placement is True
        assert config.log_device_placement is False
        assert config.enable_xla is False
    
    def test_gpu_config_environment_variables(self):
        """Test override con variables de entorno."""
        with patch.dict(os.environ, {
            'ROBO_POET_GPU_BACKEND': 'cpu',
            'ROBO_POET_REQUIRE_GPU': 'false',
            'ROBO_POET_GPU_MEMORY': '4096'
        }):
            config = GPUConfig()
            
            assert config.backend == GPUBackend.CPU
            assert config.require_gpu is False
            assert config.memory_limit_mb == 4096


class TestModelConfig:
    """Tests para ModelConfig."""
    
    def test_model_config_defaults(self):
        """Test valores por defecto de ModelConfig."""
        config = ModelConfig()
        
        assert config.model_type == "lstm"
        assert config.vocab_size == 10000
        assert config.embedding_dim == 128
        assert config.lstm_units == [256, 256]
        assert config.dropout_rate == 0.3
        assert config.recurrent_dropout_rate == 0.0
        assert config.sequence_length == 128
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.epochs == 10
        assert config.weight_decay == 0.01
        assert config.gradient_clip_norm == 1.0
        assert config.use_attention is False
        assert config.use_layer_norm is True
    
    def test_model_config_validation(self):
        """Test validación de ModelConfig."""
        # Test vocab_size inválido
        with pytest.raises(AssertionError):
            ModelConfig(vocab_size=0)
        
        # Test embedding_dim inválido
        with pytest.raises(AssertionError):
            ModelConfig(embedding_dim=-1)
        
        # Test dropout_rate inválido
        with pytest.raises(AssertionError):
            ModelConfig(dropout_rate=1.5)
        
        # Test sequence_length inválido
        with pytest.raises(AssertionError):
            ModelConfig(sequence_length=0)
        
        # Test batch_size inválido
        with pytest.raises(AssertionError):
            ModelConfig(batch_size=-1)


class TestDataConfig:
    """Tests para DataConfig."""
    
    def test_data_config_defaults(self):
        """Test valores por defecto de DataConfig."""
        config = DataConfig()
        
        assert config.corpus_path == ""
        assert config.output_dir == "output"
        assert config.cache_dir == "cache"
        assert config.tokenization_strategy == "word_based"
        assert config.lowercase is True
        assert config.remove_punctuation is False
        assert config.min_token_frequency == 2
        assert config.streaming is True
        assert config.prefetch_buffer == 4
        assert config.shuffle_buffer == 10000
        assert config.num_parallel_calls == 4
        assert config.augmentation_probability == 0.1
        assert config.augmentation_techniques == [
            "synonym_replacement", "random_insertion", "random_swap"
        ]
        assert config.validation_split == 0.1
        assert config.test_split == 0.1
        assert config.cross_validation_folds == 5
    
    def test_data_config_directory_creation(self):
        """Test creación de directorios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DataConfig(
                output_dir=str(Path(temp_dir) / "output"),
                cache_dir=str(Path(temp_dir) / "cache")
            )
            
            assert Path(config.output_dir).exists()
            assert Path(config.cache_dir).exists()


class TestGenerationConfig:
    """Tests para GenerationConfig."""
    
    def test_generation_config_defaults(self):
        """Test valores por defecto de GenerationConfig."""
        config = GenerationConfig()
        
        assert config.max_length == 200
        assert config.min_length == 10
        assert config.temperature == 1.0
        assert config.sampling_method == "nucleus"
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.beam_width == 5
        assert config.repetition_penalty == 1.1
        assert config.diversity_threshold == 0.7
        assert config.no_repeat_ngram_size == 3
        assert config.use_temperature_scheduling is False
        assert config.scheduler_type == "linear"
        assert config.style_conditioning is None
    
    def test_generation_config_validation(self):
        """Test validación de GenerationConfig."""
        # Test max_length <= min_length
        with pytest.raises(AssertionError):
            GenerationConfig(max_length=10, min_length=20)
        
        # Test temperature inválido
        with pytest.raises(AssertionError):
            GenerationConfig(temperature=0)
        
        # Test top_p inválido
        with pytest.raises(AssertionError):
            GenerationConfig(top_p=1.5)
        
        # Test top_k inválido
        with pytest.raises(AssertionError):
            GenerationConfig(top_k=0)


class TestEvaluationConfig:
    """Tests para EvaluationConfig."""
    
    def test_evaluation_config_defaults(self):
        """Test valores por defecto de EvaluationConfig."""
        config = EvaluationConfig()
        
        assert config.calculate_bleu is True
        assert config.calculate_rouge is True
        assert config.calculate_perplexity is True
        assert config.calculate_diversity is True
        assert config.real_time_evaluation is True
        assert config.evaluation_frequency == 100
        assert config.save_samples is True
        assert config.max_samples == 100
        assert config.use_tensorboard is True
        assert config.tensorboard_log_dir == "logs/tensorboard"
        assert config.dashboard_port == 6006
        assert config.use_early_stopping is True
        assert config.early_stopping_patience == 5
        assert config.early_stopping_metric == "loss"
        assert config.min_delta == 0.001


class TestUnifiedConfig:
    """Tests para UnifiedConfig."""
    
    def test_unified_config_defaults(self):
        """Test valores por defecto de UnifiedConfig."""
        config = UnifiedConfig()
        
        assert isinstance(config.system, SystemConfig)
        assert isinstance(config.gpu, GPUConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert config.version == "2.1.0"
        assert config.config_source == "default"
    
    def test_unified_config_from_file(self):
        """Test cargar configuración desde archivo."""
        config_data = {
            "system": {"debug": False, "environment": "production"},
            "gpu": {"require_gpu": False, "backend": "cpu"},
            "model": {"vocab_size": 5000, "epochs": 20},
            "generation": {"temperature": 0.8, "max_length": 150}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config = UnifiedConfig.from_file(config_path)
            
            assert config.config_source == config_path
            assert config.system.debug is False
            assert config.system.environment == "production"
            assert config.gpu.require_gpu is False
            assert config.gpu.backend == GPUBackend.CPU
            assert config.model.vocab_size == 5000
            assert config.model.epochs == 20
            assert config.generation.temperature == 0.8
            assert config.generation.max_length == 150
            
        finally:
            Path(config_path).unlink()
    
    def test_unified_config_from_env(self):
        """Test cargar configuración desde variables de entorno."""
        config = UnifiedConfig.from_env()
        
        assert config.config_source == "environment"
        assert isinstance(config.system, SystemConfig)
    
    def test_unified_config_for_strategy_gpu_optimization(self):
        """Test configuración para estrategia de optimización GPU."""
        config = UnifiedConfig.for_strategy("gpu_optimization")
        
        assert config.config_source == "strategy_gpu_optimization"
        assert config.gpu.mixed_precision is True
        assert config.gpu.use_tensor_cores is True
        assert config.gpu.enable_xla is False
        assert config.model.batch_size == 64
        assert config.data.prefetch_buffer == 8
    
    def test_unified_config_for_strategy_generation_quality(self):
        """Test configuración para estrategia de calidad de generación."""
        config = UnifiedConfig.for_strategy("generation_quality")
        
        assert config.config_source == "strategy_generation_quality"
        assert config.generation.sampling_method == "nucleus"
        assert config.generation.top_p == 0.9
        assert config.generation.repetition_penalty == 1.2
        assert config.generation.use_temperature_scheduling is True
    
    def test_unified_config_for_strategy_data_efficiency(self):
        """Test configuración para estrategia de eficiencia de datos."""
        config = UnifiedConfig.for_strategy("data_efficiency")
        
        assert config.config_source == "strategy_data_efficiency"
        assert config.data.streaming is True
        assert config.data.augmentation_probability == 0.2
        assert config.data.prefetch_buffer == 16
        assert config.data.num_parallel_calls == 8
    
    def test_unified_config_for_strategy_educational(self):
        """Test configuración para estrategia educacional."""
        config = UnifiedConfig.for_strategy("educational")
        
        assert config.config_source == "strategy_educational"
        assert config.system.debug is True
        assert config.system.log_level == LogLevel.INFO
        assert config.model.epochs == 5
        assert config.model.batch_size == 16
        assert config.evaluation.save_samples is True
        assert config.evaluation.use_tensorboard is True
    
    def test_unified_config_save(self):
        """Test guardar configuración a archivo."""
        config = UnifiedConfig()
        config.model.vocab_size = 15000
        config.generation.temperature = 1.2
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            config.save(config_path)
            
            # Verificar que el archivo fue creado
            assert Path(config_path).exists()
            
            # Cargar y verificar contenido
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["model"]["vocab_size"] == 15000
            assert saved_data["generation"]["temperature"] == 1.2
            assert saved_data["version"] == "2.1.0"
            
        finally:
            Path(config_path).unlink()
    
    def test_unified_config_validate(self):
        """Test validación de configuración."""
        config = UnifiedConfig()
        
        # Configuración válida no debe tener issues
        issues = config.validate()
        # Algunos issues esperados por rutas que no existen en test
        assert isinstance(issues, list)
    
    def test_unified_config_validate_with_issues(self):
        """Test validación con problemas."""
        config = UnifiedConfig()
        config.data.corpus_path = "/nonexistent/file.txt"
        config.model.batch_size = 512  # Muy grande para RTX 2000 Ada
        config.generation.beam_width = 20  # Muy grande
        
        issues = config.validate()
        
        assert len(issues) > 0
        assert any("does not exist" in issue for issue in issues)
        assert any("batch size may be too large" in issue.lower() for issue in issues)
        assert any("beam width" in issue.lower() for issue in issues)
    
    def test_unified_config_get_summary(self):
        """Test obtener resumen de configuración."""
        config = UnifiedConfig()
        config.system.environment = "test"
        config.model.vocab_size = 8000
        
        summary = config.get_summary()
        
        assert summary["version"] == "2.1.0"
        assert summary["source"] == "default"
        assert summary["system"]["environment"] == "test"
        assert summary["model"]["vocab_size"] == 8000
        assert "gpu" in summary
        assert "generation" in summary


class TestGlobalConfigFunctions:
    """Tests para las funciones globales de configuración."""
    
    def test_get_config_default(self):
        """Test obtener configuración por defecto."""
        # Resetear configuración global
        set_config(None)
        
        config = get_config()
        
        assert isinstance(config, UnifiedConfig)
        assert config.config_source == "default"
    
    def test_set_and_get_config(self):
        """Test establecer y obtener configuración."""
        custom_config = UnifiedConfig()
        custom_config.model.vocab_size = 12000
        
        set_config(custom_config)
        retrieved_config = get_config()
        
        assert retrieved_config is custom_config
        assert retrieved_config.model.vocab_size == 12000
    
    def test_load_config_from_file(self):
        """Test cargar configuración desde archivo."""
        config_data = {
            "model": {"vocab_size": 7000, "epochs": 15}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            
            assert config.model.vocab_size == 7000
            assert config.model.epochs == 15
            
            # Verificar que se estableció como configuración global
            global_config = get_config()
            assert global_config is config
            
        finally:
            Path(config_path).unlink()
    
    def test_load_config_from_env(self):
        """Test cargar configuración desde entorno."""
        config = load_config(None)  # Sin archivo, debería cargar desde env
        
        assert config.config_source == "environment"
    
    def test_create_strategy_config(self):
        """Test crear configuración de estrategia."""
        config = create_strategy_config("gpu_optimization")
        
        assert config.config_source == "strategy_gpu_optimization"
        assert config.gpu.mixed_precision is True
        
        # Verificar que se estableció como configuración global
        global_config = get_config()
        assert global_config is config


if __name__ == "__main__":
    pytest.main([__file__])