"""
Tests unitarios para entidades del dominio.

Prueba las entidades del dominio según los principios de Domain-Driven Design.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

# Setup sys.path for imports
import sys
from pathlib import Path as PathlibPath
sys.path.insert(0, str(PathlibPath(__file__).parent.parent.parent / 'src'))

from domain.entities.text_corpus import TextCorpus
from domain.entities.generation_model import GenerationModel
from domain.value_objects.model_config import ModelConfig
from domain.value_objects.generation_params import GenerationParams
from core.exceptions import CorpusError, ModelError, ValidationError


class TestTextCorpus:
    """Tests unitarios para la entidad TextCorpus."""
    
    def test_create_text_corpus_with_valid_path(self):
        """Test crear corpus con ruta válida."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test corpus with multiple words.")
            temp_path = f.name
        
        try:
            corpus = TextCorpus(file_path=temp_path)
            assert corpus.file_path == temp_path
            assert corpus.size_bytes > 0
            assert corpus.encoding == 'utf-8'
            assert not corpus.is_processed
        finally:
            Path(temp_path).unlink()
    
    def test_create_text_corpus_with_invalid_path(self):
        """Test crear corpus con ruta inválida debe lanzar CorpusError."""
        with pytest.raises(CorpusError):
            TextCorpus(file_path="/nonexistent/file.txt")
    
    def test_text_corpus_process_content(self):
        """Test procesar contenido del corpus."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "Hello world! This is a test. Multiple sentences here."
            f.write(test_content)
            temp_path = f.name
        
        try:
            corpus = TextCorpus(file_path=temp_path)
            corpus.process_content()
            
            assert corpus.is_processed
            assert corpus.vocabulary_size > 0
            assert corpus.total_tokens > 0
            assert corpus.unique_tokens > 0
            assert len(corpus.sample_text) > 0
        finally:
            Path(temp_path).unlink()
    
    def test_text_corpus_statistics(self):
        """Test estadísticas del corpus."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = "word1 word2 word1 word3. Sentence two here!"
            f.write(test_content)
            temp_path = f.name
        
        try:
            corpus = TextCorpus(file_path=temp_path)
            corpus.process_content()
            
            stats = corpus.get_statistics()
            assert 'vocabulary_size' in stats
            assert 'total_tokens' in stats
            assert 'unique_tokens' in stats
            assert 'file_size_mb' in stats
            assert stats['vocabulary_size'] > 0
        finally:
            Path(temp_path).unlink()
    
    def test_text_corpus_validate_content(self):
        """Test validación de contenido del corpus."""
        # Test archivo vacío
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            empty_path = f.name
        
        try:
            with pytest.raises(CorpusError):
                corpus = TextCorpus(file_path=empty_path)
                corpus.validate_content()
        finally:
            Path(empty_path).unlink()
    
    def test_text_corpus_repr(self):
        """Test representación string del corpus."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            corpus = TextCorpus(file_path=temp_path)
            repr_str = repr(corpus)
            assert "TextCorpus" in repr_str
            assert temp_path in repr_str
        finally:
            Path(temp_path).unlink()


class TestGenerationModel:
    """Tests unitarios para la entidad GenerationModel."""
    
    def test_create_generation_model_with_valid_config(self):
        """Test crear modelo con configuración válida."""
        config = ModelConfig(
            vocab_size=1000,
            embedding_dim=128,
            lstm_units=[256, 256],
            dropout_rate=0.3,
            sequence_length=100
        )
        
        model = GenerationModel(config=config)
        assert model.config == config
        assert not model.is_trained
        assert not model.is_loaded
        assert model.training_history is None
        assert model.model_path is None
    
    def test_generation_model_build_architecture(self):
        """Test construcción de arquitectura del modelo."""
        config = ModelConfig(
            vocab_size=1000,
            embedding_dim=128,
            lstm_units=[256, 256],
            dropout_rate=0.3,
            sequence_length=100
        )
        
        model = GenerationModel(config=config)
        
        # Mock TensorFlow para evitar dependencias
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_sequential.return_value = mock_model
            
            with patch('tensorflow.keras.layers.Embedding'):
                with patch('tensorflow.keras.layers.LSTM'):
                    with patch('tensorflow.keras.layers.Dense'):
                        model.build_architecture()
                        
                        assert model.tf_model is not None
                        mock_sequential.assert_called_once()
    
    def test_generation_model_compile_model(self):
        """Test compilación del modelo."""
        config = ModelConfig(
            vocab_size=1000,
            embedding_dim=128,
            lstm_units=[256, 256],
            dropout_rate=0.3,
            sequence_length=100
        )
        
        model = GenerationModel(config=config)
        model.tf_model = Mock()  # Mock TensorFlow model
        
        model.compile_model(learning_rate=0.001)
        model.tf_model.compile.assert_called_once()
    
    def test_generation_model_save_and_load(self):
        """Test guardar y cargar modelo."""
        config = ModelConfig(
            vocab_size=1000,
            embedding_dim=128,
            lstm_units=[256, 256],
            dropout_rate=0.3,
            sequence_length=100
        )
        
        model = GenerationModel(config=config)
        model.tf_model = Mock()  # Mock TensorFlow model
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.keras"
            
            # Test save
            model.save(save_path)
            assert model.model_path == save_path
            model.tf_model.save.assert_called_once()
            
            # Test load
            new_model = GenerationModel(config=config)
            with patch('tensorflow.keras.models.load_model') as mock_load:
                mock_load.return_value = Mock()
                new_model.load(save_path)
                
                assert new_model.is_loaded
                assert new_model.model_path == save_path
                mock_load.assert_called_once()
    
    def test_generation_model_invalid_save_path(self):
        """Test guardar modelo con ruta inválida debe lanzar ModelError."""
        config = ModelConfig(
            vocab_size=1000,
            embedding_dim=128,
            lstm_units=[256, 256],
            dropout_rate=0.3,
            sequence_length=100
        )
        
        model = GenerationModel(config=config)
        
        with pytest.raises(ModelError):
            model.save("/nonexistent/directory/model.keras")
    
    def test_generation_model_get_summary(self):
        """Test obtener resumen del modelo."""
        config = ModelConfig(
            vocab_size=1000,
            embedding_dim=128,
            lstm_units=[256, 256],
            dropout_rate=0.3,
            sequence_length=100
        )
        
        model = GenerationModel(config=config)
        model.training_history = {'loss': [1.5, 1.2, 1.0], 'accuracy': [0.3, 0.5, 0.7]}
        
        summary = model.get_summary()
        
        assert 'config' in summary
        assert 'is_trained' in summary
        assert 'is_loaded' in summary
        assert 'training_metrics' in summary
        assert summary['config']['vocab_size'] == 1000


class TestModelConfig:
    """Tests unitarios para el value object ModelConfig."""
    
    def test_create_valid_model_config(self):
        """Test crear configuración válida del modelo."""
        config = ModelConfig(
            vocab_size=5000,
            embedding_dim=256,
            lstm_units=[512, 512],
            dropout_rate=0.2,
            sequence_length=128,
            batch_size=64,
            learning_rate=0.001,
            epochs=50
        )
        
        assert config.vocab_size == 5000
        assert config.embedding_dim == 256
        assert config.lstm_units == [512, 512]
        assert config.dropout_rate == 0.2
        assert config.sequence_length == 128
        assert config.batch_size == 64
        assert config.learning_rate == 0.001
        assert config.epochs == 50
    
    def test_model_config_validation_invalid_vocab_size(self):
        """Test validación falla con vocab_size inválido."""
        with pytest.raises(ValidationError):
            ModelConfig(vocab_size=0)
    
    def test_model_config_validation_invalid_dropout(self):
        """Test validación falla con dropout rate inválido."""
        with pytest.raises(ValidationError):
            ModelConfig(dropout_rate=1.5)
    
    def test_model_config_validation_invalid_learning_rate(self):
        """Test validación falla con learning rate inválido."""
        with pytest.raises(ValidationError):
            ModelConfig(learning_rate=0)
    
    def test_model_config_to_dict(self):
        """Test conversión a diccionario."""
        config = ModelConfig(
            vocab_size=1000,
            embedding_dim=128,
            lstm_units=[256, 256],
            dropout_rate=0.3,
            sequence_length=100
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 1000
        assert config_dict['embedding_dim'] == 128
        assert config_dict['lstm_units'] == [256, 256]
    
    def test_model_config_from_dict(self):
        """Test creación desde diccionario."""
        config_data = {
            'vocab_size': 2000,
            'embedding_dim': 128,
            'lstm_units': [256, 256],
            'dropout_rate': 0.4,
            'sequence_length': 80
        }
        
        config = ModelConfig.from_dict(config_data)
        assert config.vocab_size == 2000
        assert config.dropout_rate == 0.4


class TestGenerationParams:
    """Tests unitarios para el value object GenerationParams."""
    
    def test_create_valid_generation_params(self):
        """Test crear parámetros válidos de generación."""
        params = GenerationParams(
            seed_text="The power of",
            max_length=200,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        assert params.seed_text == "The power of"
        assert params.max_length == 200
        assert params.temperature == 0.8
        assert params.top_k == 50
        assert params.top_p == 0.9
        assert params.repetition_penalty == 1.1
    
    def test_generation_params_validation_invalid_temperature(self):
        """Test validación falla con temperature inválido."""
        with pytest.raises(ValidationError):
            GenerationParams(temperature=0)
    
    def test_generation_params_validation_invalid_max_length(self):
        """Test validación falla con max_length inválido."""
        with pytest.raises(ValidationError):
            GenerationParams(max_length=0)
    
    def test_generation_params_validation_invalid_top_p(self):
        """Test validación falla con top_p inválido."""
        with pytest.raises(ValidationError):
            GenerationParams(top_p=1.5)
    
    def test_generation_params_to_dict(self):
        """Test conversión a diccionario."""
        params = GenerationParams(
            seed_text="Test seed",
            max_length=100,
            temperature=1.0,
            top_k=40,
            top_p=0.8
        )
        
        params_dict = params.to_dict()
        assert isinstance(params_dict, dict)
        assert params_dict['seed_text'] == "Test seed"
        assert params_dict['max_length'] == 100
        assert params_dict['temperature'] == 1.0
    
    def test_generation_params_from_dict(self):
        """Test creación desde diccionario."""
        params_data = {
            'seed_text': "From dict",
            'max_length': 150,
            'temperature': 1.2,
            'top_k': 30,
            'top_p': 0.85
        }
        
        params = GenerationParams.from_dict(params_data)
        assert params.seed_text == "From dict"
        assert params.max_length == 150
        assert params.temperature == 1.2


if __name__ == "__main__":
    pytest.main([__file__])