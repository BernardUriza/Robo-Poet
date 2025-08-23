"""
Tests de integración para servicios de aplicación.

Prueba la integración entre servicios, repositorios y dominio.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Setup sys.path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from application.services.training_service import TrainingService
from application.services.generation_service import GenerationService
from application.commands.training_commands import StartTrainingCommand, SaveModelCommand
from domain.entities.text_corpus import TextCorpus
from domain.entities.generation_model import GenerationModel
from domain.value_objects.model_config import ModelConfig
from domain.value_objects.generation_params import GenerationParams
from domain.events.training_events import TrainingStartedEvent, TrainingCompletedEvent
from domain.events.generation_events import TextGeneratedEvent
from core.exceptions import ModelError, CorpusError, TrainingError


class TestTrainingServiceIntegration:
    """Tests de integración para TrainingService."""
    
    @pytest.fixture
    def mock_repositories(self):
        """Mock repositories para testing."""
        corpus_repo = Mock()
        model_repo = Mock()
        event_repo = Mock()
        return corpus_repo, model_repo, event_repo
    
    @pytest.fixture
    def mock_uow(self, mock_repositories):
        """Mock Unit of Work."""
        corpus_repo, model_repo, event_repo = mock_repositories
        
        uow = Mock()
        uow.corpus_repository = corpus_repo
        uow.model_repository = model_repo
        uow.event_repository = event_repo
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        
        return uow, mock_repositories
    
    @pytest.fixture
    def training_service(self, mock_uow):
        """TrainingService con dependencias mockeadas."""
        uow, _ = mock_uow
        return TrainingService(uow)
    
    @pytest.fixture
    def sample_corpus_file(self):
        """Archivo de corpus de ejemplo."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            content = """
            This is a sample text corpus for testing the training service.
            It contains multiple sentences and paragraphs to simulate real training data.
            The content is designed to test tokenization, preprocessing, and model training.
            We need enough content to make the training meaningful for testing purposes.
            """
            f.write(content)
            return f.name
    
    def test_start_training_integration(self, training_service, sample_corpus_file, mock_uow):
        """Test integración completa de inicio de entrenamiento."""
        uow, (corpus_repo, model_repo, event_repo) = mock_uow
        
        # Configurar mocks
        corpus_repo.get_by_path = AsyncMock(return_value=None)
        corpus_repo.add = AsyncMock()
        model_repo.add = AsyncMock()
        event_repo.add = AsyncMock()
        
        # Crear comando
        command = StartTrainingCommand(
            corpus_path=sample_corpus_file,
            config=ModelConfig(
                vocab_size=1000,
                embedding_dim=64,
                lstm_units=[128, 128],
                dropout_rate=0.3,
                sequence_length=50,
                epochs=2
            )
        )
        
        # Mock TensorFlow components
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=Mock(history={'loss': [1.5, 1.2]}))
            mock_sequential.return_value = mock_model
            
            with patch('tensorflow.keras.layers.Embedding'):
                with patch('tensorflow.keras.layers.LSTM'):
                    with patch('tensorflow.keras.layers.Dense'):
                        # Ejecutar entrenamiento
                        result = asyncio.run(training_service.start_training(command))
                        
                        # Verificar resultado
                        assert result is not None
                        assert 'training_id' in result
                        assert 'model_id' in result
                        
                        # Verificar llamadas a repositorios
                        corpus_repo.add.assert_called_once()
                        model_repo.add.assert_called_once()
                        event_repo.add.assert_called()
                        uow.commit.assert_called_once()
        
        # Cleanup
        Path(sample_corpus_file).unlink()
    
    def test_save_model_integration(self, training_service, mock_uow):
        """Test integración de guardado de modelo."""
        uow, (corpus_repo, model_repo, event_repo) = mock_uow
        
        # Mock modelo existente
        mock_model_entity = Mock()
        mock_model_entity.id = "test-model-id"
        mock_model_entity.tf_model = Mock()
        mock_model_entity.save = Mock()
        
        model_repo.get_by_id = AsyncMock(return_value=mock_model_entity)
        model_repo.update = AsyncMock()
        
        # Crear comando
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model.keras"
            command = SaveModelCommand(
                model_id="test-model-id",
                save_path=str(save_path)
            )
            
            # Ejecutar guardado
            result = asyncio.run(training_service.save_model(command))
            
            # Verificar resultado
            assert result is not None
            assert result['success'] is True
            assert 'save_path' in result
            
            # Verificar llamadas
            model_repo.get_by_id.assert_called_once_with("test-model-id")
            mock_model_entity.save.assert_called_once()
            model_repo.update.assert_called_once()
            uow.commit.assert_called_once()
    
    def test_training_error_handling(self, training_service, mock_uow):
        """Test manejo de errores durante entrenamiento."""
        uow, (corpus_repo, model_repo, event_repo) = mock_uow
        
        # Configurar error en corpus
        corpus_repo.get_by_path = AsyncMock(side_effect=CorpusError("Corpus error"))
        
        command = StartTrainingCommand(
            corpus_path="/nonexistent/file.txt",
            config=ModelConfig()
        )
        
        # Ejecutar y verificar error
        with pytest.raises(CorpusError):
            asyncio.run(training_service.start_training(command))
        
        # Verificar rollback
        uow.rollback.assert_called_once()
    
    def test_training_event_publishing(self, training_service, sample_corpus_file, mock_uow):
        """Test publicación de eventos durante entrenamiento."""
        uow, (corpus_repo, model_repo, event_repo) = mock_uow
        
        # Configurar mocks
        corpus_repo.get_by_path = AsyncMock(return_value=None)
        corpus_repo.add = AsyncMock()
        model_repo.add = AsyncMock()
        event_repo.add = AsyncMock()
        
        command = StartTrainingCommand(
            corpus_path=sample_corpus_file,
            config=ModelConfig(epochs=1)
        )
        
        # Mock training con eventos
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=Mock(history={'loss': [1.0]}))
            mock_sequential.return_value = mock_model
            
            with patch('tensorflow.keras.layers.Embedding'):
                with patch('tensorflow.keras.layers.LSTM'):
                    with patch('tensorflow.keras.layers.Dense'):
                        asyncio.run(training_service.start_training(command))
                        
                        # Verificar eventos publicados
                        calls = event_repo.add.call_args_list
                        event_types = [call[0][0].__class__.__name__ for call in calls]
                        
                        assert 'TrainingStartedEvent' in event_types
                        assert 'TrainingCompletedEvent' in event_types
        
        Path(sample_corpus_file).unlink()


class TestGenerationServiceIntegration:
    """Tests de integración para GenerationService."""
    
    @pytest.fixture
    def mock_uow_generation(self):
        """Mock Unit of Work para generación."""
        model_repo = Mock()
        event_repo = Mock()
        
        uow = Mock()
        uow.model_repository = model_repo
        uow.event_repository = event_repo
        uow.commit = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        
        return uow, (model_repo, event_repo)
    
    @pytest.fixture
    def generation_service(self, mock_uow_generation):
        """GenerationService con dependencias mockeadas."""
        uow, _ = mock_uow_generation
        return GenerationService(uow)
    
    @pytest.fixture
    def mock_trained_model(self):
        """Modelo entrenado mockeado."""
        model = Mock()
        model.id = "test-model-id"
        model.is_trained = True
        model.is_loaded = True
        model.config = ModelConfig(vocab_size=1000, sequence_length=50)
        
        # Mock TensorFlow model
        tf_model = Mock()
        tf_model.predict = Mock(return_value=[[0.1, 0.7, 0.2]])  # Mock predictions
        model.tf_model = tf_model
        
        return model
    
    def test_generate_text_integration(self, generation_service, mock_trained_model, mock_uow_generation):
        """Test integración completa de generación de texto."""
        uow, (model_repo, event_repo) = mock_uow_generation
        
        # Configurar mocks
        model_repo.get_by_id = AsyncMock(return_value=mock_trained_model)
        event_repo.add = AsyncMock()
        
        # Parámetros de generación
        params = GenerationParams(
            seed_text="The power of",
            max_length=50,
            temperature=0.8,
            top_k=40,
            top_p=0.9
        )
        
        # Mock tokenization and generation
        with patch('src.generation.samplers.NucleusSampler') as mock_sampler_class:
            mock_sampler = Mock()
            mock_sampler.sample = Mock(return_value=[1, 2, 3, 4, 5])  # Mock token ids
            mock_sampler_class.return_value = mock_sampler
            
            with patch('src.data.preprocessing.AdvancedPreprocessor') as mock_preprocessor_class:
                mock_preprocessor = Mock()
                mock_preprocessor.encode_text = Mock(return_value=[0, 1, 2])
                mock_preprocessor.decode_tokens = Mock(return_value="The power of knowledge is immense")
                mock_preprocessor_class.return_value = mock_preprocessor
                
                # Ejecutar generación
                result = asyncio.run(
                    generation_service.generate_text("test-model-id", params)
                )
                
                # Verificar resultado
                assert result is not None
                assert 'text' in result
                assert 'metadata' in result
                assert result['text'] == "The power of knowledge is immense"
                
                # Verificar llamadas
                model_repo.get_by_id.assert_called_once_with("test-model-id")
                event_repo.add.assert_called_once()
                uow.commit.assert_called_once()
    
    def test_generate_text_model_not_found(self, generation_service, mock_uow_generation):
        """Test generación con modelo no encontrado."""
        uow, (model_repo, event_repo) = mock_uow_generation
        
        model_repo.get_by_id = AsyncMock(return_value=None)
        
        params = GenerationParams(seed_text="Test")
        
        # Ejecutar y verificar error
        with pytest.raises(ModelError):
            asyncio.run(generation_service.generate_text("nonexistent-id", params))
    
    def test_generate_text_untrained_model(self, generation_service, mock_uow_generation):
        """Test generación con modelo no entrenado."""
        uow, (model_repo, event_repo) = mock_uow_generation
        
        # Modelo no entrenado
        untrained_model = Mock()
        untrained_model.is_trained = False
        untrained_model.is_loaded = False
        
        model_repo.get_by_id = AsyncMock(return_value=untrained_model)
        
        params = GenerationParams(seed_text="Test")
        
        # Ejecutar y verificar error
        with pytest.raises(ModelError):
            asyncio.run(generation_service.generate_text("untrained-id", params))
    
    def test_generation_event_publishing(self, generation_service, mock_trained_model, mock_uow_generation):
        """Test publicación de eventos durante generación."""
        uow, (model_repo, event_repo) = mock_uow_generation
        
        model_repo.get_by_id = AsyncMock(return_value=mock_trained_model)
        event_repo.add = AsyncMock()
        
        params = GenerationParams(seed_text="Test")
        
        # Mock generation components
        with patch('src.generation.samplers.NucleusSampler') as mock_sampler_class:
            mock_sampler = Mock()
            mock_sampler.sample = Mock(return_value=[1, 2, 3])
            mock_sampler_class.return_value = mock_sampler
            
            with patch('src.data.preprocessing.AdvancedPreprocessor') as mock_preprocessor_class:
                mock_preprocessor = Mock()
                mock_preprocessor.encode_text = Mock(return_value=[0])
                mock_preprocessor.decode_tokens = Mock(return_value="Generated text")
                mock_preprocessor_class.return_value = mock_preprocessor
                
                asyncio.run(generation_service.generate_text("test-model-id", params))
                
                # Verificar evento publicado
                event_repo.add.assert_called_once()
                event_call = event_repo.add.call_args[0][0]
                assert isinstance(event_call, TextGeneratedEvent)
    
    def test_batch_generation(self, generation_service, mock_trained_model, mock_uow_generation):
        """Test generación en lote."""
        uow, (model_repo, event_repo) = mock_uow_generation
        
        model_repo.get_by_id = AsyncMock(return_value=mock_trained_model)
        event_repo.add = AsyncMock()
        
        # Múltiples parámetros
        params_list = [
            GenerationParams(seed_text="The power", max_length=30),
            GenerationParams(seed_text="Knowledge is", max_length=40),
            GenerationParams(seed_text="Artificial intelligence", max_length=50)
        ]
        
        # Mock generation components
        with patch('src.generation.samplers.NucleusSampler') as mock_sampler_class:
            mock_sampler = Mock()
            mock_sampler.sample = Mock(return_value=[1, 2, 3])
            mock_sampler_class.return_value = mock_sampler
            
            with patch('src.data.preprocessing.AdvancedPreprocessor') as mock_preprocessor_class:
                mock_preprocessor = Mock()
                mock_preprocessor.encode_text = Mock(return_value=[0])
                mock_preprocessor.decode_tokens = Mock(side_effect=[
                    "Generated text 1", "Generated text 2", "Generated text 3"
                ])
                mock_preprocessor_class.return_value = mock_preprocessor
                
                # Ejecutar generación en lote
                results = asyncio.run(
                    generation_service.batch_generate("test-model-id", params_list)
                )
                
                # Verificar resultados
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert 'text' in result
                    assert result['text'] == f"Generated text {i + 1}"
                
                # Verificar eventos (uno por generación)
                assert event_repo.add.call_count == 3


class TestServiceLayerErrorHandling:
    """Tests de manejo de errores en la capa de servicios."""
    
    def test_training_service_database_error(self):
        """Test manejo de errores de base de datos en training service."""
        # Mock UoW que falla en commit
        uow = Mock()
        uow.commit = AsyncMock(side_effect=Exception("Database error"))
        uow.rollback = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        
        service = TrainingService(uow)
        
        # Configurar mocks básicos
        uow.corpus_repository = Mock()
        uow.model_repository = Mock()
        uow.event_repository = Mock()
        uow.corpus_repository.get_by_path = AsyncMock(return_value=None)
        uow.corpus_repository.add = AsyncMock()
        uow.model_repository.add = AsyncMock()
        uow.event_repository.add = AsyncMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            f.write("Test content")
            f.flush()
            
            command = StartTrainingCommand(
                corpus_path=f.name,
                config=ModelConfig(epochs=1)
            )
            
            # Mock TensorFlow
            with patch('tensorflow.keras.Sequential'):
                with patch('tensorflow.keras.layers.Embedding'):
                    with patch('tensorflow.keras.layers.LSTM'):
                        with patch('tensorflow.keras.layers.Dense'):
                            # Verificar que se maneja el error
                            with pytest.raises(Exception):
                                asyncio.run(service.start_training(command))
                            
                            # Verificar rollback
                            uow.rollback.assert_called_once()
    
    def test_generation_service_memory_error(self):
        """Test manejo de errores de memoria en generation service."""
        uow = Mock()
        uow.model_repository = Mock()
        uow.event_repository = Mock()
        uow.commit = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        
        service = GenerationService(uow)
        
        # Mock modelo que falla en generación
        model = Mock()
        model.is_trained = True
        model.is_loaded = True
        model.tf_model = Mock()
        model.tf_model.predict = Mock(side_effect=MemoryError("Out of memory"))
        
        uow.model_repository.get_by_id = AsyncMock(return_value=model)
        
        params = GenerationParams(seed_text="Test")
        
        # Verificar manejo del error
        with pytest.raises(MemoryError):
            asyncio.run(service.generate_text("test-id", params))


if __name__ == "__main__":
    pytest.main([__file__])