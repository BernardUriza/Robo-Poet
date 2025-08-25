"""
CLI Application Service

Orchestrates high-level CLI operations following Clean Architecture principles.
Replaces the monolithic orchestrator with proper separation of concerns.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

from src.domain.value_objects.model_config import ModelConfig
from src.domain.value_objects.generation_params import GenerationParams
from src.application.commands.training_commands import TrainModelCommand, CreateCorpusCommand
from src.application.services.training_service import TrainingService
from src.application.services.generation_service import GenerationService
from src.config.settings import Settings, get_cached_settings
from src.core.exceptions import RoboPoetError


@dataclass
class CLIConfig:
    """Configuration for CLI operations."""
    text_file: Optional[str] = None
    epochs: Optional[int] = None
    model_file: Optional[str] = None
    seed_text: str = "The power of"
    temperature: float = 0.8
    length: int = 200


class CLIApplicationService:
    """
    Application service for CLI operations.
    
    Coordinates domain services to fulfill CLI use cases while maintaining
    separation of concerns and proper error handling.
    """
    
    def __init__(
        self,
        training_service: TrainingService,
        generation_service: GenerationService,
        settings: Optional[Settings] = None
    ):
        self.training_service = training_service
        self.generation_service = generation_service
        self.settings = settings or get_cached_settings()
        self.logger = logging.getLogger(__name__)
    
    def train_model_from_file(self, config: CLIConfig) -> str:
        """
        Train a model from text file.
        
        Args:
            config: CLI configuration with text file and training params
            
        Returns:
            Model ID of the trained model
            
        Raises:
            RoboPoetError: When training fails or file is invalid
        """
        try:
            if not config.text_file:
                raise ValueError("Text file is required for training")
            
            # Create corpus from file
            corpus_command = CreateCorpusCommand(
                name=f"Corpus from {config.text_file}",
                content=self._read_text_file(config.text_file),
                source_path=config.text_file
            )
            corpus_id = self.training_service.create_corpus(corpus_command)
            
            # Configure model training
            model_config = self._create_model_config(config)
            
            # Train model
            train_command = TrainModelCommand(
                model_name=f"Model from {config.text_file}",
                corpus_id=corpus_id,
                config=model_config,
                description=f"Model trained on {config.text_file} with {config.epochs} epochs"
            )
            
            model_id = self.training_service.train_model(train_command)
            
            self.logger.info(f"Successfully trained model {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise RoboPoetError(
                message=f"Failed to train model: {e}",
                category="training",
                context={"config": config}
            ) from e
    
    def generate_text(self, config: CLIConfig) -> str:
        """
        Generate text using specified model.
        
        Args:
            config: CLI configuration with generation parameters
            
        Returns:
            Generated text
            
        Raises:
            RoboPoetError: When generation fails or model not found
        """
        try:
            if not config.model_file:
                raise ValueError("Model file is required for generation")
            
            # Create generation parameters
            generation_params = GenerationParams(
                seed_text=config.seed_text,
                length=config.length,
                temperature=config.temperature
            )
            
            # Generate text
            result = self.generation_service.generate_text(
                model_path=config.model_file,
                params=generation_params
            )
            
            self.logger.info("Text generation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise RoboPoetError(
                message=f"Failed to generate text: {e}",
                category="generation", 
                context={"config": config}
            ) from e
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for CLI display.
        
        Returns:
            Dictionary with system information
        """
        try:
            return {
                "gpu_available": self._check_gpu_availability(),
                "model_config": self.settings.model.to_dict(),
                "storage_paths": {
                    "models": str(self.settings.storage.models_dir),
                    "corpus": str(self.settings.storage.corpus_dir),
                    "logs": str(self.settings.storage.logs_dir)
                },
                "training_config": {
                    "default_epochs": self.settings.training.default_epochs,
                    "default_batch_size": self.settings.training.default_batch_size,
                    "early_stopping": self.settings.training.early_stopping_patience
                }
            }
        except Exception as e:
            self.logger.warning(f"Could not get full system status: {e}")
            return {"status": "partial", "error": str(e)}
    
    def _read_text_file(self, file_path: str) -> str:
        """Read and validate text file."""
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
            
        content = path.read_text(encoding='utf-8')
        
        if len(content) < 1000:
            raise ValueError(f"Text file too small (minimum 1000 characters): {len(content)}")
            
        return content
    
    def _create_model_config(self, config: CLIConfig) -> ModelConfig:
        """Create model configuration from CLI config."""
        return ModelConfig(
            epochs=config.epochs or self.settings.training.default_epochs,
            batch_size=self.settings.training.default_batch_size,
            vocab_size=self.settings.model.vocab_size,
            sequence_length=self.settings.model.sequence_length,
            lstm_units=self.settings.model.lstm_units,
            embedding_dim=self.settings.model.embedding_dim,
            variational_dropout_rate=self.settings.model.dropout_rate
        )
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
        except ImportError:
            return False
        except Exception:
            return False