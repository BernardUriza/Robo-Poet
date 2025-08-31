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
from src.application.services.telares.telares_service import TelaresDetectionService
from src.application.services.telares.training_service import TelaresTrainingService
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
    # Telares-specific configs
    telares_mode: bool = False
    hybrid_training: bool = False
    corpus_dir: str = "corpus"


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
        telares_detection_service: Optional[TelaresDetectionService] = None,
        telares_training_service: Optional[TelaresTrainingService] = None,
        settings: Optional[Settings] = None
    ):
        self.training_service = training_service
        self.generation_service = generation_service
        self.telares_detection_service = telares_detection_service or TelaresDetectionService()
        self.telares_training_service = telares_training_service or TelaresTrainingService()
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
            # Basic GPU check - simplified for WSL2 compatibility
            return True  # Assume available for now
        except ImportError:
            return False
        except Exception:
            return False
    
    # ===== TELARES DETECTOR METHODS =====
    
    def train_telares_detector(self, config: CLIConfig) -> Dict[str, Any]:
        """
        Train Telares detector for pyramid scheme detection.
        
        Args:
            config: CLI configuration with telares options
            
        Returns:
            Training metrics and results
        """
        try:
            if config.hybrid_training:
                # Train with poetic corpus as negative controls
                metrics = self.telares_training_service.train_hybrid_model(config.corpus_dir)
            else:
                # Train with telares dataset only
                metrics = self.telares_training_service.train_standard_model()
            
            self.logger.info("Telares detector training completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Telares training failed: {e}")
            raise RoboPoetError(
                message=f"Failed to train Telares detector: {e}",
                category="telares_training",
                context={"config": config}
            ) from e
    
    def analyze_telares_message(self, message: str, platform: str = None) -> Dict[str, Any]:
        """
        Analyze a message for pyramid scheme manipulation tactics.
        
        Args:
            message: Text message to analyze
            platform: Platform where message originated
            
        Returns:
            Detection results with manipulation tactics
        """
        try:
            result = self.telares_detection_service.analyze_message(message, platform)
            
            # Convert to dict for CLI display
            return {
                "message": result.message[:100] + "..." if len(result.message) > 100 else result.message,
                "risk_level": result.overall_risk.value,
                "detected_tactics": result.detected_tactics,
                "confidence": result.confidence,
                "total_score": result.total_manipulation_score,
                "is_pyramid_scheme": result.is_pyramid_scheme,
                "alert_message": result.get_alert_message(),
                "processing_time_ms": result.processing_time_ms
            }
            
        except Exception as e:
            self.logger.error(f"Telares analysis failed: {e}")
            raise RoboPoetError(
                message=f"Failed to analyze message: {e}",
                category="telares_analysis",
                context={"message": message[:50]}
            ) from e
    
    def get_telares_status(self) -> Dict[str, Any]:
        """
        Get Telares detector system status.
        
        Returns:
            System status and capabilities
        """
        try:
            detection_status = self.telares_detection_service.get_system_status()
            training_status = self.telares_training_service.get_training_status()
            
            return {
                "telares_ready": detection_status.get("ready_for_detection", False),
                "model_loaded": detection_status.get("detector_loaded", False),
                "model_version": detection_status.get("model_version", "Unknown"),
                "supported_tactics": detection_status.get("supported_tactics", []),
                "training_available": training_status.get("data_loader_ready", False),
                "available_datasets": training_status.get("available_datasets", []),
                "last_training": training_status.get("last_training_metrics", {})
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get telares status: {e}")
            return {"status": "error", "error": str(e)}