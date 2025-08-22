"""
GenerationModel entity - represents a trained text generation model.

Core domain entity with business rules for model lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid


class ModelStatus(Enum):
    """Model lifecycle status."""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    ARCHIVED = "archived"


class ModelType(Enum):
    """Model architecture type."""
    LSTM = "lstm"
    WEIGHT_DROPPED_LSTM = "weight_dropped_lstm"
    TRANSFORMER = "transformer"


@dataclass
class TrainingMetrics:
    """Training metrics value object."""
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    final_perplexity: Optional[float] = None
    best_val_loss: Optional[float] = None
    epochs_completed: int = 0
    training_duration_seconds: int = 0
    convergence_epoch: Optional[int] = None


@dataclass
class GenerationModel:
    """Core entity representing a trained text generation model."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    corpus_id: str = ""
    model_type: ModelType = ModelType.WEIGHT_DROPPED_LSTM
    status: ModelStatus = ModelStatus.CREATED
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    training_started_at: Optional[datetime] = None
    training_completed_at: Optional[datetime] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Training results
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    
    # File paths
    model_path: Optional[str] = None
    weights_path: Optional[str] = None
    vocabulary_path: Optional[str] = None
    
    # Metadata
    version: str = "1.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate invariants after creation."""
        if not self.name:
            self.name = f"Model_{self.created_at.strftime('%Y%m%d_%H%M%S')}"
        
        if not self.corpus_id:
            raise ValueError("Model must be associated with a corpus")
    
    def start_training(self) -> None:
        """Mark model as training started."""
        if self.status != ModelStatus.CREATED:
            raise ValueError(f"Cannot start training from status: {self.status}")
        
        self.status = ModelStatus.TRAINING
        self.training_started_at = datetime.now()
    
    def complete_training(self, metrics: TrainingMetrics, model_path: str) -> None:
        """Mark training as completed with results."""
        if self.status != ModelStatus.TRAINING:
            raise ValueError(f"Cannot complete training from status: {self.status}")
        
        self.status = ModelStatus.TRAINED
        self.training_completed_at = datetime.now()
        self.metrics = metrics
        self.model_path = model_path
        
        # Calculate training duration
        if self.training_started_at:
            duration = self.training_completed_at - self.training_started_at
            self.metrics.training_duration_seconds = int(duration.total_seconds())
    
    def fail_training(self, error_message: str) -> None:
        """Mark training as failed."""
        if self.status != ModelStatus.TRAINING:
            raise ValueError(f"Cannot fail training from status: {self.status}")
        
        self.status = ModelStatus.FAILED
        self.description = f"Training failed: {error_message}"
    
    def archive(self) -> None:
        """Archive the model."""
        if self.status not in [ModelStatus.TRAINED, ModelStatus.FAILED]:
            raise ValueError(f"Cannot archive model with status: {self.status}")
        
        self.status = ModelStatus.ARCHIVED
    
    def is_ready_for_generation(self) -> bool:
        """Check if model is ready for text generation."""
        return (
            self.status == ModelStatus.TRAINED and
            self.model_path is not None and
            self.metrics.final_loss is not None
        )
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update model configuration."""
        if self.status == ModelStatus.TRAINING:
            raise ValueError("Cannot update config during training")
        
        self.config.update(config)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def get_training_duration(self) -> Optional[int]:
        """Get training duration in seconds."""
        if self.training_started_at and self.training_completed_at:
            duration = self.training_completed_at - self.training_started_at
            return int(duration.total_seconds())
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'corpus_id': self.corpus_id,
            'model_type': self.model_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'training_started_at': self.training_started_at.isoformat() if self.training_started_at else None,
            'training_completed_at': self.training_completed_at.isoformat() if self.training_completed_at else None,
            'config': self.config,
            'metrics': {
                'final_loss': self.metrics.final_loss,
                'final_accuracy': self.metrics.final_accuracy,
                'final_perplexity': self.metrics.final_perplexity,
                'best_val_loss': self.metrics.best_val_loss,
                'epochs_completed': self.metrics.epochs_completed,
                'training_duration_seconds': self.metrics.training_duration_seconds,
                'convergence_epoch': self.metrics.convergence_epoch
            },
            'model_path': self.model_path,
            'weights_path': self.weights_path,
            'vocabulary_path': self.vocabulary_path,
            'version': self.version,
            'description': self.description,
            'tags': self.tags,
            'is_ready': self.is_ready_for_generation()
        }
    
    @classmethod
    def create_for_training(cls, name: str, corpus_id: str, config: Dict[str, Any]) -> 'GenerationModel':
        """Factory method to create model for training."""
        return cls(
            name=name,
            corpus_id=corpus_id,
            config=config,
            model_type=ModelType.WEIGHT_DROPPED_LSTM  # Default to our advanced architecture
        )