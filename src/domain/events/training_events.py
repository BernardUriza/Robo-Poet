"""
Domain events for training lifecycle.

Events represent significant business occurrences in the domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class DomainEvent:
    """Base class for all domain events."""
    event_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    aggregate_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.__class__.__name__,
            'timestamp': self.timestamp.isoformat(),
            'aggregate_id': self.aggregate_id
        }


@dataclass
class TrainingStarted(DomainEvent):
    """Event: Model training has started."""
    model_id: str = ""
    corpus_id: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    epochs: int = 0
    
    def __post_init__(self):
        self.aggregate_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'model_id': self.model_id,
            'corpus_id': self.corpus_id,
            'config': self.config,
            'epochs': self.epochs
        })
        return base


@dataclass
class TrainingEpochCompleted(DomainEvent):
    """Event: Training epoch has completed."""
    model_id: str = ""
    epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    
    def __post_init__(self):
        self.aggregate_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'model_id': self.model_id,
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'accuracy': self.accuracy,
            'learning_rate': self.learning_rate,
            'progress_percentage': (self.epoch / self.total_epochs) * 100 if self.total_epochs > 0 else 0
        })
        return base


@dataclass
class TrainingCompleted(DomainEvent):
    """Event: Model training has completed successfully."""
    model_id: str = ""
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    final_perplexity: Optional[float] = None
    best_val_loss: float = 0.0
    epochs_completed: int = 0
    duration_seconds: int = 0
    convergence_epoch: Optional[int] = None
    model_path: str = ""
    
    def __post_init__(self):
        self.aggregate_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'model_id': self.model_id,
            'final_loss': self.final_loss,
            'final_accuracy': self.final_accuracy,
            'final_perplexity': self.final_perplexity,
            'best_val_loss': self.best_val_loss,
            'epochs_completed': self.epochs_completed,
            'duration_seconds': self.duration_seconds,
            'convergence_epoch': self.convergence_epoch,
            'model_path': self.model_path,
            'duration_minutes': round(self.duration_seconds / 60, 2)
        })
        return base


@dataclass
class TrainingFailed(DomainEvent):
    """Event: Model training has failed."""
    model_id: str = ""
    error_message: str = ""
    error_type: str = ""
    epoch_failed: Optional[int] = None
    partial_loss: Optional[float] = None
    
    def __post_init__(self):
        self.aggregate_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'model_id': self.model_id,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'epoch_failed': self.epoch_failed,
            'partial_loss': self.partial_loss
        })
        return base


@dataclass
class ModelArchived(DomainEvent):
    """Event: Model has been archived."""
    model_id: str = ""
    reason: str = ""
    archived_by: str = "system"
    
    def __post_init__(self):
        self.aggregate_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'model_id': self.model_id,
            'reason': self.reason,
            'archived_by': self.archived_by
        })
        return base


@dataclass
class CorpusCreated(DomainEvent):
    """Event: New text corpus has been created."""
    corpus_id: str = ""
    name: str = ""
    size: int = 0
    language: str = "en"
    source_path: Optional[str] = None
    
    def __post_init__(self):
        self.aggregate_id = self.corpus_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'corpus_id': self.corpus_id,
            'name': self.name,
            'size': self.size,
            'language': self.language,
            'source_path': self.source_path
        })
        return base


@dataclass
class EarlyStoppingStopped(DomainEvent):
    """Event: Training stopped early due to early stopping."""
    model_id: str = ""
    stopped_at_epoch: int = 0
    patience: int = 0
    best_val_loss: float = 0.0
    final_val_loss: float = 0.0
    
    def __post_init__(self):
        self.aggregate_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'model_id': self.model_id,
            'stopped_at_epoch': self.stopped_at_epoch,
            'patience': self.patience,
            'best_val_loss': self.best_val_loss,
            'final_val_loss': self.final_val_loss,
            'improvement_threshold': abs(self.final_val_loss - self.best_val_loss)
        })
        return base