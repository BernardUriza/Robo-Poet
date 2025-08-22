"""
SQLAlchemy ORM models for persistence.

Maps domain entities to database tables.
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, List

from src.domain.entities.text_corpus import TextCorpus
from src.domain.entities.generation_model import GenerationModel, ModelStatus, ModelType, TrainingMetrics
from src.domain.events.training_events import DomainEvent

Base = declarative_base()


class CorpusORM(Base):
    """SQLAlchemy model for TextCorpus."""
    
    __tablename__ = 'corpora'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Computed properties
    preprocessed = Column(Boolean, default=False)
    vocabulary_size = Column(Integer, default=0)
    token_count = Column(Integer, default=0)
    sequence_count = Column(Integer, default=0)
    
    # Metadata
    language = Column(String, default='en')
    encoding = Column(String, default='utf-8')
    tags = Column(JSON, default=list)
    
    # Relationships
    models = relationship("ModelORM", back_populates="corpus")
    
    def to_domain(self) -> TextCorpus:
        """Convert ORM model to domain entity."""
        return TextCorpus(
            id=self.id,
            name=self.name,
            content=self.content,
            source_path=self.source_path,
            created_at=self.created_at,
            updated_at=self.updated_at,
            preprocessed=self.preprocessed,
            vocabulary_size=self.vocabulary_size,
            token_count=self.token_count,
            sequence_count=self.sequence_count,
            language=self.language,
            encoding=self.encoding,
            tags=self.tags or []
        )
    
    @classmethod
    def from_domain(cls, corpus: TextCorpus) -> 'CorpusORM':
        """Create ORM model from domain entity."""
        return cls(
            id=corpus.id,
            name=corpus.name,
            content=corpus.content,
            source_path=corpus.source_path,
            created_at=corpus.created_at,
            updated_at=corpus.updated_at,
            preprocessed=corpus.preprocessed,
            vocabulary_size=corpus.vocabulary_size,
            token_count=corpus.token_count,
            sequence_count=corpus.sequence_count,
            language=corpus.language,
            encoding=corpus.encoding,
            tags=corpus.tags
        )


class ModelORM(Base):
    """SQLAlchemy model for GenerationModel."""
    
    __tablename__ = 'models'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    corpus_id = Column(String, ForeignKey('corpora.id'), nullable=False)
    model_type = Column(String, default=ModelType.WEIGHT_DROPPED_LSTM.value)
    status = Column(String, default=ModelStatus.CREATED.value)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    
    # Configuration and metrics as JSON
    config = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    
    # File paths
    model_path = Column(String, nullable=True)
    weights_path = Column(String, nullable=True)
    vocabulary_path = Column(String, nullable=True)
    
    # Metadata
    version = Column(String, default="1.0")
    description = Column(Text, default="")
    tags = Column(JSON, default=list)
    
    # Relationships
    corpus = relationship("CorpusORM", back_populates="models")
    
    def to_domain(self) -> GenerationModel:
        """Convert ORM model to domain entity."""
        metrics_data = self.metrics or {}
        metrics = TrainingMetrics(
            final_loss=metrics_data.get('final_loss'),
            final_accuracy=metrics_data.get('final_accuracy'),
            final_perplexity=metrics_data.get('final_perplexity'),
            best_val_loss=metrics_data.get('best_val_loss'),
            epochs_completed=metrics_data.get('epochs_completed', 0),
            training_duration_seconds=metrics_data.get('training_duration_seconds', 0),
            convergence_epoch=metrics_data.get('convergence_epoch')
        )
        
        return GenerationModel(
            id=self.id,
            name=self.name,
            corpus_id=self.corpus_id,
            model_type=ModelType(self.model_type),
            status=ModelStatus(self.status),
            created_at=self.created_at,
            training_started_at=self.training_started_at,
            training_completed_at=self.training_completed_at,
            config=self.config or {},
            metrics=metrics,
            model_path=self.model_path,
            weights_path=self.weights_path,
            vocabulary_path=self.vocabulary_path,
            version=self.version,
            description=self.description or "",
            tags=self.tags or []
        )
    
    @classmethod
    def from_domain(cls, model: GenerationModel) -> 'ModelORM':
        """Create ORM model from domain entity."""
        metrics_dict = {
            'final_loss': model.metrics.final_loss,
            'final_accuracy': model.metrics.final_accuracy,
            'final_perplexity': model.metrics.final_perplexity,
            'best_val_loss': model.metrics.best_val_loss,
            'epochs_completed': model.metrics.epochs_completed,
            'training_duration_seconds': model.metrics.training_duration_seconds,
            'convergence_epoch': model.metrics.convergence_epoch
        }
        
        return cls(
            id=model.id,
            name=model.name,
            corpus_id=model.corpus_id,
            model_type=model.model_type.value,
            status=model.status.value,
            created_at=model.created_at,
            training_started_at=model.training_started_at,
            training_completed_at=model.training_completed_at,
            config=model.config,
            metrics=metrics_dict,
            model_path=model.model_path,
            weights_path=model.weights_path,
            vocabulary_path=model.vocabulary_path,
            version=model.version,
            description=model.description,
            tags=model.tags
        )


class EventORM(Base):
    """SQLAlchemy model for DomainEvent."""
    
    __tablename__ = 'events'
    
    event_id = Column(String, primary_key=True)
    event_type = Column(String, nullable=False)
    aggregate_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    data = Column(JSON, nullable=False)
    
    def to_domain(self) -> Dict[str, Any]:
        """Convert to domain event data."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'aggregate_id': self.aggregate_id,
            'timestamp': self.timestamp,
            'data': self.data or {}
        }
    
    @classmethod
    def from_domain(cls, event: DomainEvent) -> 'EventORM':
        """Create ORM model from domain event."""
        return cls(
            event_id=event.event_id,
            event_type=event.__class__.__name__,
            aggregate_id=event.aggregate_id,
            timestamp=event.timestamp,
            data=event.to_dict()
        )