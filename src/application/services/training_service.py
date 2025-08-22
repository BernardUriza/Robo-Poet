"""
Training service for orchestrating training operations.

This service handles the coordination of domain objects to fulfill
training-related business use cases.
"""

from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from src.domain.entities.text_corpus import TextCorpus
from src.domain.entities.generation_model import GenerationModel, ModelStatus
from src.domain.value_objects.model_config import ModelConfig
from src.domain.events.training_events import (
    TrainingStarted, TrainingCompleted, TrainingFailed,
    CorpusCreated, ModelArchived
)
from src.domain.repositories import CorpusRepository, ModelRepository, UnitOfWork
from src.application.commands.training_commands import (
    CreateCorpusCommand, TrainModelCommand, StopTrainingCommand,
    ArchiveModelCommand, DeleteCorpusCommand, UpdateCorpusCommand
)


class TrainingService:
    """Service for training operations."""
    
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
    
    def create_corpus(self, command: CreateCorpusCommand) -> str:
        """Create a new text corpus."""
        command.validate()
        
        with self.uow:
            # Check for duplicate names
            existing = self.uow.corpus_repository.get_by_name(command.name)
            if existing:
                raise ValueError(f"Corpus with name '{command.name}' already exists")
            
            # Create corpus entity
            corpus = TextCorpus.create(
                name=command.name,
                content=command.content,
                source_path=command.source_path,
                language=command.language,
                tags=command.tags or []
            )
            
            # Store corpus
            self.uow.corpus_repository.add(corpus)
            
            # Publish domain event
            event = CorpusCreated(
                corpus_id=corpus.id,
                name=corpus.name,
                size=len(corpus.content),
                language=corpus.language,
                source_path=corpus.source_path
            )
            self.uow.event_repository.add(event)
            
            self.uow.commit()
            return corpus.id
    
    def train_model(self, command: TrainModelCommand) -> str:
        """Start training a new model."""
        command.validate()
        
        with self.uow:
            # Verify corpus exists
            corpus = self.uow.corpus_repository.get_by_id(command.corpus_id)
            if not corpus:
                raise ValueError(f"Corpus with ID '{command.corpus_id}' not found")
            
            # Check for duplicate model names
            existing = self.uow.model_repository.get_by_name(command.model_name)
            if existing:
                raise ValueError(f"Model with name '{command.model_name}' already exists")
            
            # Create model entity
            model = GenerationModel.create(
                name=command.model_name,
                corpus_id=command.corpus_id,
                config=command.config or ModelConfig(),
                description=command.description,
                tags=command.tags or []
            )
            
            # Store model
            self.uow.model_repository.add(model)
            
            # Publish training started event
            event = TrainingStarted(
                model_id=model.id,
                corpus_id=command.corpus_id,
                config=model.config.to_dict()
            )
            self.uow.event_repository.add(event)
            
            self.uow.commit()
            return model.id
    
    def complete_training(self, model_id: str, metrics: Dict[str, Any]) -> None:
        """Mark training as completed with metrics."""
        with self.uow:
            model = self.uow.model_repository.get_by_id(model_id)
            if not model:
                raise ValueError(f"Model with ID '{model_id}' not found")
            
            # Update model status
            model.complete_training(metrics)
            self.uow.model_repository.update(model)
            
            # Publish completion event
            event = TrainingCompleted(
                model_id=model_id,
                final_loss=metrics.get('final_loss', 0.0),
                epochs_completed=metrics.get('epochs_completed', 0),
                duration_seconds=int(metrics.get('training_time', 0.0))
            )
            self.uow.event_repository.add(event)
            
            self.uow.commit()
    
    def fail_training(self, model_id: str, error_message: str) -> None:
        """Mark training as failed."""
        with self.uow:
            model = self.uow.model_repository.get_by_id(model_id)
            if not model:
                raise ValueError(f"Model with ID '{model_id}' not found")
            
            # Update model status
            model.fail_training(error_message)
            self.uow.model_repository.update(model)
            
            # Publish failure event
            event = TrainingFailed(
                model_id=model_id,
                error_message=error_message
            )
            self.uow.event_repository.add(event)
            
            self.uow.commit()
    
    def stop_training(self, command: StopTrainingCommand) -> None:
        """Stop training of a model."""
        command.validate()
        
        with self.uow:
            model = self.uow.model_repository.get_by_id(command.model_id)
            if not model:
                raise ValueError(f"Model with ID '{command.model_id}' not found")
            
            if model.status != ModelStatus.TRAINING:
                raise ValueError(f"Model is not currently training (status: {model.status})")
            
            # Mark as failed with stop reason
            model.fail_training(f"Training stopped: {command.reason}")
            self.uow.model_repository.update(model)
            
            self.uow.commit()
    
    def archive_model(self, command: ArchiveModelCommand) -> None:
        """Archive a model."""
        command.validate()
        
        with self.uow:
            model = self.uow.model_repository.get_by_id(command.model_id)
            if not model:
                raise ValueError(f"Model with ID '{command.model_id}' not found")
            
            # Archive model
            model.archive(command.reason)
            self.uow.model_repository.update(model)
            
            # Publish archive event
            event = ModelArchived(
                model_id=command.model_id,
                reason=command.reason
            )
            self.uow.event_repository.add(event)
            
            self.uow.commit()
    
    def delete_corpus(self, command: DeleteCorpusCommand) -> None:
        """Delete a corpus."""
        command.validate()
        
        with self.uow:
            corpus = self.uow.corpus_repository.get_by_id(command.corpus_id)
            if not corpus:
                raise ValueError(f"Corpus with ID '{command.corpus_id}' not found")
            
            # Check for dependent models unless force deletion
            if not command.force:
                dependent_models = self.uow.model_repository.get_by_corpus_id(command.corpus_id)
                active_models = [m for m in dependent_models if m.status != ModelStatus.ARCHIVED]
                if active_models:
                    model_names = [m.name for m in active_models]
                    raise ValueError(
                        f"Cannot delete corpus: {len(active_models)} active models depend on it: {model_names}. "
                        f"Use force=True to delete anyway."
                    )
            
            # Delete corpus
            self.uow.corpus_repository.delete(command.corpus_id)
            self.uow.commit()
    
    def update_corpus(self, command: UpdateCorpusCommand) -> None:
        """Update corpus content or metadata."""
        command.validate()
        
        with self.uow:
            corpus = self.uow.corpus_repository.get_by_id(command.corpus_id)
            if not corpus:
                raise ValueError(f"Corpus with ID '{command.corpus_id}' not found")
            
            # Apply updates
            if command.new_content is not None:
                corpus.update_content(command.new_content)
            
            if command.new_name is not None:
                # Check for name conflicts
                existing = self.uow.corpus_repository.get_by_name(command.new_name)
                if existing and existing.id != command.corpus_id:
                    raise ValueError(f"Corpus with name '{command.new_name}' already exists")
                corpus.update_name(command.new_name)
            
            if command.add_tags:
                for tag in command.add_tags:
                    corpus.add_tag(tag)
            
            if command.remove_tags:
                for tag in command.remove_tags:
                    corpus.remove_tag(tag)
            
            # Update corpus
            self.uow.corpus_repository.update(corpus)
            self.uow.commit()
    
    def get_training_status(self, model_id: str) -> Dict[str, Any]:
        """Get current training status for a model."""
        with self.uow:
            model = self.uow.model_repository.get_by_id(model_id)
            if not model:
                raise ValueError(f"Model with ID '{model_id}' not found")
            
            return {
                'model_id': model.id,
                'name': model.name,
                'status': model.status.value,
                'created_at': model.created_at.isoformat(),
                'training_started_at': model.training_started_at.isoformat() if model.training_started_at else None,
                'training_completed_at': model.training_completed_at.isoformat() if model.training_completed_at else None,
                'metrics': model.training_metrics.to_dict() if model.training_metrics else None,
                'error_message': model.error_message
            }
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[Dict[str, Any]]:
        """List all models, optionally filtered by status."""
        with self.uow:
            if status:
                models = self.uow.model_repository.get_by_status(status)
            else:
                models = self.uow.model_repository.get_all()
            
            return [
                {
                    'id': model.id,
                    'name': model.name,
                    'status': model.status.value,
                    'corpus_id': model.corpus_id,
                    'created_at': model.created_at.isoformat(),
                    'description': model.description,
                    'tags': model.tags
                }
                for model in models
            ]
    
    def list_corpuses(self) -> List[Dict[str, Any]]:
        """List all corpuses."""
        with self.uow:
            corpuses = self.uow.corpus_repository.get_all()
            
            return [
                {
                    'id': corpus.id,
                    'name': corpus.name,
                    'size': len(corpus.content),
                    'language': corpus.language,
                    'created_at': corpus.created_at.isoformat(),
                    'tags': corpus.tags,
                    'source_path': corpus.source_path
                }
                for corpus in corpuses
            ]