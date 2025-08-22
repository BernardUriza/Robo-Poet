"""
SQLAlchemy implementation of ModelRepository.

Concrete implementation using SQLAlchemy for model data access.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from src.domain.entities.generation_model import GenerationModel, ModelStatus
from src.infrastructure.persistence.sqlalchemy_models import ModelORM


class SQLAlchemyModelRepository:
    """SQLAlchemy implementation of the model repository."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def add(self, model: GenerationModel) -> None:
        """Add a new model to the repository."""
        orm_model = ModelORM.from_domain(model)
        self.session.add(orm_model)
        # Note: commit is handled by UnitOfWork
    
    def get(self, model_id: str) -> Optional[GenerationModel]:
        """Get model by ID."""
        orm_model = self.session.query(ModelORM).filter(
            ModelORM.id == model_id
        ).first()
        
        return orm_model.to_domain() if orm_model else None
    
    def list(self, status: Optional[str] = None, limit: Optional[int] = None) -> List[GenerationModel]:
        """List models with optional status filtering."""
        query = self.session.query(ModelORM)
        
        # Filter by status if provided
        if status:
            query = query.filter(ModelORM.status == status)
        
        # Apply limit
        if limit:
            query = query.limit(limit)
        
        # Order by creation date (newest first)
        query = query.order_by(ModelORM.created_at.desc())
        
        orm_models = query.all()
        return [model.to_domain() for model in orm_models]
    
    def update(self, model: GenerationModel) -> None:
        """Update existing model."""
        orm_model = self.session.query(ModelORM).filter(
            ModelORM.id == model.id
        ).first()
        
        if not orm_model:
            raise ValueError(f"Model with ID {model.id} not found")
        
        # Update fields
        orm_model.name = model.name
        orm_model.corpus_id = model.corpus_id
        orm_model.model_type = model.model_type.value
        orm_model.status = model.status.value
        orm_model.training_started_at = model.training_started_at
        orm_model.training_completed_at = model.training_completed_at
        orm_model.config = model.config
        
        # Update metrics
        metrics_dict = {
            'final_loss': model.metrics.final_loss,
            'final_accuracy': model.metrics.final_accuracy,
            'final_perplexity': model.metrics.final_perplexity,
            'best_val_loss': model.metrics.best_val_loss,
            'epochs_completed': model.metrics.epochs_completed,
            'training_duration_seconds': model.metrics.training_duration_seconds,
            'convergence_epoch': model.metrics.convergence_epoch
        }
        orm_model.metrics = metrics_dict
        
        # Update paths and metadata
        orm_model.model_path = model.model_path
        orm_model.weights_path = model.weights_path
        orm_model.vocabulary_path = model.vocabulary_path
        orm_model.version = model.version
        orm_model.description = model.description
        orm_model.tags = model.tags
    
    def delete(self, model_id: str) -> None:
        """Delete model by ID."""
        orm_model = self.session.query(ModelORM).filter(
            ModelORM.id == model_id
        ).first()
        
        if orm_model:
            self.session.delete(orm_model)
    
    def find_by_corpus_id(self, corpus_id: str) -> List[GenerationModel]:
        """Find all models trained on a specific corpus."""
        orm_models = self.session.query(ModelORM).filter(
            ModelORM.corpus_id == corpus_id
        ).order_by(ModelORM.created_at.desc()).all()
        
        return [model.to_domain() for model in orm_models]
    
    def find_by_name(self, name: str) -> Optional[GenerationModel]:
        """Find model by name."""
        orm_model = self.session.query(ModelORM).filter(
            ModelORM.name == name
        ).first()
        
        return orm_model.to_domain() if orm_model else None
    
    def find_ready_models(self) -> List[GenerationModel]:
        """Find all models ready for generation."""
        orm_models = self.session.query(ModelORM).filter(
            ModelORM.status == ModelStatus.TRAINED.value,
            ModelORM.model_path.isnot(None)
        ).order_by(ModelORM.created_at.desc()).all()
        
        return [model.to_domain() for model in orm_models]
    
    def count_by_status(self, status: str) -> int:
        """Count models by status."""
        return self.session.query(ModelORM).filter(
            ModelORM.status == status
        ).count()
    
    def find_training_models(self) -> List[GenerationModel]:
        """Find all models currently training."""
        orm_models = self.session.query(ModelORM).filter(
            ModelORM.status == ModelStatus.TRAINING.value
        ).all()
        
        return [model.to_domain() for model in orm_models]
    
    def find_recent(self, limit: int = 10) -> List[GenerationModel]:
        """Find most recently created models."""
        orm_models = self.session.query(ModelORM).order_by(
            ModelORM.created_at.desc()
        ).limit(limit).all()
        
        return [model.to_domain() for model in orm_models]