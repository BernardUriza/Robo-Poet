"""
Repository interfaces using Protocols for clean architecture.

These protocols define the contracts for data access without coupling to specific implementations.
"""

from typing import Protocol, List, Optional, Dict, Any
from abc import abstractmethod

from .entities.text_corpus import TextCorpus
from .entities.generation_model import GenerationModel
from .events.training_events import DomainEvent


class CorpusRepository(Protocol):
    """Protocol for corpus data access operations."""
    
    @abstractmethod
    def add(self, corpus: TextCorpus) -> None:
        """Add a new corpus to the repository."""
        ...
    
    @abstractmethod
    def get(self, corpus_id: str) -> Optional[TextCorpus]:
        """Get corpus by ID."""
        ...
    
    @abstractmethod
    def list(self, tags: Optional[List[str]] = None, limit: Optional[int] = None) -> List[TextCorpus]:
        """List corpora with optional filtering."""
        ...
    
    @abstractmethod
    def update(self, corpus: TextCorpus) -> None:
        """Update existing corpus."""
        ...
    
    @abstractmethod
    def delete(self, corpus_id: str) -> None:
        """Delete corpus by ID."""
        ...
    
    @abstractmethod
    def find_by_name(self, name: str) -> Optional[TextCorpus]:
        """Find corpus by name."""
        ...
    
    @abstractmethod
    def find_by_content_hash(self, content_hash: str) -> Optional[TextCorpus]:
        """Find corpus by content hash to detect duplicates."""
        ...


class ModelRepository(Protocol):
    """Protocol for model data access operations."""
    
    @abstractmethod
    def add(self, model: GenerationModel) -> None:
        """Add a new model to the repository."""
        ...
    
    @abstractmethod
    def get(self, model_id: str) -> Optional[GenerationModel]:
        """Get model by ID."""
        ...
    
    @abstractmethod
    def list(self, status: Optional[str] = None, limit: Optional[int] = None) -> List[GenerationModel]:
        """List models with optional status filtering."""
        ...
    
    @abstractmethod
    def update(self, model: GenerationModel) -> None:
        """Update existing model."""
        ...
    
    @abstractmethod
    def delete(self, model_id: str) -> None:
        """Delete model by ID."""
        ...
    
    @abstractmethod
    def find_by_corpus_id(self, corpus_id: str) -> List[GenerationModel]:
        """Find all models trained on a specific corpus."""
        ...
    
    @abstractmethod
    def find_by_name(self, name: str) -> Optional[GenerationModel]:
        """Find model by name."""
        ...
    
    @abstractmethod
    def find_ready_models(self) -> List[GenerationModel]:
        """Find all models ready for generation."""
        ...


class EventRepository(Protocol):
    """Protocol for domain event storage."""
    
    @abstractmethod
    def add(self, event: DomainEvent) -> None:
        """Store a domain event."""
        ...
    
    @abstractmethod
    def get_by_aggregate_id(self, aggregate_id: str) -> List[DomainEvent]:
        """Get all events for an aggregate."""
        ...
    
    @abstractmethod
    def get_by_type(self, event_type: str) -> List[DomainEvent]:
        """Get all events of a specific type."""
        ...
    
    @abstractmethod
    def get_recent(self, limit: int = 100) -> List[DomainEvent]:
        """Get most recent events."""
        ...


class UnitOfWork(Protocol):
    """Protocol for Unit of Work pattern."""
    
    # Repository instances
    corpus: CorpusRepository
    models: ModelRepository
    events: EventRepository
    
    @abstractmethod
    def commit(self) -> None:
        """Commit all changes in this unit of work."""
        ...
    
    @abstractmethod
    def rollback(self) -> None:
        """Rollback all changes in this unit of work."""
        ...
    
    @abstractmethod
    def __enter__(self):
        """Enter the unit of work context."""
        ...
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the unit of work context."""
        ...