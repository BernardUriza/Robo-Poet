"""
Commands for training operations.

Commands encapsulate the intent to perform training operations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.domain.value_objects.model_config import ModelConfig


@dataclass
class CreateCorpusCommand:
    """Command to create a new text corpus."""
    name: str
    content: str
    source_path: Optional[str] = None
    language: str = "en"
    tags: Optional[list] = None
    
    def validate(self) -> None:
        """Validate command parameters."""
        if not self.name.strip():
            raise ValueError("Corpus name cannot be empty")
        
        if not self.content.strip():
            raise ValueError("Corpus content cannot be empty")
        
        if len(self.content) < 1000:
            raise ValueError("Corpus content too short (minimum 1000 characters)")


@dataclass
class TrainModelCommand:
    """Command to train a new model."""
    model_name: str
    corpus_id: str
    config: Optional[ModelConfig] = None
    description: str = ""
    tags: Optional[list] = None
    
    def validate(self) -> None:
        """Validate command parameters."""
        if not self.model_name.strip():
            raise ValueError("Model name cannot be empty")
        
        if not self.corpus_id.strip():
            raise ValueError("Corpus ID cannot be empty")
        
        if self.config:
            self.config._validate()


@dataclass
class StopTrainingCommand:
    """Command to stop training of a model."""
    model_id: str
    reason: str = "User requested stop"
    
    def validate(self) -> None:
        """Validate command parameters."""
        if not self.model_id.strip():
            raise ValueError("Model ID cannot be empty")


@dataclass
class ArchiveModelCommand:
    """Command to archive a model."""
    model_id: str
    reason: str = "Manual archive"
    
    def validate(self) -> None:
        """Validate command parameters."""
        if not self.model_id.strip():
            raise ValueError("Model ID cannot be empty")


@dataclass
class DeleteCorpusCommand:
    """Command to delete a corpus."""
    corpus_id: str
    force: bool = False  # Delete even if models depend on it
    
    def validate(self) -> None:
        """Validate command parameters."""
        if not self.corpus_id.strip():
            raise ValueError("Corpus ID cannot be empty")


@dataclass
class UpdateCorpusCommand:
    """Command to update corpus content."""
    corpus_id: str
    new_content: Optional[str] = None
    new_name: Optional[str] = None
    add_tags: Optional[list] = None
    remove_tags: Optional[list] = None
    
    def validate(self) -> None:
        """Validate command parameters."""
        if not self.corpus_id.strip():
            raise ValueError("Corpus ID cannot be empty")
        
        if self.new_content is not None and len(self.new_content) < 1000:
            raise ValueError("New content too short (minimum 1000 characters)")
        
        # At least one field should be updated
        if all(field is None for field in [
            self.new_content, self.new_name, self.add_tags, self.remove_tags
        ]):
            raise ValueError("At least one field must be updated")