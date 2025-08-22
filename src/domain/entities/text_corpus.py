"""
TextCorpus entity - represents a text corpus for training.

Domain entity with business rules and invariants.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import hashlib
import uuid


@dataclass
class TextCorpus:
    """Core entity representing a text corpus for model training."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    content: str = ""
    source_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Computed properties
    preprocessed: bool = False
    vocabulary_size: int = 0
    token_count: int = 0
    sequence_count: int = 0
    
    # Metadata
    language: str = "en"
    encoding: str = "utf-8"
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate invariants after creation."""
        if self.content:
            self.validate()
    
    def validate(self) -> None:
        """Validate domain invariants."""
        if not self.content:
            raise ValueError("Corpus content cannot be empty")
        
        if len(self.content) < 1000:
            raise ValueError("Corpus too small for training (minimum 1000 characters)")
        
        if not self.name:
            self.name = f"Corpus_{self.created_at.strftime('%Y%m%d_%H%M%S')}"
    
    def update_content(self, new_content: str) -> None:
        """Update corpus content with validation."""
        if not new_content:
            raise ValueError("Cannot update with empty content")
        
        self.content = new_content
        self.updated_at = datetime.now()
        self.preprocessed = False  # Reset preprocessing flag
        self.validate()
    
    def mark_preprocessed(self, vocab_size: int, token_count: int, sequence_count: int) -> None:
        """Mark corpus as preprocessed with statistics."""
        self.preprocessed = True
        self.vocabulary_size = vocab_size
        self.token_count = token_count
        self.sequence_count = sequence_count
        self.updated_at = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the corpus."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the corpus."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def content_hash(self) -> str:
        """Generate hash of content for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'source_path': self.source_path,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'preprocessed': self.preprocessed,
            'vocabulary_size': self.vocabulary_size,
            'token_count': self.token_count,
            'sequence_count': self.sequence_count,
            'language': self.language,
            'encoding': self.encoding,
            'tags': self.tags,
            'content_length': len(self.content),
            'content_hash': self.content_hash()
        }
    
    @classmethod
    def from_file(cls, filepath: str, name: Optional[str] = None) -> 'TextCorpus':
        """Create corpus from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read file {filepath}: {e}")
        
        return cls(
            name=name or f"Corpus from {filepath}",
            content=content,
            source_path=filepath
        )