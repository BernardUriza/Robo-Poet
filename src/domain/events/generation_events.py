"""
Domain events for text generation lifecycle.

Events represent significant business occurrences in the generation domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

from .training_events import DomainEvent


@dataclass
class GenerationRequested(DomainEvent):
    """Event: Text generation has been requested."""
    request_id: str = ""
    model_id: str = ""
    prompt: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    user_id: str = "anonymous"
    
    def __post_init__(self):
        self.aggregate_id = self.request_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'request_id': self.request_id,
            'model_id': self.model_id,
            'prompt': self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            'prompt_length': len(self.prompt),
            'params': self.params,
            'user_id': self.user_id
        })
        return base


@dataclass
class GenerationCompleted(DomainEvent):
    """Event: Text generation has completed successfully."""
    request_id: str = ""
    model_id: str = ""
    generated_length: int = 0
    generation_time: float = 0.0
    tokens_per_second: Optional[float] = None
    
    def __post_init__(self):
        self.aggregate_id = self.request_id
        if self.generation_time > 0:
            self.tokens_per_second = self.generated_length / self.generation_time
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'request_id': self.request_id,
            'model_id': self.model_id,
            'generated_length': self.generated_length,
            'generation_time': self.generation_time,
            'tokens_per_second': self.tokens_per_second
        })
        return base


@dataclass
class GenerationFailed(DomainEvent):
    """Event: Text generation has failed."""
    request_id: str = ""
    model_id: str = ""
    error_message: str = ""
    error_type: str = ""
    partial_output: bool = False
    
    def __post_init__(self):
        self.aggregate_id = self.request_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'request_id': self.request_id,
            'model_id': self.model_id,
            'error_message': self.error_message,
            'error_type': self.error_type,
            'partial_output': self.partial_output
        })
        return base


@dataclass
class ModelLoaded(DomainEvent):
    """Event: Model has been loaded into memory for generation."""
    model_id: str = ""
    model_name: str = ""
    memory_usage_mb: Optional[float] = None
    load_time: float = 0.0
    
    def __post_init__(self):
        self.aggregate_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'model_id': self.model_id,
            'model_name': self.model_name,
            'memory_usage_mb': self.memory_usage_mb,
            'load_time': self.load_time
        })
        return base


@dataclass
class ModelUnloaded(DomainEvent):
    """Event: Model has been unloaded from memory."""
    model_id: str = ""
    model_name: str = ""
    memory_freed_mb: Optional[float] = None
    reason: str = "manual"
    
    def __post_init__(self):
        self.aggregate_id = self.model_id
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            'model_id': self.model_id,
            'model_name': self.model_name,
            'memory_freed_mb': self.memory_freed_mb,
            'reason': self.reason
        })
        return base