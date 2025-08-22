"""
GenerationParams value object - immutable parameters for text generation.

Value object containing all parameters for text generation.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SamplingStrategy(Enum):
    """Text generation sampling strategies."""
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    NUCLEUS = "nucleus"  # Top-p
    BEAM_SEARCH = "beam_search"


@dataclass(frozen=True)
class GenerationParams:
    """Immutable value object for text generation parameters."""
    
    seed_text: str = "The power of"
    length: int = 200
    temperature: float = 0.8
    
    # Advanced sampling
    sampling_strategy: SamplingStrategy = SamplingStrategy.TEMPERATURE
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    beam_width: int = 5
    
    # Output control
    min_length: int = 10
    max_length: int = 1000
    stop_tokens: tuple = ("<END>", "<STOP>")
    
    # Repetition control
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    
    def __post_init__(self):
        """Validate generation parameters."""
        self._validate()
    
    def _validate(self):
        """Validate all generation parameters."""
        if not self.seed_text:
            raise ValueError("seed_text cannot be empty")
        
        if self.length <= 0:
            raise ValueError("length must be positive")
        
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        
        if self.min_length < 0:
            raise ValueError("min_length cannot be negative")
        
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        
        if self.length > self.max_length:
            raise ValueError("length cannot exceed max_length")
        
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive if specified")
        
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1 if specified")
        
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
    
    @classmethod
    def creative(cls, seed_text: str = "The power of", length: int = 200) -> 'GenerationParams':
        """Factory method for creative generation."""
        return cls(
            seed_text=seed_text,
            length=length,
            temperature=1.2,
            sampling_strategy=SamplingStrategy.NUCLEUS,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    @classmethod
    def conservative(cls, seed_text: str = "The power of", length: int = 200) -> 'GenerationParams':
        """Factory method for conservative generation."""
        return cls(
            seed_text=seed_text,
            length=length,
            temperature=0.5,
            sampling_strategy=SamplingStrategy.TOP_K,
            top_k=10,
            repetition_penalty=1.2
        )
    
    @classmethod
    def balanced(cls, seed_text: str = "The power of", length: int = 200) -> 'GenerationParams':
        """Factory method for balanced generation."""
        return cls(
            seed_text=seed_text,
            length=length,
            temperature=0.8,
            sampling_strategy=SamplingStrategy.TEMPERATURE,
            repetition_penalty=1.1
        )
    
    @classmethod
    def deterministic(cls, seed_text: str = "The power of", length: int = 200) -> 'GenerationParams':
        """Factory method for deterministic generation."""
        return cls(
            seed_text=seed_text,
            length=length,
            temperature=0.1,
            sampling_strategy=SamplingStrategy.GREEDY,
            repetition_penalty=1.0
        )
    
    def with_changes(self, **kwargs) -> 'GenerationParams':
        """Create new params with specified changes."""
        current_dict = {
            'seed_text': self.seed_text,
            'length': self.length,
            'temperature': self.temperature,
            'sampling_strategy': self.sampling_strategy,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'beam_width': self.beam_width,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'stop_tokens': self.stop_tokens,
            'repetition_penalty': self.repetition_penalty,
            'no_repeat_ngram_size': self.no_repeat_ngram_size
        }
        current_dict.update(kwargs)
        return self.__class__(**current_dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'seed_text': self.seed_text,
            'length': self.length,
            'temperature': self.temperature,
            'sampling_strategy': self.sampling_strategy.value,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'beam_width': self.beam_width,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'stop_tokens': list(self.stop_tokens),
            'repetition_penalty': self.repetition_penalty,
            'no_repeat_ngram_size': self.no_repeat_ngram_size
        }
    
    def get_strategy_description(self) -> str:
        """Get human-readable strategy description."""
        descriptions = {
            SamplingStrategy.GREEDY: "Deterministic (always pick most likely)",
            SamplingStrategy.TEMPERATURE: f"Temperature sampling (T={self.temperature})",
            SamplingStrategy.TOP_K: f"Top-K sampling (K={self.top_k})",
            SamplingStrategy.NUCLEUS: f"Nucleus sampling (p={self.top_p})",
            SamplingStrategy.BEAM_SEARCH: f"Beam search (width={self.beam_width})"
        }
        return descriptions.get(self.sampling_strategy, "Unknown strategy")