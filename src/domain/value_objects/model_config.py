"""
ModelConfig value object - immutable configuration for models.

Value object containing all hyperparameters and configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Immutable value object for model configuration."""
    
    # Architecture
    vocab_size: int = 5000
    sequence_length: int = 40
    lstm_units: int = 256
    lstm_layers: int = 2
    embedding_dim: int = 128
    
    # Regularization (Strategy 2)
    variational_dropout_rate: float = 0.3
    dropconnect_rate: float = 0.2
    use_weight_tying: bool = True
    
    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    
    # Data processing
    tokenization: str = 'word'  # 'word' or 'char'
    max_text_length: int = 50_000_000
    step_size: int = 3
    
    # GPU optimization
    mixed_precision: bool = True
    memory_growth: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        self._validate()
    
    def _validate(self):
        """Validate all configuration values."""
        if not 10 <= self.sequence_length <= 200:
            raise ValueError("sequence_length must be between 10 and 200")
        
        if not 0 <= self.variational_dropout_rate < 1:
            raise ValueError("variational_dropout_rate must be between 0 and 1")
        
        if not 0 <= self.dropconnect_rate < 1:
            raise ValueError("dropconnect_rate must be between 0 and 1")
        
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if not 0 <= self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        
        if self.tokenization not in ['word', 'char']:
            raise ValueError("tokenization must be 'word' or 'char'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            # Architecture
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'lstm_layers': self.lstm_layers,
            'embedding_dim': self.embedding_dim,
            
            # Regularization
            'variational_dropout_rate': self.variational_dropout_rate,
            'dropconnect_rate': self.dropconnect_rate,
            'use_weight_tying': self.use_weight_tying,
            
            # Training
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            
            # Data processing
            'tokenization': self.tokenization,
            'max_text_length': self.max_text_length,
            'step_size': self.step_size,
            
            # GPU optimization
            'mixed_precision': self.mixed_precision,
            'memory_growth': self.memory_growth
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def light_regularization(cls, **kwargs) -> 'ModelConfig':
        """Factory method for light regularization."""
        defaults = {
            'variational_dropout_rate': 0.1,
            'dropconnect_rate': 0.05
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def heavy_regularization(cls, **kwargs) -> 'ModelConfig':
        """Factory method for heavy regularization."""
        defaults = {
            'variational_dropout_rate': 0.5,
            'dropconnect_rate': 0.3
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def fast_training(cls, **kwargs) -> 'ModelConfig':
        """Factory method for fast training configuration."""
        defaults = {
            'epochs': 20,
            'batch_size': 128,
            'early_stopping_patience': 5
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def production_ready(cls, **kwargs) -> 'ModelConfig':
        """Factory method for production-ready configuration."""
        defaults = {
            'epochs': 200,
            'batch_size': 32,
            'early_stopping_patience': 25,
            'mixed_precision': True,
            'variational_dropout_rate': 0.3,
            'dropconnect_rate': 0.2
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    def with_changes(self, **kwargs) -> 'ModelConfig':
        """Create new config with specified changes."""
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return self.__class__.from_dict(current_dict)
    
    def get_regularization_summary(self) -> str:
        """Get human-readable regularization summary."""
        level = "Light"
        if self.variational_dropout_rate >= 0.4:
            level = "Heavy"
        elif self.variational_dropout_rate >= 0.25:
            level = "Moderate"
        
        return f"{level} (VarDrop: {self.variational_dropout_rate}, DropConnect: {self.dropconnect_rate})"
    
    def estimate_memory_usage_mb(self) -> int:
        """Estimate memory usage in MB."""
        # Rough estimation based on model parameters
        params = (
            self.vocab_size * self.embedding_dim +  # Embedding
            4 * (self.lstm_units * (self.lstm_units + self.embedding_dim + 1)) +  # LSTM1
            4 * (self.lstm_units * (self.lstm_units + self.lstm_units + 1)) +  # LSTM2
            self.lstm_units * self.vocab_size  # Output layer
        )
        
        # 4 bytes per float32, batch size multiplier, gradients
        memory_mb = (params * 4 * self.batch_size * 3) // (1024 * 1024)
        return max(memory_mb, 100)  # Minimum 100MB