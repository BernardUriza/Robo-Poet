"""
Neural network model definitions for text generation - PyTorch Implementation.

Implements GPT-based architecture optimized for character-level language modeling,
replacing the previous TensorFlow LSTM implementation with modern transformer architecture.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path

# Import PyTorch GPT implementation
from models.pytorch_model_wrapper import PyTorchModelWrapper, create_pytorch_model

class RoboPoetModel:
    """
    Main RoboPoet model class using PyTorch GPT architecture.
    
    This replaces the previous TensorFlow LSTM implementation with a modern
    transformer-based approach while maintaining API compatibility.
    """
    
    def __init__(self, vocab_size: int = 6725, **kwargs):
        """
        Initialize RoboPoet model with PyTorch GPT architecture.
        
        Args:
            vocab_size: Size of vocabulary
            **kwargs: Additional model configuration
        """
        self.vocab_size = vocab_size
        self.model_wrapper = create_pytorch_model(vocab_size=vocab_size, **kwargs)
        
        print(f"[LAUNCH] RoboPoet PyTorch Model initialized")
        print(f"   [CYCLE] Migrated from TensorFlow LSTM to PyTorch GPT")
        print(f"   [CHART] Architecture: Transformer (vs previous LSTM)")
    
    def compile_model(self, **kwargs):
        """Compatibility method - PyTorch models don't need compilation."""
        print("[OK] PyTorch model ready (compilation not required)")
        return self
    
    def fit(self, *args, **kwargs):
        """
        Training compatibility method.
        
        Note: For full training, use the robo-poet-pytorch training system.
        This is a compatibility wrapper for the main system.
        """
        print(" For full training, use: robo-poet-pytorch/main.py train")
        return self
    
    def predict(self, input_data, **kwargs):
        """Predict using PyTorch GPT model."""
        return self.model_wrapper.predict(input_data, **kwargs)
    
    def generate_text(self, prompt: str = "", **kwargs) -> str:
        """Generate text using PyTorch GPT model."""
        return self.model_wrapper.generate_text(prompt, **kwargs)
    
    def save(self, filepath: str):
        """Save model (redirects to PyTorch checkpoint)."""
        print(f"[SAVE] PyTorch model saving handled by training system")
        print(f"   Use checkpoints in: robo-poet-pytorch/checkpoints/")
    
    def load_weights(self, filepath: str):
        """Load model weights from PyTorch checkpoint."""
        return self.model_wrapper.load_checkpoint(filepath)
    
    def summary(self):
        """Print model summary."""
        return self.model_wrapper.summary()
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return self.model_wrapper.get_model_info()


def create_model(vocab_size: int = 6725, **kwargs) -> RoboPoetModel:
    """
    Create RoboPoet model with PyTorch GPT architecture.
    
    Args:
        vocab_size: Vocabulary size
        **kwargs: Additional configuration
        
    Returns:
        RoboPoetModel instance
    """
    return RoboPoetModel(vocab_size=vocab_size, **kwargs)


def load_model(model_path: str) -> RoboPoetModel:
    """Load trained model from checkpoint."""
    model = create_model()
    model.load_weights(model_path)
    return model


# Legacy compatibility functions for existing code
def build_lstm_model(*args, **kwargs) -> RoboPoetModel:
    """Legacy function - now creates GPT model instead of LSTM."""
    print("[CYCLE] Legacy LSTM function redirected to PyTorch GPT")
    return create_model(**kwargs)


def build_model(*args, **kwargs) -> RoboPoetModel:
    """Build model with PyTorch GPT architecture."""
    return create_model(**kwargs)


# Model configuration constants (updated for GPT)
DEFAULT_MODEL_CONFIG = {
    'architecture': 'GPT',
    'framework': 'PyTorch',
    'n_layer': 6,
    'n_head': 8,
    'n_embd': 256,
    'block_size': 128,
    'dropout': 0.1,
    'vocab_size': 6725
}


if __name__ == "__main__":
    # Test the PyTorch model integration
    print(" Testing RoboPoet PyTorch Model Integration...")
    
    try:
        # Create model
        model = create_model()
        model.summary()
        
        # Test text generation
        generated = model.generate_text("To be or not to be", max_tokens=50)
        print(f"\n[DOC] Generated text: {generated[:100]}...")
        
        # Show configuration
        config = model.get_config()
        print(f"\n Model configuration: {config['framework']} {config['architecture']}")
        
        print(f"\n[OK] PyTorch integration test completed!")
        
    except Exception as e:
        print(f"[X] Integration test failed: {e}")
        import traceback
        traceback.print_exc()