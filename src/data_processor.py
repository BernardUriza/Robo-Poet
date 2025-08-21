"""
Data processing module for text tokenization and sequence generation.

Implements character-level tokenization and sequence preparation for LSTM training,
following standard practices in neural language modeling.
"""

import numpy as np
from typing import Tuple, Dict, List
import re
from pathlib import Path

class TextProcessor:
    """Handles text preprocessing and tokenization for character-level modeling."""
    
    def __init__(self, sequence_length: int = 40, step_size: int = 3):
        """
        Initialize text processor.
        
        Args:
            sequence_length: Length of input sequences for LSTM
            step_size: Stride for sliding window sequence generation
        """
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
    
    def load_text(self, filepath: str, max_length: int = 50_000) -> str:
        """
        Load and preprocess text from file.
        
        Args:
            filepath: Path to text file
            max_length: Maximum characters to load
            
        Returns:
            Preprocessed text string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding issues
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()[:max_length]
            
            # Basic text cleaning
            text = self._clean_text(text)
            
            print(f"✅ Text loaded: {len(text):,} characters")
            return text
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Text file not found: {filepath}")
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(f"Encoding error in {filepath}: {e}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except newlines
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text.strip()
    
    def build_vocabulary(self, text: str) -> None:
        """
        Build character-level vocabulary from text.
        
        Args:
            text: Input text for vocabulary extraction
        """
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        
        # Create bidirectional mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
        print(f"📊 Vocabulary built: {self.vocab_size} unique characters")
        
        # Log sample vocabulary
        sample_chars = list(chars)[:10]
        print(f"   Sample chars: {sample_chars}")
    
    def create_sequences(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training sequences using sliding window approach.
        
        Args:
            text: Input text for sequence generation
            
        Returns:
            Tuple of (input_sequences, target_chars) as numpy arrays
            
        Raises:
            ValueError: If vocabulary not built or text too short
        """
        if not self.char_to_idx:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        if len(text) < self.sequence_length + 1:
            raise ValueError(f"Text too short. Need at least {self.sequence_length + 1} characters.")
        
        sentences = []
        next_chars = []
        
        # Sliding window sequence generation
        for i in range(0, len(text) - self.sequence_length, self.step_size):
            sentences.append(text[i:i + self.sequence_length])
            next_chars.append(text[i + self.sequence_length])
        
        print(f"🔨 Generated {len(sentences):,} training sequences")
        
        # Convert to one-hot encoded arrays
        X = np.zeros((len(sentences), self.sequence_length, self.vocab_size), dtype=np.bool_)
        y = np.zeros((len(sentences), self.vocab_size), dtype=np.bool_)
        
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, self.char_to_idx[char]] = 1
            y[i, self.char_to_idx[next_chars[i]]] = 1
        
        print(f"📈 Arrays created: X{X.shape}, y{y.shape}")
        print(f"   Memory usage: {X.nbytes + y.nbytes / 1024**2:.1f} MB")
        
        return X, y
    
    def prepare_data(self, filepath: str, max_length: int = 50_000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Args:
            filepath: Path to text file
            max_length: Maximum characters to process
            
        Returns:
            Tuple of prepared (X, y) arrays for training
        """
        print("📚 Starting data preparation pipeline...")
        
        # Load and preprocess text
        text = self.load_text(filepath, max_length)
        
        # Build vocabulary
        self.build_vocabulary(text)
        
        # Create training sequences
        X, y = self.create_sequences(text)
        
        return X, y

class TextGenerator:
    """Handles text generation from trained models."""
    
    def __init__(self, model, char_to_idx: Dict[str, int], idx_to_char: Dict[int, str]):
        """
        Initialize text generator.
        
        Args:
            model: Trained Keras model
            char_to_idx: Character to index mapping
            idx_to_char: Index to character mapping
        """
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        self.sequence_length = model.input_shape[1]
    
    def generate(self, seed_text: str, length: int = 200, temperature: float = 1.0) -> str:
        """
        Generate text using trained model.
        
        Args:
            seed_text: Initial text to start generation
            length: Number of characters to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text string
        """
        # Prepare seed
        if len(seed_text) < self.sequence_length:
            seed_text = seed_text + ' ' * (self.sequence_length - len(seed_text))
        elif len(seed_text) > self.sequence_length:
            seed_text = seed_text[-self.sequence_length:]
        
        generated = seed_text
        
        for _ in range(length):
            # Prepare input
            x_pred = np.zeros((1, self.sequence_length, self.vocab_size))
            for t, char in enumerate(seed_text):
                if char in self.char_to_idx:
                    x_pred[0, t, self.char_to_idx[char]] = 1
            
            # Predict next character
            preds = self.model.predict(x_pred, verbose=0)[0]
            
            # Apply temperature
            if temperature != 1.0:
                preds = np.log(preds + 1e-8) / temperature
                preds = np.exp(preds) / np.sum(np.exp(preds))
            
            # Sample next character
            next_index = self._sample(preds)
            next_char = self.idx_to_char[next_index]
            
            # Update for next iteration
            generated += next_char
            seed_text = seed_text[1:] + next_char
        
        return generated
    
    def _sample(self, preds: np.ndarray) -> int:
        """
        Sample from probability distribution.
        
        Args:
            preds: Probability distribution over vocabulary
            
        Returns:
            Sampled character index
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)