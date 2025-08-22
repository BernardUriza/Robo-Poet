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
    """Handles text preprocessing and tokenization for word-level modeling with expanded vocabulary."""
    
    def __init__(self, sequence_length: int = 40, step_size: int = 3, vocab_size: int = 5000, tokenization: str = 'word'):
        """
        Initialize text processor.
        
        Args:
            sequence_length: Length of input sequences for LSTM
            step_size: Stride for sliding window sequence generation
            vocab_size: Maximum vocabulary size (5000 for Strategy 1.1)
            tokenization: 'char' for character-level, 'word' for word-level
        """
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.max_vocab_size = vocab_size
        self.tokenization = tokenization
        
        # Token mappings (renamed from char_to_idx)
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0
        
        # Special tokens for word-level tokenization
        self.special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    
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
        Clean and normalize text for word-level tokenization.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if self.tokenization == 'word':
            # Enhanced cleaning for word-level tokenization
            # Convert to lowercase for consistency
            text = text.lower()
            
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Keep only alphanumeric, punctuation, and whitespace
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']', '', text)
            
            # Normalize punctuation spacing
            text = re.sub(r'\s*([.!?;:])\s*', r' \1 ', text)
            text = re.sub(r'\s*([,])\s*', r'\1 ', text)
            
        else:
            # Original character-level cleaning
            text = re.sub(r'\s+', ' ', text)
            text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text.strip()
    
    def build_vocabulary(self, text: str) -> None:
        """
        Build vocabulary from text with expanded token support.
        
        Args:
            text: Input text for vocabulary extraction
        """
        if self.tokenization == 'word':
            # Word-level tokenization with 5000 vocabulary limit
            words = text.split()
            
            # Count word frequencies
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and take top words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Build vocabulary: special tokens + most frequent words
            vocab_tokens = self.special_tokens.copy()
            
            # Add most frequent words up to max_vocab_size
            for word, freq in sorted_words:
                if len(vocab_tokens) >= self.max_vocab_size:
                    break
                if word not in vocab_tokens:
                    vocab_tokens.append(word)
            
            self.vocab_size = len(vocab_tokens)
            
            # Create bidirectional mappings
            self.token_to_idx = {token: idx for idx, token in enumerate(vocab_tokens)}
            self.idx_to_token = {idx: token for idx, token in enumerate(vocab_tokens)}
            
            print(f"📊 Word-level vocabulary built: {self.vocab_size:,} unique tokens")
            print(f"   Most frequent: {[word for word, _ in sorted_words[:10]]}")
            print(f"   Special tokens: {self.special_tokens}")
            print(f"   Out-of-vocab words will map to: <UNK>")
            
        else:
            # Original character-level approach
            chars = sorted(set(text))
            self.vocab_size = len(chars)
            
            # Create bidirectional mappings
            self.token_to_idx = {char: idx for idx, char in enumerate(chars)}
            self.idx_to_token = {idx: char for idx, char in enumerate(chars)}
            
            print(f"📊 Character-level vocabulary built: {self.vocab_size} unique characters")
            sample_chars = list(chars)[:10]
            print(f"   Sample chars: {sample_chars}")
    
    def create_sequences(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training sequences using sliding window approach with word/char support.
        
        Args:
            text: Input text for sequence generation
            
        Returns:
            Tuple of (input_sequences, target_tokens) as numpy arrays
            
        Raises:
            ValueError: If vocabulary not built or text too short
        """
        if not self.token_to_idx:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        if self.tokenization == 'word':
            # Word-level sequence generation
            tokens = text.split()
            
            if len(tokens) < self.sequence_length + 1:
                raise ValueError(f"Text too short. Need at least {self.sequence_length + 1} tokens.")
            
            sequences = []
            next_tokens = []
            
            # Sliding window over tokens
            for i in range(0, len(tokens) - self.sequence_length, self.step_size):
                sequence = tokens[i:i + self.sequence_length]
                next_token = tokens[i + self.sequence_length]
                sequences.append(sequence)
                next_tokens.append(next_token)
            
            print(f"🔨 Generated {len(sequences):,} word-level training sequences")
            
            # Convert to one-hot encoded arrays
            X = np.zeros((len(sequences), self.sequence_length, self.vocab_size), dtype=np.bool_)
            y = np.zeros((len(sequences), self.vocab_size), dtype=np.bool_)
            
            unk_idx = self.token_to_idx['<UNK>']
            
            for i, sequence in enumerate(sequences):
                for t, token in enumerate(sequence):
                    token_idx = self.token_to_idx.get(token, unk_idx)
                    X[i, t, token_idx] = 1
                
                next_token_idx = self.token_to_idx.get(next_tokens[i], unk_idx)
                y[i, next_token_idx] = 1
            
        else:
            # Original character-level approach
            if len(text) < self.sequence_length + 1:
                raise ValueError(f"Text too short. Need at least {self.sequence_length + 1} characters.")
            
            sentences = []
            next_chars = []
            
            # Sliding window sequence generation
            for i in range(0, len(text) - self.sequence_length, self.step_size):
                sentences.append(text[i:i + self.sequence_length])
                next_chars.append(text[i + self.sequence_length])
            
            print(f"🔨 Generated {len(sentences):,} character-level training sequences")
            
            # Convert to one-hot encoded arrays
            X = np.zeros((len(sentences), self.sequence_length, self.vocab_size), dtype=np.bool_)
            y = np.zeros((len(sentences), self.vocab_size), dtype=np.bool_)
            
            for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    X[i, t, self.token_to_idx[char]] = 1
                y[i, self.token_to_idx[next_chars[i]]] = 1
        
        print(f"📈 Arrays created: X{X.shape}, y{y.shape}")
        print(f"   Memory usage: {(X.nbytes + y.nbytes) / 1024**2:.1f} MB")
        
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
    """Handles text generation from trained models with word/character support."""
    
    def __init__(self, model, token_to_idx: Dict[str, int], idx_to_token: Dict[int, str], tokenization: str = 'word'):
        """
        Initialize text generator.
        
        Args:
            model: Trained Keras model
            token_to_idx: Token to index mapping
            idx_to_token: Index to token mapping
            tokenization: 'char' or 'word' level generation
        """
        self.model = model
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.tokenization = tokenization
        self.vocab_size = len(token_to_idx)
        self.sequence_length = model.input_shape[1]
    
    def generate(self, seed_text: str, length: int = 200, temperature: float = 1.0) -> str:
        """
        Generate text using trained model with word/character support.
        
        Args:
            seed_text: Initial text to start generation
            length: Number of tokens/characters to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text string
        """
        if self.tokenization == 'word':
            # Word-level generation
            seed_tokens = seed_text.lower().split()
            unk_idx = self.token_to_idx.get('<UNK>', 0)
            
            # Prepare seed sequence
            if len(seed_tokens) < self.sequence_length:
                # Pad with <PAD> tokens
                pad_idx = self.token_to_idx.get('<PAD>', 0)
                seed_tokens = ['<PAD>'] * (self.sequence_length - len(seed_tokens)) + seed_tokens
            elif len(seed_tokens) > self.sequence_length:
                seed_tokens = seed_tokens[-self.sequence_length:]
            
            generated_tokens = seed_tokens.copy()
            current_sequence = seed_tokens.copy()
            
            for _ in range(length):
                # Prepare input
                x_pred = np.zeros((1, self.sequence_length, self.vocab_size))
                for t, token in enumerate(current_sequence):
                    token_idx = self.token_to_idx.get(token, unk_idx)
                    x_pred[0, t, token_idx] = 1
                
                # Predict next token
                preds = self.model.predict(x_pred, verbose=0)[0]
                
                # Apply temperature
                if temperature != 1.0:
                    preds = np.log(preds + 1e-8) / temperature
                    preds = np.exp(preds) / np.sum(np.exp(preds))
                
                # Sample next token
                next_index = self._sample(preds)
                next_token = self.idx_to_token.get(next_index, '<UNK>')
                
                # Skip special tokens in output
                if next_token not in ['<PAD>', '<START>', '<END>']:
                    generated_tokens.append(next_token)
                    
                # Update sequence for next iteration
                current_sequence = current_sequence[1:] + [next_token]
            
            # Join tokens, skip initial padding
            result_tokens = [t for t in generated_tokens if t != '<PAD>']
            return ' '.join(result_tokens)
            
        else:
            # Original character-level generation
            if len(seed_text) < self.sequence_length:
                seed_text = seed_text + ' ' * (self.sequence_length - len(seed_text))
            elif len(seed_text) > self.sequence_length:
                seed_text = seed_text[-self.sequence_length:]
            
            generated = seed_text
            
            for _ in range(length):
                # Prepare input
                x_pred = np.zeros((1, self.sequence_length, self.vocab_size))
                for t, char in enumerate(seed_text):
                    if char in self.token_to_idx:
                        x_pred[0, t, self.token_to_idx[char]] = 1
                
                # Predict next character
                preds = self.model.predict(x_pred, verbose=0)[0]
                
                # Apply temperature
                if temperature != 1.0:
                    preds = np.log(preds + 1e-8) / temperature
                    preds = np.exp(preds) / np.sum(np.exp(preds))
                
                # Sample next character
                next_index = self._sample(preds)
                next_char = self.idx_to_token.get(next_index, '?')
                
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
        return int(np.argmax(probas))