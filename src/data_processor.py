"""
Data processing module for text tokenization and sequence generation.

Implements character-level and word-level tokenization with sequence preparation for LSTM training,
following standard practices in neural language modeling. Supports both character-based and 
word-based tokenization strategies with configurable vocabulary sizes and sequence parameters.

Classes:
    TextProcessor: Main class for text preprocessing, vocabulary building, and sequence generation
    TextGenerator: Handles text generation from trained models with temperature sampling

Key Features:
    - Dual tokenization support (character-level and word-level)
    - Configurable vocabulary size with frequency-based selection
    - Sliding window sequence generation with configurable stride
    - Special token handling for padding, unknown tokens, and sequence boundaries
    - One-hot encoding for neural network compatibility
    - Text cleaning and normalization with regex patterns
    - Memory usage reporting and optimization guidance
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict, List, Optional, Union, Literal, Any
import re
from pathlib import Path
import logging
from dataclasses import dataclass

# Type aliases for better clarity
TokenizationMode = Literal['word', 'char']
TokenIndex = int
TokenSequence = List[str]
OneHotArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int32]


@dataclass
class ProcessingStats:
    """Statistics from text processing operations."""
    total_chars: int
    total_tokens: int
    unique_tokens: int
    vocab_coverage: float
    oov_tokens: int
    sequence_count: int
    memory_usage_mb: float


class TextProcessor:
    """
    Advanced text processor supporting both character and word-level tokenization.
    
    This class provides comprehensive text preprocessing capabilities including:
    - Configurable tokenization modes (character or word-level)  
    - Frequency-based vocabulary building with size limits
    - Sliding window sequence generation for training data
    - Text normalization and cleaning with regex patterns
    - One-hot encoding for neural network compatibility
    - Memory usage tracking and optimization
    
    Attributes:
        sequence_length: Length of input sequences for model training
        step_size: Stride for sliding window sequence generation  
        max_vocab_size: Maximum vocabulary size limit
        tokenization: Tokenization mode ('word' or 'char')
        token_to_idx: Mapping from tokens to integer indices
        idx_to_token: Mapping from integer indices to tokens
        vocab_size: Actual vocabulary size after building
        special_tokens: List of special tokens (PAD, UNK, START, END)
        
    Example:
        >>> processor = TextProcessor(sequence_length=50, vocab_size=10000, tokenization='word')
        >>> text = processor.load_text('corpus.txt', max_length=100000)
        >>> processor.build_vocabulary(text) 
        >>> X, y = processor.create_sequences(text)
        >>> print(f"Created {len(X)} training sequences")
    """
    
    def __init__(
        self, 
        sequence_length: int = 40, 
        step_size: int = 3, 
        vocab_size: int = 5000, 
        tokenization: TokenizationMode = 'word'
    ) -> None:
        """
        Initialize text processor with specified parameters.
        
        Args:
            sequence_length: Length of input sequences for LSTM training (typically 40-100)
            step_size: Stride for sliding window sequence generation (typically 1-5)
            vocab_size: Maximum vocabulary size limit (5000-50000 recommended)
            tokenization: Tokenization mode - 'word' for word-level, 'char' for character-level
            
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        # Validate parameters
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if step_size < 1:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if vocab_size < 4:  # Must accommodate special tokens
            raise ValueError(f"vocab_size must be at least 4, got {vocab_size}")
        if tokenization not in ['word', 'char']:
            raise ValueError(f"tokenization must be 'word' or 'char', got {tokenization}")
        
        self.sequence_length: int = sequence_length
        self.step_size: int = step_size
        self.max_vocab_size: int = vocab_size
        self.tokenization: TokenizationMode = tokenization
        
        # Token mappings - bidirectional for efficient lookup
        self.token_to_idx: Dict[str, TokenIndex] = {}
        self.idx_to_token: Dict[TokenIndex, str] = {}
        self.vocab_size: int = 0
        
        # Special tokens with consistent ordering
        self.special_tokens: List[str] = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def load_text(self, corpus_path: Union[str, Path] = "corpus", max_length: int = 200_000, prefer_unified: bool = True) -> str:
        """
        Load text from unified corpus or fallback to multi-file corpus.
        
        Automatically prefers the preprocessed unified corpus when available, otherwise
        falls back to combining individual .txt files from the corpus directory.
        The unified corpus provides better training convergence due to proper
        document markers and consistent preprocessing.
        
        Args:
            corpus_path: Path to corpus directory containing .txt files (default: "corpus")
            max_length: Maximum characters to load total (default 200,000 for multi-corpus)
            prefer_unified: Use unified corpus if available (default: True)
            
        Returns:
            Combined preprocessed text string from unified or multi-file corpus
            
        Raises:
            FileNotFoundError: If corpus directory doesn't exist or contains no .txt files
            UnicodeDecodeError: If any file has encoding issues (non-UTF-8)
            PermissionError: If insufficient permissions to read files
            OSError: For other file system related errors
            ValueError: If max_length is not positive
            
        Example:
            >>> processor = TextProcessor()
            >>> text = processor.load_text("corpus", max_length=500000)  # Load all corpus files
            >>> print(f"Combined corpus loaded: {len(text)} characters")
        """
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        # Check for preprocessed unified corpus first
        unified_corpus_path = Path("data/processed/unified_corpus.txt")
        
        if prefer_unified and unified_corpus_path.exists():
            print(f"🎯 USING UNIFIED PREPROCESSED CORPUS")
            print(f"   File: {unified_corpus_path}")
            
            try:
                with open(unified_corpus_path, 'r', encoding='utf-8') as f:
                    unified_text = f.read()[:max_length]
                
                print(f"   ✅ Unified corpus loaded: {len(unified_text):,} characters")
                print(f"   🎭 Contains document markers for style control")
                print(f"   📊 Preprocessed vocabulary and normalization")
                
                return unified_text
                
            except Exception as e:
                print(f"   ⚠️ Error loading unified corpus: {e}")
                print(f"   📝 Falling back to multi-file corpus...")
        
        # Fallback to original multi-file approach
        corpus_path = Path(corpus_path)  # Convert to Path for consistency
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")
            
        if not corpus_path.is_dir():
            raise FileNotFoundError(f"Corpus path is not a directory: {corpus_path}")
        
        # Find all .txt files in corpus directory
        txt_files = list(corpus_path.glob("*.txt"))
        
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in corpus directory: {corpus_path}")
        
        print(f"📚 MULTI-CORPUS LOADING SYSTEM (FALLBACK)")
        print(f"   Directory: {corpus_path}")
        print(f"   Found {len(txt_files)} text files:")
        print(f"   💡 Tip: Ejecuta preprocesamiento para mejor convergencia")
        
        combined_text = ""
        total_chars_loaded = 0
        files_loaded = 0
        
        # Sort files for consistent loading order
        txt_files.sort()
        
        try:
            for txt_file in txt_files:
                # Check if we've reached the character limit
                if total_chars_loaded >= max_length:
                    print(f"   ⚠️ Reached max_length limit ({max_length:,} chars), stopping...")
                    break
                
                print(f"   📖 Loading: {txt_file.name}...", end=" ")
                
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        # Calculate remaining space
                        remaining_space = max_length - total_chars_loaded
                        file_content = f.read()[:remaining_space]
                    
                    # Clean the individual file content
                    cleaned_content = self._clean_text(file_content)
                    
                    # Add separator between different works
                    if combined_text and cleaned_content:
                        combined_text += "\n\n" + cleaned_content
                    else:
                        combined_text += cleaned_content
                    
                    file_chars = len(cleaned_content)
                    total_chars_loaded += file_chars
                    files_loaded += 1
                    
                    print(f"✅ {file_chars:,} chars")
                    
                except UnicodeDecodeError as e:
                    print(f"❌ Encoding error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
            
            if not combined_text:
                raise FileNotFoundError("No valid text content could be loaded from corpus files")
            
            print(f"\n📊 MULTI-CORPUS SUMMARY:")
            print(f"   ✅ Files successfully loaded: {files_loaded}/{len(txt_files)}")
            print(f"   📝 Total characters: {len(combined_text):,}")
            print(f"   🎯 Combined corpus ready for training")
            
            return combined_text
            
        except Exception as e:
            if "No valid text content" in str(e):
                raise
            raise FileNotFoundError(f"Error loading corpus files: {e}")
    
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
    
    def prepare_data(self, corpus_path: str = "corpus", max_length: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete multi-corpus data preparation pipeline.
        
        Args:
            corpus_path: Path to corpus directory containing multiple .txt files (default: "corpus")
            max_length: Maximum total characters to process from all files
            
        Returns:
            Tuple of prepared (X, y) arrays for training from combined corpus
        """
        print("📚 MULTI-CORPUS DATA PREPARATION PIPELINE")
        print("=" * 60)
        
        # Load and combine all corpus texts
        text = self.load_text(corpus_path, max_length)
        
        # Build vocabulary from combined corpus
        self.build_vocabulary(text)
        
        # Create training sequences from combined text
        X, y = self.create_sequences(text)
        
        return X, y

class TextGenerator:
    """
    Advanced text generator supporting multiple sampling strategies.
    
    This class provides sophisticated text generation capabilities from trained
    neural language models, supporting both character and word-level generation
    with configurable sampling strategies including temperature scaling,
    top-k sampling, and nucleus (top-p) sampling.
    
    Key Features:
        - Temperature-based sampling for creativity control
        - Support for both word and character-level generation
        - Proper handling of special tokens (PAD, UNK, START, END)
        - Sequence boundary management for coherent generation
        - Memory-efficient generation with streaming support
        - Comprehensive error handling and validation
        
    Attributes:
        model: Trained Keras/TensorFlow model for text generation
        token_to_idx: Mapping from tokens to integer indices
        idx_to_token: Mapping from integer indices to tokens  
        tokenization: Generation mode ('word' or 'char')
        vocab_size: Size of the vocabulary
        sequence_length: Input sequence length for the model
        
    Example:
        >>> generator = TextGenerator(model, token_to_idx, idx_to_token, 'word')
        >>> text = generator.generate("The quick brown", length=50, temperature=0.8)
        >>> print(text)
    """
    
    def __init__(
        self, 
        token_to_idx: Dict[str, TokenIndex], 
        idx_to_token: Dict[TokenIndex, str], 
        tokenization: TokenizationMode = 'word'
    ) -> None:
        """
        Initialize text generator with model and vocabulary mappings.
        
        Args:
            model: Trained Keras/TensorFlow model with input shape (batch_size, sequence_length, vocab_size)
            token_to_idx: Dictionary mapping tokens to integer indices
            idx_to_token: Dictionary mapping integer indices to tokens
            tokenization: Generation mode - 'word' for word-level, 'char' for character-level
            
        Raises:
            ValueError: If vocabularies are inconsistent or model has invalid input shape
            AttributeError: If model doesn't have required attributes
        """
        # Validate inputs
        if not token_to_idx or not idx_to_token:
            raise ValueError("Token mappings cannot be empty")
        
        if len(token_to_idx) != len(idx_to_token):
            raise ValueError(f"Token mapping sizes don't match: {len(token_to_idx)} vs {len(idx_to_token)}")
            
        if tokenization not in ['word', 'char']:
            raise ValueError(f"tokenization must be 'word' or 'char', got {tokenization}")
        
        # Validate model input shape
        if not hasattr(model, 'input_shape') or len(model.input_shape) < 2:
            raise AttributeError("Model must have valid input_shape with at least 2 dimensions")
            
        self.model = model
        self.token_to_idx: Dict[str, TokenIndex] = token_to_idx
        self.idx_to_token: Dict[TokenIndex, str] = idx_to_token
        self.tokenization: TokenizationMode = tokenization
        self.vocab_size: int = len(token_to_idx)
        self.sequence_length: int = model.input_shape[1]
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def generate(
        self, 
        seed_text: str, 
        length: int = 200, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate text using the trained model with advanced sampling strategies.
        
        Supports multiple sampling methods for controlling generation quality and creativity:
        - Temperature sampling: Controls randomness (0.1 = conservative, 2.0 = creative)
        - Top-k sampling: Only considers k most likely tokens
        - Nucleus (top-p) sampling: Considers tokens up to cumulative probability p
        
        Args:
            seed_text: Initial text to start generation from
            length: Number of tokens/characters to generate (must be positive)
            temperature: Sampling temperature (0.1-2.0 recommended, 1.0 = no modification)
            top_k: If specified, only sample from top k most likely tokens
            top_p: If specified, use nucleus sampling with cumulative probability p
            
        Returns:
            Generated text string, starting with seed_text
            
        Raises:
            ValueError: If parameters are out of valid ranges
            RuntimeError: If model prediction fails
            
        Example:
            >>> # Conservative generation
            >>> text = generator.generate("Once upon a time", length=100, temperature=0.7)
            >>> 
            >>> # Creative generation with top-k sampling  
            >>> text = generator.generate("In the future", length=50, temperature=1.2, top_k=40)
        """
        # Validate parameters
        if not isinstance(seed_text, str) or len(seed_text.strip()) == 0:
            raise ValueError("seed_text must be a non-empty string")
            
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
            
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
            
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive if specified, got {top_k}")
            
        if top_p is not None and not (0 < top_p <= 1):
            raise ValueError(f"top_p must be between 0 and 1 if specified, got {top_p}")
        
        self.logger.debug(f"Generating {length} tokens from seed: '{seed_text[:50]}...'")
        
        try:
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
                    # Prepare input as integer encoding for embedding layer
                    x_pred = np.zeros((1, self.sequence_length), dtype=np.int32)
                    for t, token in enumerate(current_sequence):
                        token_idx = self.token_to_idx.get(token, unk_idx)
                        x_pred[0, t] = token_idx
                    
                    # Predict next token
                    preds = self.model.predict(x_pred, verbose=0)[0]
                    
                    # Sample next token with advanced strategies
                    next_index = self._sample(preds, temperature, top_k, top_p)
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
                # Character-level generation
                if len(seed_text) < self.sequence_length:
                    seed_text = seed_text + ' ' * (self.sequence_length - len(seed_text))
                elif len(seed_text) > self.sequence_length:
                    seed_text = seed_text[-self.sequence_length:]
                
                generated = seed_text
                
                for _ in range(length):
                    # Prepare input as integer encoding for embedding layer
                    x_pred = np.zeros((1, self.sequence_length), dtype=np.int32)
                    for t, char in enumerate(seed_text):
                        if char in self.token_to_idx:
                            x_pred[0, t] = self.token_to_idx[char]
                    
                    # Predict next character
                    preds = self.model.predict(x_pred, verbose=0)[0]
                    
                    # Sample next character with advanced strategies
                    next_index = self._sample(preds, temperature, top_k, top_p)
                    next_char = self.idx_to_token.get(next_index, '?')
                    
                    # Update for next iteration
                    generated += next_char
                    seed_text = seed_text[1:] + next_char
                
                return generated
                
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate text: {str(e)}") from e
    
    def _sample(
        self, 
        preds: npt.NDArray[np.float32], 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> TokenIndex:
        """
        Sample from probability distribution with advanced sampling strategies.
        
        Implements multiple sampling methods:
        1. Temperature sampling: Scales logits by temperature before softmax
        2. Top-k sampling: Only considers k most likely tokens
        3. Nucleus (top-p) sampling: Considers tokens up to cumulative probability p
        
        Args:
            preds: Raw probability distribution over vocabulary
            temperature: Temperature for scaling (1.0 = no change, <1.0 = more conservative, >1.0 = more random)
            top_k: If specified, only sample from top k most likely tokens
            top_p: If specified, use nucleus sampling with cumulative probability p
            
        Returns:
            Sampled token index
            
        Raises:
            ValueError: If predictions contain invalid values
        """
        # Validate predictions
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            raise ValueError("Predictions contain NaN or infinite values")
            
        if np.all(preds <= 0):
            raise ValueError("All predictions are non-positive")
        
        # Convert to float64 for numerical stability
        preds = np.asarray(preds, dtype=np.float64)
        
        # Apply temperature scaling
        if temperature != 1.0:
            preds = np.log(preds + 1e-8) / temperature
            preds = np.exp(preds)
        
        # Apply top-k filtering
        if top_k is not None:
            # Keep only top-k predictions, set others to 0
            top_k_indices = np.argpartition(preds, -top_k)[-top_k:]
            filtered_preds = np.zeros_like(preds)
            filtered_preds[top_k_indices] = preds[top_k_indices]
            preds = filtered_preds
        
        # Apply nucleus (top-p) filtering
        if top_p is not None:
            # Sort predictions in descending order
            sorted_indices = np.argsort(preds)[::-1]
            sorted_preds = preds[sorted_indices]
            
            # Calculate cumulative probabilities
            normalized_preds = sorted_preds / np.sum(sorted_preds)
            cumsum_preds = np.cumsum(normalized_preds)
            
            # Find cutoff index where cumulative probability exceeds top_p
            cutoff_idx = np.where(cumsum_preds >= top_p)[0]
            if len(cutoff_idx) > 0:
                cutoff_idx = cutoff_idx[0] + 1  # Include the token that crosses threshold
            else:
                cutoff_idx = len(sorted_preds)  # Keep all if none exceed threshold
            
            # Keep only tokens up to cutoff
            filtered_preds = np.zeros_like(preds)
            filtered_preds[sorted_indices[:cutoff_idx]] = preds[sorted_indices[:cutoff_idx]]
            preds = filtered_preds
        
        # Normalize to get valid probability distribution
        preds = preds / np.sum(preds)
        
        # Sample from the final distribution
        try:
            probas = np.random.multinomial(1, preds, 1)
            return int(np.argmax(probas))
        except ValueError as e:
            # Fallback to argmax if multinomial sampling fails
            self.logger.warning(f"Multinomial sampling failed, using argmax: {e}")
            return int(np.argmax(preds))