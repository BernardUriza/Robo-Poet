"""
Advanced preprocessing pipeline for text data with multiple tokenization strategies
and comprehensive text cleaning capabilities.
"""

import re
import string
import logging
import time
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class TokenizationStrategy(Enum):
    """Available tokenization strategies."""
    WORD_BASED = "word_based"
    SUBWORD_BPE = "subword_bpe"
    CHARACTER_BASED = "character_based"
    SENTENCE_PIECE = "sentence_piece"
    WHITESPACE = "whitespace"
    CUSTOM = "custom"


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing pipeline."""
    
    # Tokenization settings
    tokenization_strategy: TokenizationStrategy = TokenizationStrategy.WORD_BASED
    vocab_size: int = 10000
    min_token_frequency: int = 2
    max_sequence_length: int = 512
    
    # Text cleaning settings
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_numbers: bool = False
    remove_special_chars: bool = False
    normalize_whitespace: bool = True
    remove_accents: bool = False
    
    # Unicode handling
    unicode_normalization: str = "NFC"  # NFC, NFD, NFKC, NFKD
    handle_emojis: str = "keep"  # keep, remove, replace
    emoji_replacement: str = " <emoji> "
    
    # Language-specific settings
    language: str = "auto"  # auto, en, es, fr, de, etc.
    stem_words: bool = False
    lemmatize_words: bool = False
    
    # Special tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    mask_token: str = "<MASK>"
    
    # BPE settings (for subword tokenization)
    bpe_merges: int = 1000
    bpe_dropout: float = 0.0
    
    # Filtering settings
    min_line_length: int = 1
    max_line_length: int = 1000
    filter_non_printable: bool = True
    filter_duplicates: bool = True
    
    # Performance settings
    batch_size: int = 1000
    num_workers: int = 4
    cache_tokenizer: bool = True
    cache_dir: Optional[str] = None
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.vocab_size > 0
        assert self.min_token_frequency >= 1
        assert self.max_sequence_length > 0
        assert self.min_line_length >= 0
        assert self.max_line_length > self.min_line_length
        assert self.unicode_normalization in ["NFC", "NFD", "NFKC", "NFKD"]
        assert self.handle_emojis in ["keep", "remove", "replace"]


class AdvancedPreprocessor:
    """Advanced text preprocessing with multiple strategies."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.config.validate()
        
        # Tokenizer state
        self.vocabulary = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.token_frequencies = Counter()
        self.is_fitted = False
        
        # Special token IDs
        self.special_token_ids = {}
        
        # BPE state (for subword tokenization)
        self.bpe_merges = {}
        self.bpe_vocab = set()
        
        # Compiled regex patterns for efficiency
        self._compile_regex_patterns()
        
        # Statistics
        self.processing_stats = {
            'lines_processed': 0,
            'lines_filtered': 0,
            'avg_line_length': 0,
            'vocab_size': 0,
            'processing_time': 0
        }
        
        logger.info(f"AdvancedPreprocessor initialized: {config.tokenization_strategy.value}")
    
    def _compile_regex_patterns(self):
        """Compile regex patterns for efficient text processing."""
        self.patterns = {}
        
        # Whitespace normalization
        self.patterns['whitespace'] = re.compile(r'\s+')
        
        # Punctuation removal
        if self.config.remove_punctuation:
            self.patterns['punctuation'] = re.compile(f'[{re.escape(string.punctuation)}]')
        
        # Number removal
        if self.config.remove_numbers:
            self.patterns['numbers'] = re.compile(r'\d+')
        
        # Special characters
        if self.config.remove_special_chars:
            self.patterns['special'] = re.compile(r'[^\w\s]', re.UNICODE)
        
        # Non-printable characters
        if self.config.filter_non_printable:
            self.patterns['non_printable'] = re.compile(r'[^\x20-\x7E\u00A0-\uFFFF]')
        
        # Emoji detection
        if self.config.handle_emojis in ['remove', 'replace']:
            # Simplified emoji pattern - would use more comprehensive in practice
            self.patterns['emoji'] = re.compile(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+'
            )
    
    def clean_text(self, text: str) -> str:
        """
        Clean text according to configuration.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize(self.config.unicode_normalization, text)
        
        # Handle emojis
        if self.config.handle_emojis == 'remove' and 'emoji' in self.patterns:
            text = self.patterns['emoji'].sub('', text)
        elif self.config.handle_emojis == 'replace' and 'emoji' in self.patterns:
            text = self.patterns['emoji'].sub(self.config.emoji_replacement, text)
        
        # Remove accents
        if self.config.remove_accents:
            text = self._remove_accents(text)
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.config.remove_punctuation and 'punctuation' in self.patterns:
            text = self.patterns['punctuation'].sub(' ', text)
        
        # Remove numbers
        if self.config.remove_numbers and 'numbers' in self.patterns:
            text = self.patterns['numbers'].sub(' ', text)
        
        # Remove special characters
        if self.config.remove_special_chars and 'special' in self.patterns:
            text = self.patterns['special'].sub(' ', text)
        
        # Remove non-printable characters
        if self.config.filter_non_printable and 'non_printable' in self.patterns:
            text = self.patterns['non_printable'].sub('', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.patterns['whitespace'].sub(' ', text)
            text = text.strip()
        
        return text
    
    def _remove_accents(self, text: str) -> str:
        """Remove accents from text."""
        return ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text according to strategy.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        strategy = self.config.tokenization_strategy
        
        if strategy == TokenizationStrategy.WORD_BASED:
            return self._word_tokenize(text)
        elif strategy == TokenizationStrategy.SUBWORD_BPE:
            return self._bpe_tokenize(text)
        elif strategy == TokenizationStrategy.CHARACTER_BASED:
            return self._char_tokenize(text)
        elif strategy == TokenizationStrategy.WHITESPACE:
            return text.split()
        else:
            return text.split()  # Fallback
    
    def _word_tokenize(self, text: str) -> List[str]:
        """Word-based tokenization."""
        # Simple word tokenization - could be enhanced with NLTK/spaCy
        tokens = re.findall(r'\b\w+\b', text, re.UNICODE)
        return tokens
    
    def _char_tokenize(self, text: str) -> List[str]:
        """Character-based tokenization."""
        return list(text)
    
    def _bpe_tokenize(self, text: str) -> List[str]:
        """Byte-Pair Encoding tokenization."""
        if not self.bpe_merges:
            # Fallback to word tokenization if BPE not trained
            return self._word_tokenize(text)
        
        # Start with character-level tokens
        tokens = list(text)
        
        # Apply BPE merges
        while True:
            pairs = self._get_token_pairs(tokens)
            if not pairs:
                break
            
            # Find the most frequent pair that exists in our merges
            best_pair = None
            for pair in pairs:
                if pair in self.bpe_merges:
                    if best_pair is None or self.bpe_merges[pair] < self.bpe_merges[best_pair]:
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # Merge the best pair
            tokens = self._merge_tokens(tokens, best_pair)
        
        return tokens
    
    def _get_token_pairs(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Get all adjacent token pairs."""
        pairs = []
        for i in range(len(tokens) - 1):
            pairs.append((tokens[i], tokens[i + 1]))
        return pairs
    
    def _merge_tokens(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge a token pair into a single token."""
        merged = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and 
                tokens[i] == pair[0] and 
                tokens[i + 1] == pair[1]):
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged
    
    def fit_tokenizer(self, texts: Iterator[str]) -> Dict[str, Any]:
        """
        Fit tokenizer on text corpus.
        
        Args:
            texts: Iterator over text lines
            
        Returns:
            Fitting statistics
        """
        start_time = time.time()
        logger.info("Fitting tokenizer on corpus...")
        
        # Reset state
        self.token_frequencies.clear()
        self.vocabulary.clear()
        
        # Collect token statistics
        lines_processed = 0
        lines_filtered = 0
        total_length = 0
        
        for text in texts:
            lines_processed += 1
            
            # Clean text
            cleaned = self.clean_text(text)
            
            # Filter by length
            if (len(cleaned) < self.config.min_line_length or
                len(cleaned) > self.config.max_line_length):
                lines_filtered += 1
                continue
            
            total_length += len(cleaned)
            
            # Tokenize and count
            tokens = self.tokenize_text(cleaned)
            self.token_frequencies.update(tokens)
            
            if lines_processed % 10000 == 0:
                logger.info(f"Processed {lines_processed} lines...")
        
        # Build vocabulary
        self._build_vocabulary()
        
        # Train BPE if needed
        if self.config.tokenization_strategy == TokenizationStrategy.SUBWORD_BPE:
            self._train_bpe()
        
        # Update statistics
        processing_time = time.time() - start_time
        avg_length = total_length / max(1, lines_processed - lines_filtered)
        
        self.processing_stats.update({
            'lines_processed': lines_processed,
            'lines_filtered': lines_filtered,
            'avg_line_length': avg_length,
            'vocab_size': len(self.vocabulary),
            'processing_time': processing_time
        })
        
        self.is_fitted = True
        
        logger.info(f"Tokenizer fitted in {processing_time:.2f}s")
        logger.info(f"  Lines processed: {lines_processed}")
        logger.info(f"  Lines filtered: {lines_filtered}")
        logger.info(f"  Vocabulary size: {len(self.vocabulary)}")
        logger.info(f"  Average line length: {avg_length:.1f}")
        
        return self.processing_stats.copy()
    
    def _build_vocabulary(self):
        """Build vocabulary from token frequencies."""
        # Add special tokens first
        special_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.mask_token
        ]
        
        self.vocabulary = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens
        for i, token in enumerate(special_tokens):
            self.vocabulary[token] = i
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            self.special_token_ids[token] = i
        
        # Add most frequent tokens
        next_id = len(special_tokens)
        sorted_tokens = self.token_frequencies.most_common()
        
        for token, freq in sorted_tokens:
            if freq < self.config.min_token_frequency:
                break
                
            if token not in self.vocabulary and next_id < self.config.vocab_size:
                self.vocabulary[token] = next_id
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
    
    def _train_bpe(self):
        """Train Byte-Pair Encoding on the corpus."""
        logger.info("Training BPE...")
        
        # Start with character-level vocabulary
        char_vocab = set()
        for token in self.token_frequencies:
            char_vocab.update(list(token))
        
        self.bpe_vocab = char_vocab.copy()
        self.bpe_merges = {}
        
        # Iteratively merge most frequent pairs
        for merge_idx in range(self.config.bpe_merges):
            # Count all pairs across the vocabulary
            pair_counts = defaultdict(int)
            
            for token, freq in self.token_frequencies.items():
                chars = list(token)
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_counts[pair] += freq
            
            if not pair_counts:
                break
            
            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            
            # Record this merge
            self.bpe_merges[best_pair] = merge_idx
            
            # Update vocabulary
            new_token = best_pair[0] + best_pair[1]
            self.bpe_vocab.add(new_token)
        
        logger.info(f"BPE trained with {len(self.bpe_merges)} merges")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")
        
        # Clean and tokenize
        cleaned = self.clean_text(text)
        tokens = self.tokenize_text(cleaned)
        
        # Convert to IDs
        token_ids = []
        unk_id = self.special_token_ids[self.config.unk_token]
        
        for token in tokens:
            token_id = self.token_to_id.get(token, unk_id)
            token_ids.append(token_id)
        
        # Truncate if necessary
        if len(token_ids) > self.config.max_sequence_length:
            token_ids = token_ids[:self.config.max_sequence_length]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before decoding")
        
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.config.unk_token)
            
            # Skip special tokens in output
            if token not in [self.config.pad_token, self.config.bos_token, 
                           self.config.eos_token, self.config.mask_token]:
                tokens.append(token)
        
        # Join tokens based on strategy
        if self.config.tokenization_strategy == TokenizationStrategy.CHARACTER_BASED:
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def batch_encode(self, texts: List[str], pad_to_max_length: bool = True) -> List[List[int]]:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of texts to encode
            pad_to_max_length: Whether to pad to maximum length
            
        Returns:
            List of encoded sequences
        """
        encoded = [self.encode(text) for text in texts]
        
        if pad_to_max_length:
            # Pad all sequences to same length
            max_len = max(len(seq) for seq in encoded) if encoded else 0
            pad_id = self.special_token_ids[self.config.pad_token]
            
            padded = []
            for seq in encoded:
                padded_seq = seq + [pad_id] * (max_len - len(seq))
                padded.append(padded_seq)
            
            return padded
        
        return encoded
    
    def get_vocab_info(self) -> Dict[str, Any]:
        """Get vocabulary information."""
        if not self.is_fitted:
            return {'error': 'Tokenizer not fitted'}
        
        return {
            'vocab_size': len(self.vocabulary),
            'special_tokens': list(self.special_token_ids.keys()),
            'most_frequent_tokens': self.token_frequencies.most_common(10),
            'tokenization_strategy': self.config.tokenization_strategy.value,
            'config': {
                'lowercase': self.config.lowercase,
                'remove_punctuation': self.config.remove_punctuation,
                'min_token_frequency': self.config.min_token_frequency,
                'max_sequence_length': self.config.max_sequence_length
            }
        }
    
    def save(self, filepath: str):
        """Save preprocessor state to file."""
        import pickle
        
        state = {
            'config': self.config,
            'vocabulary': self.vocabulary,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'special_token_ids': self.special_token_ids,
            'token_frequencies': dict(self.token_frequencies),
            'bpe_merges': self.bpe_merges,
            'processing_stats': self.processing_stats,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AdvancedPreprocessor':
        """Load preprocessor from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create instance
        preprocessor = cls(state['config'])
        
        # Restore state
        preprocessor.vocabulary = state['vocabulary']
        preprocessor.token_to_id = state['token_to_id']
        preprocessor.id_to_token = state['id_to_token']
        preprocessor.special_token_ids = state['special_token_ids']
        preprocessor.token_frequencies = Counter(state['token_frequencies'])
        preprocessor.bpe_merges = state['bpe_merges']
        preprocessor.processing_stats = state['processing_stats']
        preprocessor.is_fitted = state['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def create_preprocessing_pipeline(
    tokenization_strategy: str = "word_based",
    vocab_size: int = 10000,
    max_sequence_length: int = 512,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    **kwargs
) -> AdvancedPreprocessor:
    """
    Factory function to create preprocessing pipeline.
    
    Args:
        tokenization_strategy: Strategy for tokenization
        vocab_size: Maximum vocabulary size
        max_sequence_length: Maximum sequence length
        lowercase: Whether to lowercase text
        remove_punctuation: Whether to remove punctuation
        **kwargs: Additional configuration options
        
    Returns:
        Configured AdvancedPreprocessor
    """
    config = PreprocessingConfig(
        tokenization_strategy=TokenizationStrategy(tokenization_strategy),
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        **kwargs
    )
    
    return AdvancedPreprocessor(config)


def demo_preprocessing():
    """Demonstrate preprocessing capabilities."""
    sample_texts = [
        "Hello, World! This is a sample text with UPPERCASE and números 123.",
        "Another example with émojis 😊 and special chars: @#$%^&*()",
        "Short text",
        "A" * 200,  # Very long text
        "Text with    multiple   spaces\t\tand\nnewlines"
    ]
    
    print("Advanced Preprocessing Demo")
    print("=" * 40)
    
    # Test different configurations
    configs = [
        {"lowercase": True, "remove_punctuation": False},
        {"lowercase": True, "remove_punctuation": True},
        {"tokenization_strategy": "character_based"},
        {"handle_emojis": "replace"}
    ]
    
    for i, config_params in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config_params}")
        print("-" * 20)
        
        preprocessor = create_preprocessing_pipeline(**config_params)
        
        # Fit on sample texts
        preprocessor.fit_tokenizer(sample_texts)
        
        # Test encoding/decoding
        test_text = sample_texts[0]
        encoded = preprocessor.encode(test_text)
        decoded = preprocessor.decode(encoded)
        
        print(f"Original: {test_text}")
        print(f"Cleaned:  {preprocessor.clean_text(test_text)}")
        print(f"Tokens:   {preprocessor.tokenize_text(preprocessor.clean_text(test_text))}")
        print(f"Encoded:  {encoded}")
        print(f"Decoded:  {decoded}")
        
        vocab_info = preprocessor.get_vocab_info()
        print(f"Vocab size: {vocab_info['vocab_size']}")
    
    return preprocessor