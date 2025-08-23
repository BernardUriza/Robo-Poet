"""
Text data augmentation for improved model robustness and generalization.

Implements various augmentation techniques specifically designed for text
while maintaining linguistic coherence and semantic meaning.
"""

import random
import logging
import re
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import string

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Types of text augmentation."""
    SYNONYM_REPLACEMENT = "synonym_replacement"
    RANDOM_INSERTION = "random_insertion"
    RANDOM_SWAP = "random_swap"
    RANDOM_DELETION = "random_deletion"
    BACK_TRANSLATION = "back_translation"
    PARAPHRASING = "paraphrasing"
    NOISE_INJECTION = "noise_injection"
    SEQUENCE_SHUFFLING = "sequence_shuffling"
    TOKEN_MASKING = "token_masking"
    SPAN_MASKING = "span_masking"


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation."""
    
    # General settings
    augmentation_probability: float = 0.3
    preserve_length: bool = True
    min_augmentations_per_sample: int = 1
    max_augmentations_per_sample: int = 3
    
    # Synonym replacement
    synonym_prob: float = 0.1
    synonym_max_replacements: int = 3
    
    # Random insertion
    insertion_prob: float = 0.05
    insertion_max_insertions: int = 2
    
    # Random swap
    swap_prob: float = 0.1
    swap_max_swaps: int = 2
    swap_distance: int = 3  # Max distance for swapping
    
    # Random deletion
    deletion_prob: float = 0.05
    deletion_max_deletions: int = 2
    
    # Noise injection
    noise_prob: float = 0.02
    char_noise_prob: float = 0.001  # Per character
    
    # Token masking (BERT-style)
    mask_prob: float = 0.15
    mask_token_id: int = 3
    random_token_prob: float = 0.1  # Replace with random instead of mask
    
    # Span masking
    span_mask_prob: float = 0.1
    max_span_length: int = 5
    
    # Sequence shuffling
    shuffle_prob: float = 0.05
    shuffle_window: int = 10
    
    # Special tokens to preserve
    special_tokens: List[int] = None
    preserve_first_token: bool = True
    preserve_last_token: bool = True
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [0, 1, 2, 3]  # PAD, UNK, EOS, MASK
        
    def validate(self):
        """Validate configuration parameters."""
        assert 0 <= self.augmentation_probability <= 1
        assert 0 <= self.synonym_prob <= 1
        assert 0 <= self.insertion_prob <= 1
        assert 0 <= self.swap_prob <= 1
        assert 0 <= self.deletion_prob <= 1
        assert 0 <= self.noise_prob <= 1
        assert 0 <= self.mask_prob <= 1


class TextAugmenter:
    """Advanced text augmentation with multiple strategies."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None, vocab_size: int = 10000):
        self.config = config or AugmentationConfig()
        self.config.validate()
        self.vocab_size = vocab_size
        
        # Initialize augmentation functions
        self.augmentation_functions = {
            AugmentationType.SYNONYM_REPLACEMENT: self._synonym_replacement,
            AugmentationType.RANDOM_INSERTION: self._random_insertion,
            AugmentationType.RANDOM_SWAP: self._random_swap,
            AugmentationType.RANDOM_DELETION: self._random_deletion,
            AugmentationType.NOISE_INJECTION: self._noise_injection,
            AugmentationType.TOKEN_MASKING: self._token_masking,
            AugmentationType.SPAN_MASKING: self._span_masking,
            AugmentationType.SEQUENCE_SHUFFLING: self._sequence_shuffling
        }
        
        # Simple synonym dictionary (would be expanded with real data)
        self._init_simple_synonyms()
        
        logger.info(f"TextAugmenter initialized with {len(self.augmentation_functions)} techniques")
    
    def _init_simple_synonyms(self):
        """Initialize simple synonym mapping for demonstration."""
        # In a real implementation, this would load from WordNet, word2vec, etc.
        self.synonyms = {
            # These would be token IDs in practice
            "good": ["great", "excellent", "fine", "nice"],
            "bad": ["terrible", "awful", "poor", "horrible"],
            "big": ["large", "huge", "enormous", "giant"],
            "small": ["tiny", "little", "mini", "compact"],
            "fast": ["quick", "rapid", "swift", "speedy"],
            "slow": ["sluggish", "gradual", "leisurely", "steady"]
        }
    
    def augment_sequence(
        self, 
        tokens: List[int], 
        num_augmentations: Optional[int] = None
    ) -> List[List[int]]:
        """
        Apply augmentation to a token sequence.
        
        Args:
            tokens: Input token sequence
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented sequences including original
        """
        if not tokens or len(tokens) < 2:
            return [tokens]  # Return original for very short sequences
        
        augmented_sequences = [tokens]  # Include original
        
        # Determine number of augmentations
        if num_augmentations is None:
            num_augmentations = random.randint(
                self.config.min_augmentations_per_sample,
                self.config.max_augmentations_per_sample
            )
        
        # Apply random augmentations
        for _ in range(num_augmentations):
            # Choose random augmentation technique
            aug_type = random.choice(list(self.augmentation_functions.keys()))
            aug_func = self.augmentation_functions[aug_type]
            
            # Apply augmentation
            try:
                augmented = aug_func(tokens.copy())
                if augmented and augmented != tokens:
                    augmented_sequences.append(augmented)
            except Exception as e:
                logger.debug(f"Augmentation {aug_type.value} failed: {e}")
        
        return augmented_sequences
    
    def _synonym_replacement(self, tokens: List[int]) -> List[int]:
        """Replace random tokens with synonyms."""
        if random.random() > self.config.synonym_prob:
            return tokens
        
        # In a real implementation, would use actual synonym lookup
        # For now, simulate by replacing with nearby token IDs
        replacements = 0
        result = tokens.copy()
        
        for i in range(len(result)):
            if replacements >= self.config.synonym_max_replacements:
                break
                
            if (result[i] not in self.config.special_tokens and
                random.random() < 0.2):  # 20% chance per eligible token
                
                # Simple synonym simulation - nearby token ID
                synonym_id = result[i] + random.randint(-5, 5)
                synonym_id = max(len(self.config.special_tokens), 
                               min(synonym_id, self.vocab_size - 1))
                
                result[i] = synonym_id
                replacements += 1
        
        return result
    
    def _random_insertion(self, tokens: List[int]) -> List[int]:
        """Insert random tokens at random positions."""
        if random.random() > self.config.insertion_prob:
            return tokens
        
        result = tokens.copy()
        insertions = 0
        
        for _ in range(self.config.insertion_max_insertions):
            if insertions >= self.config.insertion_max_insertions:
                break
                
            # Choose insertion position (avoid start/end if configured)
            start_idx = 1 if self.config.preserve_first_token else 0
            end_idx = len(result) - 1 if self.config.preserve_last_token else len(result)
            
            if end_idx <= start_idx:
                break
                
            insert_pos = random.randint(start_idx, end_idx)
            
            # Insert a random token (avoid special tokens)
            random_token = random.randint(
                len(self.config.special_tokens),
                self.vocab_size - 1
            )
            
            result.insert(insert_pos, random_token)
            insertions += 1
        
        # Truncate if preserve_length is True
        if self.config.preserve_length and len(result) > len(tokens):
            result = result[:len(tokens)]
        
        return result
    
    def _random_swap(self, tokens: List[int]) -> List[int]:
        """Randomly swap adjacent or nearby tokens."""
        if random.random() > self.config.swap_prob or len(tokens) < 2:
            return tokens
        
        result = tokens.copy()
        swaps = 0
        
        for _ in range(self.config.swap_max_swaps):
            if swaps >= self.config.swap_max_swaps:
                break
                
            # Choose positions to swap
            start_idx = 1 if self.config.preserve_first_token else 0
            end_idx = len(result) - 2 if self.config.preserve_last_token else len(result) - 1
            
            if end_idx <= start_idx:
                break
                
            pos1 = random.randint(start_idx, end_idx)
            
            # Find swap partner within distance
            min_pos2 = max(start_idx, pos1 - self.config.swap_distance)
            max_pos2 = min(end_idx + 1, pos1 + self.config.swap_distance + 1)
            
            if max_pos2 <= min_pos2:
                continue
                
            pos2 = random.randint(min_pos2, max_pos2 - 1)
            if pos2 == pos1:
                continue
            
            # Avoid swapping special tokens
            if (result[pos1] in self.config.special_tokens or
                result[pos2] in self.config.special_tokens):
                continue
            
            # Perform swap
            result[pos1], result[pos2] = result[pos2], result[pos1]
            swaps += 1
        
        return result
    
    def _random_deletion(self, tokens: List[int]) -> List[int]:
        """Randomly delete tokens."""
        if random.random() > self.config.deletion_prob or len(tokens) < 3:
            return tokens
        
        result = tokens.copy()
        deletions = 0
        
        # Indices that can be deleted
        deletable_indices = []
        for i in range(len(result)):
            if (result[i] not in self.config.special_tokens and
                not (i == 0 and self.config.preserve_first_token) and
                not (i == len(result) - 1 and self.config.preserve_last_token)):
                deletable_indices.append(i)
        
        # Randomly select indices to delete
        num_to_delete = min(
            self.config.deletion_max_deletions,
            len(deletable_indices) // 2  # Don't delete more than half
        )
        
        if num_to_delete > 0:
            indices_to_delete = random.sample(deletable_indices, num_to_delete)
            # Sort in reverse order to maintain index validity
            indices_to_delete.sort(reverse=True)
            
            for idx in indices_to_delete:
                if idx < len(result):
                    del result[idx]
                    deletions += 1
        
        return result
    
    def _noise_injection(self, tokens: List[int]) -> List[int]:
        """Inject noise by slightly modifying token IDs."""
        if random.random() > self.config.noise_prob:
            return tokens
        
        result = tokens.copy()
        
        for i in range(len(result)):
            if (result[i] not in self.config.special_tokens and
                random.random() < self.config.char_noise_prob):
                
                # Add small noise to token ID
                noise = random.randint(-2, 2)
                noisy_token = result[i] + noise
                noisy_token = max(len(self.config.special_tokens), 
                                min(noisy_token, self.vocab_size - 1))
                
                result[i] = noisy_token
        
        return result
    
    def _token_masking(self, tokens: List[int]) -> List[int]:
        """Apply BERT-style token masking."""
        if random.random() > self.config.mask_prob:
            return tokens
        
        result = tokens.copy()
        num_to_mask = max(1, int(len(tokens) * 0.15))  # Mask ~15% of tokens
        
        # Choose positions to mask
        maskable_positions = []
        for i in range(len(tokens)):
            if (tokens[i] not in self.config.special_tokens and
                not (i == 0 and self.config.preserve_first_token) and
                not (i == len(tokens) - 1 and self.config.preserve_last_token)):
                maskable_positions.append(i)
        
        if not maskable_positions:
            return result
        
        positions_to_mask = random.sample(
            maskable_positions, 
            min(num_to_mask, len(maskable_positions))
        )
        
        for pos in positions_to_mask:
            rand_val = random.random()
            
            if rand_val < 0.8:
                # 80% of time: replace with mask token
                result[pos] = self.config.mask_token_id
            elif rand_val < 0.9:
                # 10% of time: replace with random token
                result[pos] = random.randint(
                    len(self.config.special_tokens),
                    self.vocab_size - 1
                )
            # 10% of time: keep original (no change)
        
        return result
    
    def _span_masking(self, tokens: List[int]) -> List[int]:
        """Mask spans of consecutive tokens."""
        if random.random() > self.config.span_mask_prob or len(tokens) < 3:
            return tokens
        
        result = tokens.copy()
        
        # Choose span start and length
        start_idx = 1 if self.config.preserve_first_token else 0
        end_idx = len(result) - 1 if self.config.preserve_last_token else len(result)
        
        if end_idx <= start_idx:
            return result
        
        span_length = random.randint(1, min(self.config.max_span_length, end_idx - start_idx))
        span_start = random.randint(start_idx, end_idx - span_length)
        
        # Replace span with mask tokens
        for i in range(span_start, span_start + span_length):
            if result[i] not in self.config.special_tokens:
                result[i] = self.config.mask_token_id
        
        return result
    
    def _sequence_shuffling(self, tokens: List[int]) -> List[int]:
        """Shuffle tokens within local windows."""
        if random.random() > self.config.shuffle_prob or len(tokens) < 4:
            return tokens
        
        result = tokens.copy()
        window_size = min(self.config.shuffle_window, len(tokens) // 2)
        
        if window_size < 2:
            return result
        
        # Choose shuffle window
        start_idx = 1 if self.config.preserve_first_token else 0
        end_idx = len(result) - 1 if self.config.preserve_last_token else len(result)
        
        if end_idx - start_idx < window_size:
            return result
        
        window_start = random.randint(start_idx, end_idx - window_size)
        window_end = window_start + window_size
        
        # Extract window and shuffle
        window_tokens = result[window_start:window_end]
        non_special = [t for t in window_tokens if t not in self.config.special_tokens]
        special_positions = [(i, t) for i, t in enumerate(window_tokens) 
                           if t in self.config.special_tokens]
        
        # Shuffle non-special tokens
        random.shuffle(non_special)
        
        # Reconstruct window
        shuffled_window = window_tokens.copy()
        non_special_idx = 0
        
        for i in range(len(shuffled_window)):
            if shuffled_window[i] not in self.config.special_tokens:
                shuffled_window[i] = non_special[non_special_idx]
                non_special_idx += 1
        
        # Replace window in result
        result[window_start:window_end] = shuffled_window
        
        return result
    
    def augment_batch(
        self, 
        batch_tokens: List[List[int]], 
        augmentation_probability: Optional[float] = None
    ) -> List[List[int]]:
        """
        Apply augmentation to a batch of sequences.
        
        Args:
            batch_tokens: List of token sequences
            augmentation_probability: Probability of augmenting each sequence
            
        Returns:
            List of potentially augmented sequences
        """
        if augmentation_probability is None:
            augmentation_probability = self.config.augmentation_probability
        
        augmented_batch = []
        
        for tokens in batch_tokens:
            if random.random() < augmentation_probability:
                # Apply augmentation
                augmented_versions = self.augment_sequence(tokens, num_augmentations=1)
                # Use the first augmented version (excluding original)
                if len(augmented_versions) > 1:
                    augmented_batch.append(augmented_versions[1])
                else:
                    augmented_batch.append(tokens)
            else:
                # Keep original
                augmented_batch.append(tokens)
        
        return augmented_batch
    
    def create_augmentation_dataset(
        self, 
        dataset: tf.data.Dataset,
        augmentation_factor: int = 2
    ) -> tf.data.Dataset:
        """
        Create augmented dataset using tf.data operations.
        
        Args:
            dataset: Original dataset
            augmentation_factor: How many augmented versions per original
            
        Returns:
            Augmented dataset
        """
        def augment_tf_function(input_seq, target_seq):
            """TensorFlow function for augmentation."""
            
            def py_augment(tokens):
                """Python function to perform augmentation."""
                tokens_list = tokens.numpy().tolist()
                augmented_versions = self.augment_sequence(
                    tokens_list, 
                    num_augmentations=augmentation_factor - 1
                )
                
                # Return all versions as a batch
                max_len = max(len(seq) for seq in augmented_versions)
                
                # Pad sequences to same length
                padded_versions = []
                for seq in augmented_versions:
                    padded = seq + [0] * (max_len - len(seq))  # Pad with 0
                    padded_versions.append(padded)
                
                return np.array(padded_versions, dtype=np.int32)
            
            # Apply augmentation to input sequence
            augmented_inputs = tf.py_function(
                py_augment,
                [input_seq],
                tf.int32
            )
            
            # For targets, we might want the same augmentation
            augmented_targets = tf.py_function(
                py_augment,
                [target_seq],
                tf.int32
            )
            
            # Set shapes
            augmented_inputs.set_shape([None, None])
            augmented_targets.set_shape([None, None])
            
            return augmented_inputs, augmented_targets
        
        # Apply augmentation and flatten
        augmented_dataset = dataset.map(
            augment_tf_function,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Flatten the augmented versions
        augmented_dataset = augmented_dataset.flat_map(
            lambda x, y: tf.data.Dataset.from_tensor_slices((x, y))
        )
        
        return augmented_dataset
    
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about available augmentation techniques."""
        return {
            'available_techniques': len(self.augmentation_functions),
            'techniques': list(self.augmentation_functions.keys()),
            'config': {
                'augmentation_probability': self.config.augmentation_probability,
                'preserve_length': self.config.preserve_length,
                'vocab_size': self.vocab_size
            }
        }


def create_augmented_dataset(
    dataset: tf.data.Dataset,
    vocab_size: int = 10000,
    augmentation_probability: float = 0.3,
    augmentation_factor: int = 2,
    **config_kwargs
) -> tf.data.Dataset:
    """
    Factory function to create augmented dataset.
    
    Args:
        dataset: Original dataset
        vocab_size: Vocabulary size
        augmentation_probability: Probability of augmentation
        augmentation_factor: Number of augmented versions per sample
        **config_kwargs: Additional configuration options
        
    Returns:
        Augmented dataset
    """
    config = AugmentationConfig(
        augmentation_probability=augmentation_probability,
        **config_kwargs
    )
    
    augmenter = TextAugmenter(config, vocab_size)
    
    return augmenter.create_augmentation_dataset(dataset, augmentation_factor)


def demo_augmentation():
    """Demonstrate text augmentation capabilities."""
    # Example tokens (would be real token IDs in practice)
    example_tokens = [10, 25, 67, 123, 89, 45, 156, 78, 234, 67, 89, 12]
    
    config = AugmentationConfig(
        augmentation_probability=1.0,  # Always augment for demo
        synonym_prob=0.3,
        insertion_prob=0.2,
        swap_prob=0.3,
        deletion_prob=0.2,
        mask_prob=0.2
    )
    
    augmenter = TextAugmenter(config, vocab_size=1000)
    
    print("Text Augmentation Demo")
    print("=" * 30)
    print(f"Original: {example_tokens}")
    print()
    
    # Generate multiple augmented versions
    for i in range(5):
        augmented_versions = augmenter.augment_sequence(example_tokens, num_augmentations=1)
        
        for j, aug_seq in enumerate(augmented_versions[1:], 1):  # Skip original
            print(f"Aug {i+1}.{j}: {aug_seq}")
    
    return augmenter