"""
Advanced sampling methods for text generation.

Implements Top-k, Nucleus (Top-p), Temperature, and Beam Search sampling
for high-quality and diverse text generation.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple
import heapq

logger = logging.getLogger(__name__)

BeamState = namedtuple('BeamState', ['tokens', 'score', 'log_prob'])


@dataclass
class SamplerConfig:
    """Configuration for sampling strategies."""
    
    # Temperature parameters
    temperature: float = 1.0
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    
    # Top-k parameters
    top_k: int = 50
    top_k_min: int = 1
    top_k_max: int = 100
    
    # Nucleus (Top-p) parameters  
    top_p: float = 0.9
    top_p_min: float = 0.1
    top_p_max: float = 1.0
    
    # Beam search parameters
    beam_width: int = 5
    beam_length_penalty: float = 1.0
    beam_coverage_penalty: float = 0.0
    
    # General parameters
    max_length: int = 200
    min_length: int = 10
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    # Repetition handling
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    
    def validate(self):
        """Validate configuration parameters."""
        assert 0.0 < self.temperature <= 5.0, f"Temperature must be in (0, 5], got {self.temperature}"
        assert 0 < self.top_k <= 1000, f"Top-k must be in (0, 1000], got {self.top_k}"
        assert 0.0 < self.top_p <= 1.0, f"Top-p must be in (0, 1], got {self.top_p}"
        assert 1 <= self.beam_width <= 20, f"Beam width must be in [1, 20], got {self.beam_width}"
        assert self.repetition_penalty >= 1.0, f"Repetition penalty must be >= 1.0, got {self.repetition_penalty}"


class BaseSampler(ABC):
    """Base class for all sampling methods."""
    
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.config.validate()
    
    @abstractmethod
    def sample(
        self,
        logits: tf.Tensor,
        sequence: tf.Tensor,
        step: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample next token from logits.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            sequence: Current sequence [batch_size, sequence_length]  
            step: Current generation step
            
        Returns:
            Tuple of (next_token, log_probability)
        """
        pass
    
    def _apply_repetition_penalty(
        self,
        logits: tf.Tensor,
        sequence: tf.Tensor
    ) -> tf.Tensor:
        """Apply repetition penalty to reduce repetitive generation."""
        if self.config.repetition_penalty == 1.0:
            return logits
            
        batch_size, vocab_size = tf.shape(logits)[0], tf.shape(logits)[1]
        penalty = self.config.repetition_penalty
        
        # Count occurrences of each token in sequence
        token_counts = tf.zeros([batch_size, vocab_size], dtype=tf.float32)
        
        for i in range(tf.shape(sequence)[1]):
            token = sequence[:, i]
            updates = tf.ones([batch_size], dtype=tf.float32)
            indices = tf.stack([tf.range(batch_size), token], axis=1)
            token_counts = tf.tensor_scatter_nd_add(token_counts, indices, updates)
        
        # Apply penalty - reduce probability of repeated tokens
        penalty_mask = tf.cast(token_counts > 0, tf.float32) * (penalty - 1.0)
        adjusted_logits = logits - penalty_mask * tf.nn.softmax(logits)
        
        return adjusted_logits
    
    def _apply_length_penalty(
        self,
        scores: tf.Tensor,
        length: int
    ) -> tf.Tensor:
        """Apply length penalty for beam search."""
        if self.config.beam_length_penalty == 1.0:
            return scores
            
        # Length penalty from Google's GNMT paper
        length_penalty = ((5.0 + length) / 6.0) ** self.config.beam_length_penalty
        return scores / length_penalty


class GreedySampler(BaseSampler):
    """Greedy sampling - always select most probable token."""
    
    def sample(
        self,
        logits: tf.Tensor,
        sequence: tf.Tensor,
        step: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample greedily from logits."""
        
        # Apply repetition penalty
        adjusted_logits = self._apply_repetition_penalty(logits, sequence)
        
        # Get most probable token
        next_token = tf.argmax(adjusted_logits, axis=-1, output_type=tf.int32)
        
        # Calculate log probability
        probs = tf.nn.softmax(adjusted_logits)
        log_prob = tf.nn.log_softmax(adjusted_logits)
        
        batch_indices = tf.range(tf.shape(next_token)[0])
        indices = tf.stack([batch_indices, next_token], axis=1)
        token_log_prob = tf.gather_nd(log_prob, indices)
        
        return next_token, token_log_prob


class TemperatureSampler(BaseSampler):
    """Temperature-based sampling for creativity control."""
    
    def sample(
        self,
        logits: tf.Tensor,
        sequence: tf.Tensor,
        step: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample with temperature scaling."""
        
        # Apply repetition penalty
        adjusted_logits = self._apply_repetition_penalty(logits, sequence)
        
        # Apply temperature scaling
        scaled_logits = adjusted_logits / self.config.temperature
        
        # Sample from categorical distribution
        next_token = tf.random.categorical(scaled_logits, 1, dtype=tf.int32)
        next_token = tf.squeeze(next_token, axis=-1)
        
        # Calculate log probability
        log_prob = tf.nn.log_softmax(scaled_logits)
        batch_indices = tf.range(tf.shape(next_token)[0])
        indices = tf.stack([batch_indices, next_token], axis=1)
        token_log_prob = tf.gather_nd(log_prob, indices)
        
        return next_token, token_log_prob


class TopKSampler(BaseSampler):
    """Top-k sampling - sample from k most probable tokens."""
    
    def sample(
        self,
        logits: tf.Tensor,
        sequence: tf.Tensor,
        step: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample from top-k most probable tokens."""
        
        # Apply repetition penalty
        adjusted_logits = self._apply_repetition_penalty(logits, sequence)
        
        # Apply temperature
        scaled_logits = adjusted_logits / self.config.temperature
        
        # Get top-k logits and indices
        top_k_logits, top_k_indices = tf.nn.top_k(scaled_logits, k=self.config.top_k)
        
        # Create mask for top-k tokens
        batch_size, vocab_size = tf.shape(scaled_logits)[0], tf.shape(scaled_logits)[1]
        
        # Set non-top-k logits to -inf
        mask = tf.fill([batch_size, vocab_size], -np.inf)
        
        # Update mask with top-k values
        batch_indices = tf.repeat(
            tf.range(batch_size)[:, None], 
            self.config.top_k, 
            axis=1
        )
        indices = tf.stack([
            tf.reshape(batch_indices, [-1]),
            tf.reshape(top_k_indices, [-1])
        ], axis=1)
        
        updates = tf.reshape(top_k_logits, [-1])
        filtered_logits = tf.tensor_scatter_nd_update(mask, indices, updates)
        
        # Sample from filtered distribution
        next_token = tf.random.categorical(filtered_logits, 1, dtype=tf.int32)
        next_token = tf.squeeze(next_token, axis=-1)
        
        # Calculate log probability
        log_prob = tf.nn.log_softmax(filtered_logits)
        batch_indices = tf.range(tf.shape(next_token)[0])
        token_indices = tf.stack([batch_indices, next_token], axis=1)
        token_log_prob = tf.gather_nd(log_prob, token_indices)
        
        return next_token, token_log_prob


class NucleusSampler(BaseSampler):
    """Nucleus (Top-p) sampling - sample from smallest set with cumulative probability >= p."""
    
    def sample(
        self,
        logits: tf.Tensor,
        sequence: tf.Tensor,
        step: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample using nucleus (top-p) strategy."""
        
        # Apply repetition penalty
        adjusted_logits = self._apply_repetition_penalty(logits, sequence)
        
        # Apply temperature
        scaled_logits = adjusted_logits / self.config.temperature
        
        # Sort by probability (descending)
        sorted_logits, sorted_indices = tf.nn.top_k(
            scaled_logits, 
            k=tf.shape(scaled_logits)[-1]
        )
        
        # Calculate cumulative probabilities
        sorted_probs = tf.nn.softmax(sorted_logits)
        cumsum_probs = tf.cumsum(sorted_probs, axis=-1)
        
        # Create nucleus mask - keep tokens until cumsum > p
        nucleus_mask = cumsum_probs <= self.config.top_p
        
        # Always keep at least the first token (highest probability)
        first_token_mask = tf.one_hot(0, tf.shape(nucleus_mask)[-1], dtype=tf.bool)
        first_token_mask = tf.broadcast_to(first_token_mask, tf.shape(nucleus_mask))
        nucleus_mask = tf.logical_or(nucleus_mask, first_token_mask)
        
        # Filter out tokens outside nucleus
        filtered_logits = tf.where(
            nucleus_mask,
            sorted_logits,
            tf.fill(tf.shape(sorted_logits), -np.inf)
        )
        
        # Sample from filtered distribution
        selected_indices = tf.random.categorical(filtered_logits, 1, dtype=tf.int32)
        selected_indices = tf.squeeze(selected_indices, axis=-1)
        
        # Get original token indices
        batch_indices = tf.range(tf.shape(selected_indices)[0])
        gather_indices = tf.stack([batch_indices, selected_indices], axis=1)
        next_token = tf.gather_nd(sorted_indices, gather_indices)
        
        # Calculate log probability
        log_prob = tf.nn.log_softmax(scaled_logits)
        token_indices = tf.stack([batch_indices, next_token], axis=1)
        token_log_prob = tf.gather_nd(log_prob, token_indices)
        
        return next_token, token_log_prob


class BeamSearchSampler(BaseSampler):
    """Beam search for deterministic high-quality generation."""
    
    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        self.beam_width = config.beam_width
        
    def beam_search(
        self,
        model: tf.keras.Model,
        initial_tokens: tf.Tensor,
        max_length: Optional[int] = None
    ) -> List[BeamState]:
        """
        Perform beam search generation.
        
        Args:
            model: Language model for scoring
            initial_tokens: Starting tokens [batch_size, initial_length]
            max_length: Maximum sequence length
            
        Returns:
            List of beam states sorted by score
        """
        max_length = max_length or self.config.max_length
        batch_size = tf.shape(initial_tokens)[0].numpy()
        
        if batch_size != 1:
            raise ValueError("Beam search currently only supports batch_size=1")
        
        # Initialize beams
        initial_tokens_list = initial_tokens[0].numpy().tolist()
        beams = [BeamState(
            tokens=initial_tokens_list,
            score=0.0,
            log_prob=0.0
        )]
        
        for step in range(max_length - len(initial_tokens_list)):
            candidates = []
            
            for beam in beams:
                if beam.tokens[-1] == self.config.eos_token_id:
                    # Finished beam
                    candidates.append(beam)
                    continue
                
                # Get model predictions for this beam
                input_tokens = tf.constant([beam.tokens], dtype=tf.int32)
                logits = model(input_tokens)
                
                if len(logits.shape) == 3:
                    logits = logits[0, -1, :]  # [vocab_size]
                else:
                    logits = logits[0, :]
                
                # Apply temperature and get top-k candidates
                scaled_logits = logits / self.config.temperature
                log_probs = tf.nn.log_softmax(scaled_logits)
                
                top_k_log_probs, top_k_indices = tf.nn.top_k(
                    log_probs, 
                    k=min(self.beam_width * 2, tf.shape(log_probs)[0])
                )
                
                for i in range(tf.shape(top_k_indices)[0]):
                    token_id = top_k_indices[i].numpy()
                    token_log_prob = top_k_log_probs[i].numpy()
                    
                    new_tokens = beam.tokens + [token_id]
                    new_log_prob = beam.log_prob + token_log_prob
                    new_score = new_log_prob
                    
                    # Apply length penalty
                    if self.config.beam_length_penalty != 1.0:
                        new_score = self._apply_length_penalty(
                            tf.constant([new_score]), 
                            len(new_tokens)
                        )[0].numpy()
                    
                    candidates.append(BeamState(
                        tokens=new_tokens,
                        score=new_score,
                        log_prob=new_log_prob
                    ))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x.score, reverse=True)
            beams = candidates[:self.beam_width]
            
            # Check if all beams are finished
            if all(beam.tokens[-1] == self.config.eos_token_id for beam in beams):
                break
        
        # Return sorted beams
        beams.sort(key=lambda x: x.score, reverse=True)
        return beams
    
    def sample(
        self,
        logits: tf.Tensor,
        sequence: tf.Tensor,
        step: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        For compatibility - beam search needs full model access.
        This method returns top-1 prediction.
        """
        logger.warning("BeamSearchSampler.sample() called - use beam_search() for full functionality")
        
        # Apply repetition penalty
        adjusted_logits = self._apply_repetition_penalty(logits, sequence)
        
        # Apply temperature
        scaled_logits = adjusted_logits / self.config.temperature
        
        # Get most probable token
        next_token = tf.argmax(scaled_logits, axis=-1, output_type=tf.int32)
        
        # Calculate log probability
        log_prob = tf.nn.log_softmax(scaled_logits)
        batch_indices = tf.range(tf.shape(next_token)[0])
        indices = tf.stack([batch_indices, next_token], axis=1)
        token_log_prob = tf.gather_nd(log_prob, indices)
        
        return next_token, token_log_prob


def create_sampler(method: str, config: SamplerConfig) -> BaseSampler:
    """
    Factory function to create samplers.
    
    Args:
        method: Sampling method ('greedy', 'temperature', 'top_k', 'nucleus', 'beam')
        config: Sampler configuration
        
    Returns:
        Configured sampler instance
    """
    samplers = {
        'greedy': GreedySampler,
        'temperature': TemperatureSampler,
        'top_k': TopKSampler,
        'top_p': NucleusSampler,
        'nucleus': NucleusSampler,
        'beam_search': BeamSearchSampler,
        'beam': BeamSearchSampler
    }
    
    if method not in samplers:
        raise ValueError(f"Unknown sampling method: {method}. Available: {list(samplers.keys())}")
    
    return samplers[method](config)


def compare_sampling_methods(
    model: tf.keras.Model,
    prompt: str,
    tokenizer,
    methods: List[str],
    config: SamplerConfig,
    num_samples: int = 3
) -> Dict[str, List[str]]:
    """
    Compare different sampling methods on the same prompt.
    
    Args:
        model: Language model
        prompt: Input prompt
        tokenizer: Tokenizer for encoding/decoding
        methods: List of sampling methods to compare
        config: Sampling configuration
        num_samples: Number of samples per method
        
    Returns:
        Dictionary mapping method names to generated texts
    """
    results = {}
    
    # Encode prompt
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(prompt)
    else:
        # Fallback for simple tokenizers
        input_ids = [tokenizer.word_to_index.get(word, 1) for word in prompt.split()]
    
    input_tensor = tf.constant([input_ids], dtype=tf.int32)
    
    for method in methods:
        logger.info(f"Generating samples with {method} sampling...")
        sampler = create_sampler(method, config)
        method_results = []
        
        for i in range(num_samples):
            if method == 'beam_search':
                # Beam search returns multiple candidates
                beams = sampler.beam_search(model, input_tensor, config.max_length)
                if beams:
                    tokens = beams[0].tokens  # Best beam
                    if hasattr(tokenizer, 'decode'):
                        text = tokenizer.decode(tokens)
                    else:
                        # Fallback for simple tokenizers
                        words = [tokenizer.index_to_word.get(idx, '<UNK>') for idx in tokens]
                        text = ' '.join(words)
                    method_results.append(text)
            else:
                # Regular sampling
                current_seq = input_tensor
                
                for step in range(config.max_length - tf.shape(input_tensor)[1]):
                    # Get model logits
                    logits = model(current_seq)
                    if len(logits.shape) == 3:
                        logits = logits[:, -1, :]  # [batch_size, vocab_size]
                    
                    # Sample next token
                    next_token, _ = sampler.sample(logits, current_seq, step)
                    
                    # Append to sequence
                    next_token = tf.expand_dims(next_token, axis=1)
                    current_seq = tf.concat([current_seq, next_token], axis=1)
                    
                    # Check for EOS
                    if next_token[0, 0].numpy() == config.eos_token_id:
                        break
                
                # Decode generated sequence
                tokens = current_seq[0].numpy().tolist()
                if hasattr(tokenizer, 'decode'):
                    text = tokenizer.decode(tokens)
                else:
                    words = [tokenizer.index_to_word.get(idx, '<UNK>') for idx in tokens]
                    text = ' '.join(words)
                
                method_results.append(text)
        
        results[method] = method_results
    
    return results