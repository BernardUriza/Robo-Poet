"""
Advanced text generator with multi-mode generation capabilities.

Integrates all sampling methods, temperature scheduling, and style conditioning
for professional-grade text generation with fine-grained control.
"""

import numpy as np
import tensorflow as tf
import logging
import time
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from .samplers import (
    BaseSampler, create_sampler, SamplerConfig,
    GreedySampler, TopKSampler, NucleusSampler, 
    TemperatureSampler, BeamSearchSampler
)
from .temperature_scheduler import (
    TemperatureScheduler, create_scheduler, SchedulerConfig,
    TemperatureCallback, calculate_ngram_diversity
)

logger = logging.getLogger(__name__)


class GenerationMode(Enum):
    """Generation modes for different use cases."""
    GREEDY = "greedy"
    SAMPLING = "sampling"  
    TOP_K = "top_k"
    NUCLEUS = "nucleus"
    BEAM_SEARCH = "beam_search"
    MIXED = "mixed"  # Combines multiple methods


class StyleType(Enum):
    """Style conditioning types."""
    FORMAL = "formal"
    CASUAL = "casual"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    POETIC = "poetic"
    NARRATIVE = "narrative"
    NONE = "none"


@dataclass
class GenerationConfig:
    """Comprehensive configuration for text generation."""
    
    # Basic parameters
    max_length: int = 200
    min_length: int = 10
    
    # Sampling configuration
    generation_mode: GenerationMode = GenerationMode.NUCLEUS
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    
    # Temperature scheduling
    use_temperature_scheduling: bool = True
    scheduler_type: str = "adaptive"
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Style conditioning
    style_type: StyleType = StyleType.NONE
    style_strength: float = 0.5  # 0.0 = no conditioning, 1.0 = strong conditioning
    
    # Quality control
    diversity_threshold: float = 0.7  # Minimum n-gram diversity
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    
    # Special tokens
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 1
    
    # Advanced features
    use_mixed_mode: bool = False  # Combine different sampling methods
    mixed_mode_weights: Dict[str, float] = field(default_factory=lambda: {
        'greedy': 0.2,
        'nucleus': 0.5,
        'top_k': 0.3
    })
    
    # Monitoring and logging
    log_probabilities: bool = False
    track_diversity: bool = True
    save_generation_stats: bool = False
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.max_length > self.min_length
        assert 0.0 <= self.style_strength <= 1.0
        assert 0.0 < self.diversity_threshold <= 1.0
        assert self.repetition_penalty >= 1.0
        
        if self.use_mixed_mode:
            weights_sum = sum(self.mixed_mode_weights.values())
            assert abs(weights_sum - 1.0) < 1e-6, f"Mixed mode weights must sum to 1.0, got {weights_sum}"


@dataclass 
class GenerationResult:
    """Result of text generation with metadata."""
    
    # Generated content
    text: str
    tokens: List[int]
    
    # Generation metadata
    generation_time: float
    tokens_per_second: float
    
    # Quality metrics
    perplexity: Optional[float] = None
    diversity_score: float = 0.0
    repetition_score: float = 0.0
    
    # Probability information
    token_probabilities: Optional[List[float]] = None
    sequence_log_probability: Optional[float] = None
    
    # Generation details
    mode_used: GenerationMode = GenerationMode.NUCLEUS
    temperature_history: List[float] = field(default_factory=list)
    
    # Sampling statistics
    sampling_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'text': self.text,
            'tokens': self.tokens,
            'generation_time': self.generation_time,
            'tokens_per_second': self.tokens_per_second,
            'perplexity': self.perplexity,
            'diversity_score': self.diversity_score,
            'repetition_score': self.repetition_score,
            'mode_used': self.mode_used.value,
            'temperature_history': self.temperature_history,
            'sampling_stats': self.sampling_stats
        }
    
    def save(self, filepath: str):
        """Save generation result to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class StyleConditioner:
    """Handles style conditioning for text generation."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.style_embeddings = {}
        self._initialize_style_templates()
        
    def _initialize_style_templates(self):
        """Initialize style conditioning templates."""
        # Style-specific token preferences (simplified approach)
        self.style_preferences = {
            StyleType.FORMAL: {
                'boost_tokens': [],  # Tokens to boost probability
                'suppress_tokens': [],  # Tokens to suppress
                'temperature_modifier': -0.1  # Make slightly less random
            },
            StyleType.CASUAL: {
                'boost_tokens': [],
                'suppress_tokens': [],
                'temperature_modifier': 0.1  # Make slightly more random
            },
            StyleType.CREATIVE: {
                'boost_tokens': [],
                'suppress_tokens': [],
                'temperature_modifier': 0.2  # More creative/random
            },
            StyleType.TECHNICAL: {
                'boost_tokens': [],
                'suppress_tokens': [],
                'temperature_modifier': -0.2  # More focused/deterministic
            },
            StyleType.POETIC: {
                'boost_tokens': [],
                'suppress_tokens': [],
                'temperature_modifier': 0.15  # Moderately creative
            },
            StyleType.NARRATIVE: {
                'boost_tokens': [],
                'suppress_tokens': [],
                'temperature_modifier': 0.0  # Neutral
            }
        }
    
    def condition_logits(
        self,
        logits: tf.Tensor,
        style_type: StyleType,
        strength: float = 0.5
    ) -> tf.Tensor:
        """
        Apply style conditioning to model logits.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            style_type: Target style
            strength: Conditioning strength (0.0 = no effect, 1.0 = full effect)
            
        Returns:
            Style-conditioned logits
        """
        if style_type == StyleType.NONE or strength == 0.0:
            return logits
            
        # Get style preferences
        preferences = self.style_preferences.get(style_type, {})
        
        # Apply token-level adjustments (placeholder - would need trained style vectors)
        conditioned_logits = logits
        
        # Apply temperature modification
        temp_modifier = preferences.get('temperature_modifier', 0.0)
        if temp_modifier != 0.0:
            adjustment_factor = 1.0 + (temp_modifier * strength)
            conditioned_logits = conditioned_logits / adjustment_factor
        
        return conditioned_logits
    
    def get_temperature_adjustment(
        self, 
        style_type: StyleType, 
        strength: float = 0.5
    ) -> float:
        """Get temperature adjustment for style."""
        if style_type == StyleType.NONE:
            return 0.0
            
        preferences = self.style_preferences.get(style_type, {})
        temp_modifier = preferences.get('temperature_modifier', 0.0)
        
        return temp_modifier * strength


class AdvancedTextGenerator:
    """Advanced text generator with comprehensive generation capabilities."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        tokenizer,
        config: Optional[GenerationConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.config.validate()
        
        # Initialize components
        self.sampler = self._create_sampler()
        self.scheduler = self._create_scheduler() if self.config.use_temperature_scheduling else None
        self.style_conditioner = StyleConditioner(self._get_vocab_size())
        
        # State tracking
        self.generation_history = []
        self.current_generation_stats = {}
        
        logger.info(f"Advanced text generator initialized with {self.config.generation_mode.value} mode")
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size from tokenizer or model."""
        if hasattr(self.tokenizer, 'vocab_size'):
            return self.tokenizer.vocab_size
        elif hasattr(self.model, 'output_shape'):
            return self.model.output_shape[-1]
        else:
            return 10000  # Default fallback
    
    def _create_sampler(self) -> BaseSampler:
        """Create sampler based on configuration."""
        return create_sampler(self.config.generation_mode.value, self.config.sampler_config)
    
    def _create_scheduler(self) -> Optional[TemperatureScheduler]:
        """Create temperature scheduler if enabled."""
        if not self.config.use_temperature_scheduling:
            return None
            
        return create_scheduler(
            self.config.scheduler_type,
            self.config.scheduler_config
        )
    
    def generate(
        self,
        prompt: str,
        **generation_kwargs
    ) -> GenerationResult:
        """
        Generate text with advanced sampling and conditioning.
        
        Args:
            prompt: Input prompt text
            **generation_kwargs: Override generation parameters
            
        Returns:
            GenerationResult with text and metadata
        """
        start_time = time.time()
        
        # Override config with any provided kwargs
        effective_config = self._merge_config(generation_kwargs)
        
        # Encode prompt
        input_tokens = self._encode_prompt(prompt)
        
        # Generate based on mode
        if effective_config.generation_mode == GenerationMode.BEAM_SEARCH:
            result_tokens, generation_stats = self._generate_beam_search(input_tokens, effective_config)
        elif effective_config.use_mixed_mode:
            result_tokens, generation_stats = self._generate_mixed_mode(input_tokens, effective_config)
        else:
            result_tokens, generation_stats = self._generate_sampling(input_tokens, effective_config)
        
        # Decode result
        generated_text = self._decode_tokens(result_tokens)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        tokens_per_second = len(result_tokens) / generation_time if generation_time > 0 else 0
        
        # Create result
        result = GenerationResult(
            text=generated_text,
            tokens=result_tokens,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            mode_used=effective_config.generation_mode,
            sampling_stats=generation_stats
        )
        
        # Calculate quality metrics
        self._calculate_quality_metrics(result)
        
        # Store in history
        self.generation_history.append(result)
        
        return result
    
    def _merge_config(self, kwargs: Dict[str, Any]) -> GenerationConfig:
        """Merge generation kwargs with base config."""
        # Create copy of base config
        import copy
        effective_config = copy.deepcopy(self.config)
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            if hasattr(effective_config, key):
                setattr(effective_config, key, value)
        
        return effective_config
    
    def _encode_prompt(self, prompt: str) -> List[int]:
        """Encode prompt to token IDs."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(prompt)
        else:
            # Fallback for simple tokenizers
            words = prompt.split()
            return [self.tokenizer.word_to_index.get(word, self.config.unk_token_id) for word in words]
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(tokens)
        else:
            # Fallback for simple tokenizers
            words = [self.tokenizer.index_to_word.get(token, '<UNK>') for token in tokens]
            return ' '.join(words)
    
    def _generate_sampling(
        self,
        input_tokens: List[int],
        config: GenerationConfig
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Generate using sampling methods."""
        current_tokens = input_tokens.copy()
        generation_stats = {
            'method': 'sampling',
            'temperature_history': [],
            'token_scores': [],
            'diversity_scores': []
        }
        
        # Initialize temperature callback if using scheduler
        temp_callback = None
        if self.scheduler:
            temp_callback = TemperatureCallback(self.scheduler)
        
        for step in range(config.max_length - len(input_tokens)):
            # Prepare model input
            input_tensor = tf.constant([current_tokens], dtype=tf.int32)
            
            # Get model predictions
            logits = self.model(input_tensor)
            if len(logits.shape) == 3:
                logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Apply style conditioning
            if config.style_type != StyleType.NONE:
                logits = self.style_conditioner.condition_logits(
                    logits, config.style_type, config.style_strength
                )
            
            # Get current temperature
            current_temp = 1.0
            if temp_callback:
                current_temp = temp_callback.get_current_temperature()
                # Update sampler temperature
                if hasattr(self.sampler, 'config'):
                    self.sampler.config.temperature = current_temp
                generation_stats['temperature_history'].append(current_temp)
            
            # Sample next token
            sequence_tensor = tf.constant([current_tokens], dtype=tf.int32)
            next_token, token_log_prob = self.sampler.sample(logits, sequence_tensor, step)
            
            next_token_id = next_token[0].numpy().item()
            token_score = token_log_prob[0].numpy().item()
            
            # Add to sequence
            current_tokens.append(next_token_id)
            generation_stats['token_scores'].append(token_score)
            
            # Update temperature callback
            if temp_callback:
                temp_callback.on_token_generated(next_token_id)
            
            # Calculate diversity
            if config.track_diversity and len(current_tokens) >= 3:
                diversity = calculate_ngram_diversity(current_tokens, n=3)
                generation_stats['diversity_scores'].append(diversity)
            
            # Check for early stopping
            if next_token_id == config.eos_token_id:
                break
                
            # Check minimum length
            if (len(current_tokens) >= config.min_length and 
                config.track_diversity and 
                generation_stats['diversity_scores']):
                
                recent_diversity = np.mean(generation_stats['diversity_scores'][-10:])
                if recent_diversity < config.diversity_threshold:
                    logger.debug(f"Early stopping due to low diversity: {recent_diversity:.3f}")
                    break
        
        return current_tokens, generation_stats
    
    def _generate_beam_search(
        self,
        input_tokens: List[int],
        config: GenerationConfig
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Generate using beam search."""
        if not isinstance(self.sampler, BeamSearchSampler):
            # Create beam search sampler
            beam_sampler = BeamSearchSampler(config.sampler_config)
        else:
            beam_sampler = self.sampler
        
        # Perform beam search
        input_tensor = tf.constant([input_tokens], dtype=tf.int32)
        beams = beam_sampler.beam_search(self.model, input_tensor, config.max_length)
        
        # Return best beam
        best_beam = beams[0] if beams else None
        if best_beam:
            result_tokens = best_beam.tokens
            generation_stats = {
                'method': 'beam_search',
                'beam_width': config.sampler_config.beam_width,
                'best_score': best_beam.score,
                'num_beams': len(beams)
            }
        else:
            result_tokens = input_tokens
            generation_stats = {'method': 'beam_search', 'error': 'No beams generated'}
        
        return result_tokens, generation_stats
    
    def _generate_mixed_mode(
        self,
        input_tokens: List[int],
        config: GenerationConfig
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Generate using mixed sampling modes."""
        current_tokens = input_tokens.copy()
        generation_stats = {
            'method': 'mixed_mode',
            'mode_choices': [],
            'temperature_history': []
        }
        
        # Create samplers for each mode
        samplers = {}
        for mode_name in config.mixed_mode_weights.keys():
            sampler_config = self.config.sampler_config
            samplers[mode_name] = create_sampler(mode_name, sampler_config)
        
        for step in range(config.max_length - len(input_tokens)):
            # Choose sampling method based on weights
            mode_choice = np.random.choice(
                list(config.mixed_mode_weights.keys()),
                p=list(config.mixed_mode_weights.values())
            )
            generation_stats['mode_choices'].append(mode_choice)
            
            # Get model predictions
            input_tensor = tf.constant([current_tokens], dtype=tf.int32)
            logits = self.model(input_tensor)
            if len(logits.shape) == 3:
                logits = logits[:, -1, :]
            
            # Apply style conditioning
            if config.style_type != StyleType.NONE:
                logits = self.style_conditioner.condition_logits(
                    logits, config.style_type, config.style_strength
                )
            
            # Sample with chosen method
            sampler = samplers[mode_choice]
            sequence_tensor = tf.constant([current_tokens], dtype=tf.int32)
            next_token, _ = sampler.sample(logits, sequence_tensor, step)
            
            next_token_id = next_token[0].numpy().item()
            current_tokens.append(next_token_id)
            
            # Check for EOS
            if next_token_id == config.eos_token_id:
                break
        
        return current_tokens, generation_stats
    
    def _calculate_quality_metrics(self, result: GenerationResult):
        """Calculate quality metrics for generated text."""
        if not result.tokens:
            return
        
        # Calculate diversity score
        result.diversity_score = calculate_ngram_diversity(result.tokens, n=3)
        
        # Calculate repetition score (higher = more repetitive)
        if len(result.tokens) >= 6:
            bigram_diversity = calculate_ngram_diversity(result.tokens, n=2)
            trigram_diversity = calculate_ngram_diversity(result.tokens, n=3)
            result.repetition_score = 1.0 - (bigram_diversity + trigram_diversity) / 2.0
        
        # Calculate perplexity if requested
        if self.config.log_probabilities and 'token_scores' in result.sampling_stats:
            scores = result.sampling_stats['token_scores']
            if scores:
                avg_log_prob = np.mean(scores)
                result.perplexity = np.exp(-avg_log_prob)
    
    def generate_batch(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating for prompt {i+1}/{len(prompts)}")
            result = self.generate(prompt, **generation_kwargs)
            results.append(result)
        
        return results
    
    def compare_generation_modes(
        self,
        prompt: str,
        modes: List[GenerationMode],
        num_samples: int = 3
    ) -> Dict[str, List[GenerationResult]]:
        """Compare different generation modes on the same prompt."""
        results = {}
        
        for mode in modes:
            logger.info(f"Testing {mode.value} mode...")
            mode_results = []
            
            for i in range(num_samples):
                # Update config for this mode
                temp_config = self._merge_config({'generation_mode': mode})
                old_config = self.config
                self.config = temp_config
                self.sampler = self._create_sampler()
                
                # Generate
                result = self.generate(prompt)
                mode_results.append(result)
                
                # Restore config
                self.config = old_config
                self.sampler = self._create_sampler()
            
            results[mode.value] = mode_results
        
        return results
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics from generation history."""
        if not self.generation_history:
            return {}
        
        # Aggregate statistics
        stats = {
            'total_generations': len(self.generation_history),
            'avg_generation_time': np.mean([r.generation_time for r in self.generation_history]),
            'avg_tokens_per_second': np.mean([r.tokens_per_second for r in self.generation_history]),
            'avg_diversity_score': np.mean([r.diversity_score for r in self.generation_history]),
            'avg_repetition_score': np.mean([r.repetition_score for r in self.generation_history]),
            'mode_usage': {}
        }
        
        # Count mode usage
        for result in self.generation_history:
            mode = result.mode_used.value
            stats['mode_usage'][mode] = stats['mode_usage'].get(mode, 0) + 1
        
        return stats
    
    def clear_history(self):
        """Clear generation history."""
        self.generation_history.clear()
        logger.info("Generation history cleared")


def create_advanced_generator(
    model: tf.keras.Model,
    tokenizer,
    mode: str = "nucleus",
    **config_kwargs
) -> AdvancedTextGenerator:
    """
    Factory function to create advanced text generator.
    
    Args:
        model: Trained language model
        tokenizer: Tokenizer for encoding/decoding
        mode: Default generation mode
        **config_kwargs: Configuration overrides
        
    Returns:
        Configured AdvancedTextGenerator
    """
    # Create configuration
    config = GenerationConfig(**config_kwargs)
    if mode:
        config.generation_mode = GenerationMode(mode)
    
    # Create generator
    generator = AdvancedTextGenerator(model, tokenizer, config)
    
    return generator