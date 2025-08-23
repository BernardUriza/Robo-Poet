"""
Temperature scheduling for dynamic creativity control during generation.

Implements various scheduling strategies for temperature adjustment
during text generation for optimal creativity/coherence balance.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Configuration for temperature scheduling."""
    
    # Base temperature parameters
    initial_temperature: float = 1.0
    final_temperature: float = 0.8
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    
    # Scheduling parameters
    total_steps: int = 100
    warmup_steps: int = 10
    cooldown_steps: int = 20
    
    # Adaptive parameters
    target_diversity: float = 0.7  # Target n-gram diversity
    adaptation_rate: float = 0.1
    diversity_window: int = 50  # Tokens to consider for diversity
    
    # Cyclical parameters  
    cycle_length: int = 50
    cycle_amplitude: float = 0.3
    
    def validate(self):
        """Validate configuration parameters."""
        assert 0.0 < self.initial_temperature <= 5.0
        assert 0.0 < self.final_temperature <= 5.0
        assert 0.0 < self.min_temperature <= self.max_temperature
        assert self.total_steps > 0
        assert 0 <= self.warmup_steps < self.total_steps
        assert 0 <= self.cooldown_steps < self.total_steps
        assert 0.0 < self.target_diversity <= 1.0
        assert 0.0 < self.adaptation_rate <= 1.0


class TemperatureScheduler(ABC):
    """Base class for temperature scheduling strategies."""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.config.validate()
        self.step = 0
        self.temperature_history = []
        self.diversity_history = []
        
    @abstractmethod
    def get_temperature(self, step: Optional[int] = None) -> float:
        """
        Get temperature for current step.
        
        Args:
            step: Optional step override
            
        Returns:
            Current temperature value
        """
        pass
    
    def update_step(self, diversity: Optional[float] = None):
        """Update internal step counter and history."""
        current_temp = self.get_temperature(self.step)
        self.temperature_history.append(current_temp)
        
        if diversity is not None:
            self.diversity_history.append(diversity)
            
        self.step += 1
        
    def reset(self):
        """Reset scheduler to initial state."""
        self.step = 0
        self.temperature_history.clear()
        self.diversity_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        if not self.temperature_history:
            return {}
            
        return {
            'current_step': self.step,
            'current_temperature': self.temperature_history[-1] if self.temperature_history else self.config.initial_temperature,
            'min_temperature': min(self.temperature_history) if self.temperature_history else 0,
            'max_temperature': max(self.temperature_history) if self.temperature_history else 0,
            'avg_temperature': np.mean(self.temperature_history) if self.temperature_history else 0,
            'avg_diversity': np.mean(self.diversity_history) if self.diversity_history else 0,
            'total_steps': len(self.temperature_history)
        }
    
    def _clamp_temperature(self, temp: float) -> float:
        """Clamp temperature to valid range."""
        return np.clip(temp, self.config.min_temperature, self.config.max_temperature)


class LinearScheduler(TemperatureScheduler):
    """Linear temperature scheduling from initial to final temperature."""
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        """Linear interpolation between initial and final temperature."""
        current_step = step if step is not None else self.step
        
        if current_step >= self.config.total_steps:
            return self._clamp_temperature(self.config.final_temperature)
        
        # Linear interpolation
        progress = current_step / self.config.total_steps
        temp = (
            self.config.initial_temperature + 
            progress * (self.config.final_temperature - self.config.initial_temperature)
        )
        
        return self._clamp_temperature(temp)


class ExponentialScheduler(TemperatureScheduler):
    """Exponential temperature decay."""
    
    def __init__(self, config: SchedulerConfig, decay_rate: float = 0.95):
        super().__init__(config)
        self.decay_rate = decay_rate
        
    def get_temperature(self, step: Optional[int] = None) -> float:
        """Exponential decay from initial temperature."""
        current_step = step if step is not None else self.step
        
        temp = self.config.initial_temperature * (self.decay_rate ** current_step)
        
        # Don't decay below final temperature
        temp = max(temp, self.config.final_temperature)
        
        return self._clamp_temperature(temp)


class CosineScheduler(TemperatureScheduler):
    """Cosine annealing temperature schedule."""
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        """Cosine annealing schedule."""
        current_step = step if step is not None else self.step
        
        if current_step >= self.config.total_steps:
            return self._clamp_temperature(self.config.final_temperature)
        
        # Cosine annealing formula
        progress = current_step / self.config.total_steps
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        temp = (
            self.config.final_temperature + 
            (self.config.initial_temperature - self.config.final_temperature) * cosine_factor
        )
        
        return self._clamp_temperature(temp)


class WarmupCooldownScheduler(TemperatureScheduler):
    """Temperature scheduler with warmup and cooldown phases."""
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        """Three-phase scheduling: warmup, stable, cooldown."""
        current_step = step if step is not None else self.step
        
        # Warmup phase - increase from min to initial
        if current_step < self.config.warmup_steps:
            progress = current_step / self.config.warmup_steps
            temp = (
                self.config.min_temperature + 
                progress * (self.config.initial_temperature - self.config.min_temperature)
            )
            return self._clamp_temperature(temp)
        
        # Cooldown phase - decrease from initial to final
        cooldown_start = self.config.total_steps - self.config.cooldown_steps
        if current_step >= cooldown_start:
            cooldown_progress = (current_step - cooldown_start) / self.config.cooldown_steps
            temp = (
                self.config.initial_temperature + 
                cooldown_progress * (self.config.final_temperature - self.config.initial_temperature)
            )
            return self._clamp_temperature(temp)
        
        # Stable phase - maintain initial temperature
        return self._clamp_temperature(self.config.initial_temperature)


class CyclicalScheduler(TemperatureScheduler):
    """Cyclical temperature scheduling for varied generation."""
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        """Cyclical temperature with base trend."""
        current_step = step if step is not None else self.step
        
        # Base temperature (linear decay)
        if self.config.total_steps > 0:
            progress = min(current_step / self.config.total_steps, 1.0)
        else:
            progress = 0
            
        base_temp = (
            self.config.initial_temperature + 
            progress * (self.config.final_temperature - self.config.initial_temperature)
        )
        
        # Cyclical component
        cycle_progress = (current_step % self.config.cycle_length) / self.config.cycle_length
        cycle_factor = math.sin(2 * math.pi * cycle_progress)
        
        temp = base_temp + self.config.cycle_amplitude * cycle_factor
        
        return self._clamp_temperature(temp)


class AdaptiveScheduler(TemperatureScheduler):
    """Adaptive temperature scheduling based on generation diversity."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.current_temperature = config.initial_temperature
        
    def get_temperature(self, step: Optional[int] = None) -> float:
        """Adaptive temperature based on recent diversity."""
        return self._clamp_temperature(self.current_temperature)
    
    def update_step(self, diversity: Optional[float] = None):
        """Update temperature based on diversity feedback."""
        super().update_step(diversity)
        
        if diversity is not None and len(self.diversity_history) > 0:
            # Calculate recent average diversity
            window_size = min(self.config.diversity_window, len(self.diversity_history))
            recent_diversity = np.mean(self.diversity_history[-window_size:])
            
            # Adjust temperature based on diversity
            if recent_diversity < self.config.target_diversity:
                # Too repetitive - increase temperature
                adjustment = self.config.adaptation_rate * (self.config.target_diversity - recent_diversity)
                self.current_temperature += adjustment
            else:
                # Sufficiently diverse or too random - slightly decrease temperature  
                adjustment = self.config.adaptation_rate * (recent_diversity - self.config.target_diversity)
                self.current_temperature -= adjustment * 0.5
            
            # Keep within bounds
            self.current_temperature = self._clamp_temperature(self.current_temperature)


class MultiStageScheduler(TemperatureScheduler):
    """Multi-stage temperature scheduling with different phases."""
    
    def __init__(self, config: SchedulerConfig, stages: List[Dict[str, Any]]):
        """
        Initialize multi-stage scheduler.
        
        Args:
            config: Base configuration
            stages: List of stage configurations with 'duration', 'start_temp', 'end_temp', 'method'
        """
        super().__init__(config)
        self.stages = stages
        self.total_duration = sum(stage['duration'] for stage in stages)
        
    def get_temperature(self, step: Optional[int] = None) -> float:
        """Multi-stage temperature scheduling."""
        current_step = step if step is not None else self.step
        
        # Find current stage
        elapsed = 0
        for stage in self.stages:
            if current_step < elapsed + stage['duration']:
                # Within this stage
                stage_progress = (current_step - elapsed) / stage['duration']
                
                start_temp = stage['start_temp']
                end_temp = stage['end_temp']
                method = stage.get('method', 'linear')
                
                if method == 'linear':
                    temp = start_temp + stage_progress * (end_temp - start_temp)
                elif method == 'exponential':
                    decay_rate = (end_temp / start_temp) ** (1 / stage['duration'])
                    temp = start_temp * (decay_rate ** (current_step - elapsed))
                elif method == 'cosine':
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * stage_progress))
                    temp = end_temp + (start_temp - end_temp) * cosine_factor
                else:
                    temp = start_temp  # Constant
                
                return self._clamp_temperature(temp)
            
            elapsed += stage['duration']
        
        # Past all stages - return final temperature
        return self._clamp_temperature(self.stages[-1]['end_temp'])


def create_scheduler(
    method: str, 
    config: SchedulerConfig, 
    **kwargs
) -> TemperatureScheduler:
    """
    Factory function to create temperature schedulers.
    
    Args:
        method: Scheduler type ('linear', 'exponential', 'cosine', 'warmup', 'cyclical', 'adaptive', 'multistage')
        config: Scheduler configuration
        **kwargs: Additional parameters specific to scheduler type
        
    Returns:
        Configured scheduler instance
    """
    schedulers = {
        'linear': LinearScheduler,
        'exponential': lambda cfg: ExponentialScheduler(cfg, kwargs.get('decay_rate', 0.95)),
        'cosine': CosineScheduler,
        'warmup': WarmupCooldownScheduler,
        'warmup_cooldown': WarmupCooldownScheduler,
        'cyclical': CyclicalScheduler,
        'adaptive': AdaptiveScheduler,
        'multistage': lambda cfg: MultiStageScheduler(cfg, kwargs.get('stages', []))
    }
    
    if method not in schedulers:
        raise ValueError(f"Unknown scheduler method: {method}. Available: {list(schedulers.keys())}")
    
    return schedulers[method](config)


def calculate_ngram_diversity(
    tokens: List[int], 
    n: int = 3,
    window_size: Optional[int] = None
) -> float:
    """
    Calculate n-gram diversity for a sequence of tokens.
    
    Args:
        tokens: List of token IDs
        n: N-gram size
        window_size: Only consider last window_size tokens
        
    Returns:
        Diversity score (ratio of unique n-grams to total n-grams)
    """
    if window_size is not None:
        tokens = tokens[-window_size:]
    
    if len(tokens) < n:
        return 1.0  # Perfect diversity for short sequences
    
    # Extract n-grams
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    
    if not ngrams:
        return 1.0
    
    # Calculate diversity
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    
    return unique_ngrams / total_ngrams


class TemperatureCallback:
    """Callback for monitoring temperature during generation."""
    
    def __init__(self, scheduler: TemperatureScheduler, diversity_n: int = 3):
        self.scheduler = scheduler
        self.diversity_n = diversity_n
        self.generated_tokens = []
        
    def on_token_generated(self, token_id: int):
        """Called when a token is generated."""
        self.generated_tokens.append(token_id)
        
        # Calculate recent diversity
        if len(self.generated_tokens) >= self.diversity_n:
            diversity = calculate_ngram_diversity(
                self.generated_tokens, 
                n=self.diversity_n,
                window_size=self.scheduler.config.diversity_window
            )
        else:
            diversity = 1.0
        
        # Update scheduler
        self.scheduler.update_step(diversity)
        
        # Log periodically
        if len(self.generated_tokens) % 20 == 0:
            stats = self.scheduler.get_stats()
            logger.debug(
                f"Step {stats['current_step']}: "
                f"T={stats['current_temperature']:.3f}, "
                f"Diversity={stats.get('avg_diversity', 0):.3f}"
            )
    
    def get_current_temperature(self) -> float:
        """Get current temperature value."""
        return self.scheduler.get_temperature()
    
    def reset(self):
        """Reset callback state."""
        self.generated_tokens.clear()
        self.scheduler.reset()


def demo_schedulers():
    """Demonstrate different temperature schedulers."""
    config = SchedulerConfig(
        initial_temperature=1.5,
        final_temperature=0.7,
        total_steps=100,
        warmup_steps=10,
        cooldown_steps=20
    )
    
    schedulers = {
        'Linear': create_scheduler('linear', config),
        'Exponential': create_scheduler('exponential', config, decay_rate=0.98),
        'Cosine': create_scheduler('cosine', config),
        'Warmup-Cooldown': create_scheduler('warmup', config),
        'Cyclical': create_scheduler('cyclical', config),
        'Adaptive': create_scheduler('adaptive', config)
    }
    
    print("Temperature Scheduler Comparison:")
    print("=" * 50)
    
    steps = [0, 25, 50, 75, 100]
    
    for step in steps:
        print(f"\nStep {step}:")
        for name, scheduler in schedulers.items():
            temp = scheduler.get_temperature(step)
            print(f"  {name:<15}: {temp:.3f}")
    
    return schedulers