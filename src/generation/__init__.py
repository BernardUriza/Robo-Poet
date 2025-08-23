"""
Generation module for advanced text generation methods.

Strategy 5: Generación Avanzada Multi-Modo
- Top-k sampling
- Nucleus (Top-p) sampling  
- Beam search
- Temperature scheduling
- Conditional generation by style
"""

from .samplers import (
    GreedySampler,
    TopKSampler,
    NucleusSampler,
    TemperatureSampler,
    BeamSearchSampler,
    SamplerConfig
)

from .advanced_generator import (
    AdvancedTextGenerator,
    GenerationConfig,
    GenerationResult,
    StyleConditioner
)

from .temperature_scheduler import (
    TemperatureScheduler,
    LinearScheduler,
    ExponentialScheduler,
    CosineScheduler
)

__all__ = [
    'GreedySampler',
    'TopKSampler', 
    'NucleusSampler',
    'TemperatureSampler',
    'BeamSearchSampler',
    'SamplerConfig',
    'AdvancedTextGenerator',
    'GenerationConfig',
    'GenerationResult',
    'StyleConditioner',
    'TemperatureScheduler',
    'LinearScheduler',
    'ExponentialScheduler',
    'CosineScheduler'
]