"""
Monitoring Module - Real-time Training Monitoring and Protection
Created by Bernard Orozco

Enterprise module for preventing model degradation during training.
"""

from .anti_saturation_system import AntiSaturationCallback, create_antisaturation_callback

__all__ = [
    'AntiSaturationCallback',
    'create_antisaturation_callback'
]