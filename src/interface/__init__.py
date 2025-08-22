"""
Academic Interface Package for Robo-Poet Framework

Modular interface system with specialized components for academic text generation.
"""

from .menu_system import AcademicMenuSystem
from .phase1_training import Phase1TrainingInterface
from .phase2_generation import Phase2GenerationInterface
from .generation_modes import GenerationModes

__all__ = [
    'AcademicMenuSystem',
    'Phase1TrainingInterface', 
    'Phase2GenerationInterface',
    'GenerationModes'
]