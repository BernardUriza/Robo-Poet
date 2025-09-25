"""
Intelligence module for Robo-Poet Framework.

Integrates external AI services for enhanced training and dataset optimization.
"""

from .claude_integration import (
    ClaudeIntegration,
    IntelligentDatasetManager,
    TrainingMetrics,
    DatasetSuggestion,
    test_claude_integration
)

__all__ = [
    'ClaudeIntegration',
    'IntelligentDatasetManager',
    'TrainingMetrics',
    'DatasetSuggestion',
    'test_claude_integration'
]