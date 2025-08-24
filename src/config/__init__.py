"""
Configuration module for Robo-Poet application.

Unified configuration system with backward compatibility.
Migrates from multiple config systems to single source of truth.
"""

import warnings
from typing import Optional

# Primary unified configuration system
from src.core.unified_config import (
    UnifiedConfig,
    SystemConfig,
    GPUConfig, 
    ModelConfig as UnifiedModelConfig,
    DataConfig,
    GenerationConfig,
    EvaluationConfig,
    create_strategy_config,
    get_config as get_unified_config,
    LogLevel,
    GPUBackend
)

# Legacy compatibility
from .settings import (
    Settings, 
    DatabaseSettings, 
    GPUSettings,
    ModelSettings,
    TrainingSettings,
    StorageSettings,
    SecuritySettings,
    LoggingSettings,
    Environment,
    get_settings,
    get_cached_settings,
    reload_settings
)

# Adapters for migration
from .adapters import ConfigurationAdapter, get_legacy_settings


def get_config(strategy: str = "academic") -> UnifiedConfig:
    """
    Get unified configuration with backward compatibility.
    
    Args:
        strategy: Configuration strategy ('academic', 'production', 'development')
        
    Returns:
        UnifiedConfig: Unified configuration instance
    """
    return create_strategy_config(strategy)


def get_model_config(strategy: str = "academic"):
    """
    Legacy compatibility for domain model configuration.
    
    Args:
        strategy: Configuration strategy
        
    Returns:
        Domain ModelConfig value object
    """
    warnings.warn(
        "get_model_config() is deprecated. Use get_config() and ConfigurationAdapter instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    unified = get_config(strategy)
    return ConfigurationAdapter.unified_to_domain_model_config(unified)


def migrate_to_unified(legacy_settings: Optional[Settings] = None) -> UnifiedConfig:
    """
    Migrate from legacy Pydantic settings to unified configuration.
    
    Args:
        legacy_settings: Optional legacy settings to migrate
        
    Returns:
        UnifiedConfig: Migrated configuration
    """
    if legacy_settings is None:
        legacy_settings = get_cached_settings()
    
    # Create production config as base
    unified = create_strategy_config("production")
    
    # Apply legacy overrides
    overrides = {
        "model": {
            "vocab_size": legacy_settings.model.vocab_size,
            "sequence_length": legacy_settings.model.sequence_length,
            "lstm_units": [legacy_settings.model.lstm_units, legacy_settings.model.lstm_units],
            "dropout_rate": legacy_settings.model.dropout_rate,
            "batch_size": legacy_settings.model.batch_size,
            "epochs": legacy_settings.model.epochs,
            "embedding_dim": legacy_settings.model.embedding_dim
        },
        "gpu": {
            "memory_growth": legacy_settings.gpu.memory_growth,
            "mixed_precision": legacy_settings.gpu.mixed_precision,
            "device_id": 0  # Default
        }
    }
    
    return ConfigurationAdapter.merge_configurations(unified, overrides)


__all__ = [
    # Primary unified system
    'UnifiedConfig',
    'SystemConfig',
    'GPUConfig', 
    'UnifiedModelConfig',
    'DataConfig',
    'GenerationConfig',
    'EvaluationConfig',
    'create_strategy_config',
    'get_unified_config',
    'LogLevel',
    'GPUBackend',
    
    # Main entry points
    'get_config',
    'get_model_config',
    'migrate_to_unified',
    
    # Adapter utilities
    'ConfigurationAdapter',
    'get_legacy_settings',
    
    # Legacy Pydantic system (deprecated)
    'Settings',
    'DatabaseSettings', 
    'GPUSettings',
    'ModelSettings',
    'TrainingSettings',
    'StorageSettings',
    'SecuritySettings',
    'LoggingSettings',
    'Environment',
    'get_settings',
    'get_cached_settings', 
    'reload_settings',
]