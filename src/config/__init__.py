"""
Configuration module for Robo-Poet application.

Provides environment-based configuration with validation and type safety.
"""

from .settings import (
    Settings, 
    DatabaseSettings, 
    GPUSettings, 
    TrainingSettings,
    StorageSettings,
    SecuritySettings,
    LoggingSettings,
    Environment,
    LogLevel,
    get_settings,
    get_cached_settings,
    reload_settings
)

# Compatibility layer for legacy imports
def get_config():
    """Compatibility function for legacy get_config imports."""
    return get_settings()

__all__ = [
    'Settings',
    'DatabaseSettings', 
    'GPUSettings', 
    'TrainingSettings',
    'StorageSettings',
    'SecuritySettings',
    'LoggingSettings',
    'Environment',
    'LogLevel',
    'get_settings',
    'get_cached_settings', 
    'reload_settings',
    'get_config'  # Compatibility function
]