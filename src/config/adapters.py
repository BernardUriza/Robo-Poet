"""
Configuration Adapters

Provides compatibility and conversion between different configuration systems
during the migration to unified configuration architecture.
"""

from typing import Dict, Any
import warnings

from src.core.unified_config import UnifiedConfig, create_strategy_config
from src.domain.value_objects.model_config import ModelConfig as DomainModelConfig
from src.config.settings import Settings, ModelSettings


class ConfigurationAdapter:
    """
    Adapter class for converting between different configuration formats.
    
    Provides backward compatibility during configuration consolidation.
    """
    
    @staticmethod
    def unified_to_domain_model_config(unified: UnifiedConfig) -> DomainModelConfig:
        """
        Convert UnifiedConfig to domain ModelConfig value object.
        
        Args:
            unified: UnifiedConfig instance
            
        Returns:
            DomainModelConfig: Immutable domain configuration
        """
        return DomainModelConfig(
            vocab_size=unified.model.vocab_size,
            sequence_length=unified.model.sequence_length,
            lstm_units=unified.model.lstm_units[0] if isinstance(unified.model.lstm_units, list) else unified.model.lstm_units,
            lstm_layers=len(unified.model.lstm_units) if isinstance(unified.model.lstm_units, list) else 2,
            embedding_dim=unified.model.embedding_dim,
            variational_dropout_rate=unified.model.dropout_rate,
            dropconnect_rate=unified.model.recurrent_dropout_rate,
            use_weight_tying=unified.model.use_layer_norm,  # Approximate mapping
            batch_size=unified.model.batch_size,
            epochs=unified.model.epochs,
            learning_rate=unified.model.learning_rate,
            validation_split=unified.data.validation_split,
            early_stopping_patience=unified.evaluation.early_stopping_patience,
            tokenization=unified.data.tokenization_strategy.replace('_based', ''),  # 'word_based' -> 'word'
            mixed_precision=unified.gpu.mixed_precision,
            memory_growth=unified.gpu.memory_growth
        )
    
    @staticmethod
    def unified_to_pydantic_settings(unified: UnifiedConfig) -> Settings:
        """
        Convert UnifiedConfig to Pydantic Settings.
        
        Args:
            unified: UnifiedConfig instance
            
        Returns:
            Settings: Pydantic settings instance
        """
        # Create model settings
        model_settings = ModelSettings(
            sequence_length=unified.model.sequence_length,
            lstm_units=unified.model.lstm_units[0] if isinstance(unified.model.lstm_units, list) else unified.model.lstm_units,
            dropout_rate=unified.model.dropout_rate,
            vocab_size=unified.model.vocab_size,
            embedding_dim=unified.model.embedding_dim,
            batch_size=unified.model.batch_size,
            epochs=unified.model.epochs,
            model_type=unified.model.model_type
        )
        
        return Settings(model=model_settings)
    
    @staticmethod
    def domain_to_unified_model_config(domain_config: DomainModelConfig) -> Dict[str, Any]:
        """
        Convert domain ModelConfig to unified model configuration dict.
        
        Args:
            domain_config: Domain ModelConfig value object
            
        Returns:
            Dict suitable for UnifiedConfig.model updates
        """
        return {
            'vocab_size': domain_config.vocab_size,
            'sequence_length': domain_config.sequence_length,
            'lstm_units': [domain_config.lstm_units] * domain_config.lstm_layers,
            'embedding_dim': domain_config.embedding_dim,
            'dropout_rate': domain_config.variational_dropout_rate,
            'recurrent_dropout_rate': domain_config.dropconnect_rate,
            'batch_size': domain_config.batch_size,
            'epochs': domain_config.epochs,
            'learning_rate': domain_config.learning_rate,
            'weight_decay': 0.01,  # Default from domain config
            'gradient_clip_norm': 1.0,  # Default from domain config
            'use_layer_norm': domain_config.use_weight_tying
        }
    
    @staticmethod
    def merge_configurations(primary: UnifiedConfig, overrides: Dict[str, Any]) -> UnifiedConfig:
        """
        Merge configuration overrides into primary configuration.
        
        Args:
            primary: Primary configuration
            overrides: Dictionary of override values
            
        Returns:
            New UnifiedConfig with overrides applied
        """
        # Create a copy of the primary config
        merged_dict = primary.to_dict()
        
        # Apply overrides recursively
        def deep_update(base: Dict, updates: Dict) -> Dict:
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
            return base
        
        deep_update(merged_dict, overrides)
        
        # Create new UnifiedConfig from merged dictionary
        return UnifiedConfig.from_dict(merged_dict)


# Legacy compatibility functions
def get_config(strategy: str = "academic") -> UnifiedConfig:
    """
    Legacy compatibility function for config.get_config().
    
    Args:
        strategy: Configuration strategy to use
        
    Returns:
        UnifiedConfig configured for specified strategy
    """
    warnings.warn(
        "get_config() is deprecated. Use create_strategy_config() from core.unified_config instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return create_strategy_config(strategy)


def get_model_config(strategy: str = "academic") -> DomainModelConfig:
    """
    Legacy compatibility function for getting domain model configuration.
    
    Args:
        strategy: Configuration strategy to use
        
    Returns:
        DomainModelConfig value object
    """
    warnings.warn(
        "get_model_config() is deprecated. Use ConfigurationAdapter.unified_to_domain_model_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    unified = create_strategy_config(strategy)
    return ConfigurationAdapter.unified_to_domain_model_config(unified)


def get_legacy_settings() -> Settings:
    """
    Get Pydantic Settings from unified configuration.
    
    Returns:
        Settings: Pydantic settings instance
    """
    warnings.warn(
        "get_legacy_settings() is deprecated. Use unified_config directly.",
        DeprecationWarning,
        stacklevel=2
    )
    
    unified = create_strategy_config("production")
    return ConfigurationAdapter.unified_to_pydantic_settings(unified)