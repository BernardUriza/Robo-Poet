"""
Application settings using Pydantic Settings.

Environment-based configuration with validation and type safety.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="sqlite:///./robo_poet.db",
        description="Database connection URL"
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL statements for debugging"
    )
    pool_size: int = Field(
        default=5,
        description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=10,
        description="Maximum number of connections to overflow the pool"
    )
    pool_timeout: int = Field(
        default=30,
        description="Number of seconds to wait before giving up on connection"
    )
    
    model_config = {"env_prefix": "DB_"}


class GPUSettings(BaseSettings):
    """GPU and training configuration settings."""
    
    visible_devices: str = Field(
        default="0",
        description="CUDA visible devices"
    )
    memory_growth: bool = Field(
        default=True,
        description="Enable GPU memory growth"
    )
    mixed_precision: bool = Field(
        default=True,
        description="Enable mixed precision training"
    )
    max_memory_mb: Optional[int] = Field(
        default=None,
        description="Maximum GPU memory to allocate in MB"
    )
    
    model_config = {"env_prefix": "GPU_"}


class ModelSettings(BaseSettings):
    """Model architecture configuration settings."""
    
    sequence_length: int = Field(
        default=100,
        description="Sequence length for training"
    )
    lstm_units: int = Field(
        default=256,
        description="Number of LSTM units"
    )
    dropout_rate: float = Field(
        default=0.3,
        description="Dropout rate for regularization"
    )
    vocab_size: int = Field(
        default=10000,
        description="Vocabulary size"
    )
    embedding_dim: int = Field(
        default=128,
        description="Embedding dimension"
    )
    batch_size: int = Field(
        default=64,
        description="Training batch size"
    )
    epochs: int = Field(
        default=100,
        description="Number of training epochs"
    )
    model_type: str = Field(
        default="lstm",
        description="Model architecture type"
    )
    
    @field_validator('dropout_rate')
    @classmethod
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('dropout_rate must be between 0 and 1')
        return v
    
    model_config = {"env_prefix": "MODEL_"}


class TrainingSettings(BaseSettings):
    """Training-specific configuration settings."""
    
    default_epochs: int = Field(
        default=100,
        description="Default number of training epochs"
    )
    default_batch_size: int = Field(
        default=64,
        description="Default batch size"
    )
    default_learning_rate: float = Field(
        default=0.001,
        description="Default learning rate"
    )
    checkpoint_interval: int = Field(
        default=10,
        description="Save checkpoint every N epochs"
    )
    early_stopping_patience: int = Field(
        default=15,
        description="Early stopping patience"
    )
    validation_split: float = Field(
        default=0.2,
        description="Validation split ratio"
    )
    
    @field_validator('validation_split')
    @classmethod
    def validate_split(cls, v):
        if not 0 < v < 1:
            raise ValueError('validation_split must be between 0 and 1')
        return v
    
    model_config = {"env_prefix": "TRAINING_"}


class StorageSettings(BaseSettings):
    """Storage and file system configuration."""
    
    models_dir: Path = Field(
        default=Path("./models"),
        description="Directory to store trained models"
    )
    corpus_dir: Path = Field(
        default=Path("./corpus"),
        description="Directory to store text corpuses"
    )
    logs_dir: Path = Field(
        default=Path("./logs"),
        description="Directory to store application logs"
    )
    temp_dir: Path = Field(
        default=Path("./temp"),
        description="Directory for temporary files"
    )
    max_corpus_size_mb: int = Field(
        default=500,
        description="Maximum corpus size in MB"
    )
    
    @field_validator('models_dir', 'corpus_dir', 'logs_dir', 'temp_dir', mode='before')
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    model_config = {"env_prefix": "STORAGE_"}


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""
    
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for encryption and signing"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="List of allowed hosts"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="List of allowed CORS origins"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit: requests per minute"
    )
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        """Validate secret key in production."""
        # Note: In Pydantic v2, we can't access other fields in field_validator
        # This validation will be done at the model level if needed
        if v == "dev-secret-key-change-in-production":
            import os
            if os.getenv('ENVIRONMENT', 'development') == 'production':
                raise ValueError("Must set a secure secret key in production")
        return v
    
    model_config = {"env_prefix": "SECURITY_"}


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file_enabled: bool = Field(
        default=True,
        description="Enable file logging"
    )
    file_max_size_mb: int = Field(
        default=10,
        description="Maximum log file size in MB"
    )
    file_backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    console_enabled: bool = Field(
        default=True,
        description="Enable console logging"
    )
    structured_logging: bool = Field(
        default=False,
        description="Enable structured JSON logging"
    )
    
    model_config = {"env_prefix": "LOG_"}


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application metadata
    app_name: str = Field(
        default="Robo-Poet",
        description="Application name"
    )
    app_version: str = Field(
        default="2.1.0",
        description="Application version"
    )
    app_description: str = Field(
        default="Enterprise-grade text generation with LSTM neural networks",
        description="Application description"
    )
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    gpu: GPUSettings = GPUSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    storage: StorageSettings = StorageSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()
    
    # API settings
    api_host: str = Field(
        default="localhost",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        description="API server port"
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="API path prefix"
    )
    
    # Worker settings
    worker_concurrency: int = Field(
        default=4,
        description="Number of worker processes"
    )
    background_tasks_enabled: bool = Field(
        default=True,
        description="Enable background task processing"
    )
    
    @field_validator('debug')
    @classmethod
    def debug_in_production(cls, v):
        """Warn about debug mode in production."""
        # Note: In Pydantic v2, we can't access other fields in field_validator
        # This validation will be done at the model level if needed
        if v:
            import os
            if os.getenv('ENVIRONMENT', 'development') == 'production':
                raise ValueError("Debug mode should not be enabled in production")
        return v
    
    def get_database_url(self) -> str:
        """Get the complete database URL."""
        return self.database.url
    
    def get_model_path(self, model_id: str) -> Path:
        """Get the path for a specific model."""
        return self.storage.models_dir / f"{model_id}.h5"
    
    def get_corpus_path(self, corpus_id: str) -> Path:
        """Get the path for a specific corpus."""
        return self.storage.corpus_dir / f"{corpus_id}.txt"
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (excluding sensitive data)."""
        data = self.dict()
        
        # Remove sensitive information
        if 'secret_key' in data.get('security', {}):
            data['security']['secret_key'] = "***hidden***"
        
        return data
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "json_schema_extra": {
            "example": {
                "app_name": "Robo-Poet",
                "environment": "development",
                "database": {
                    "url": "postgresql://user:pass@localhost/robo_poet"
                },
                "gpu": {
                    "visible_devices": "0",
                    "memory_growth": True
                }
            }
        }
    }


# Global settings instance
def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Cache settings for performance
_settings_cache: Optional[Settings] = None


def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = get_settings()
    return _settings_cache


def reload_settings() -> Settings:
    """Reload settings and clear cache."""
    global _settings_cache
    _settings_cache = None
    return get_cached_settings()