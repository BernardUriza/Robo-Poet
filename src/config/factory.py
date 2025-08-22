"""
Factory pattern for service configuration and initialization.

Provides centralized configuration and setup for different environments.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .settings import Settings, Environment, LogLevel, get_cached_settings
from src.infrastructure.container import Container, initialize_container, wire_container


logger = logging.getLogger(__name__)


class ConfigurationFactory:
    """Factory for creating environment-specific configurations."""
    
    @staticmethod
    def create_development_config(**overrides) -> Settings:
        """Create development environment configuration."""
        return Settings(
            environment=Environment.DEVELOPMENT,
            debug=True,
            logging={'level': LogLevel.DEBUG, 'console_enabled': True},
            database={'echo': True},
            gpu={'memory_growth': True},
            **overrides
        )
    
    @staticmethod
    def create_testing_config(**overrides) -> Settings:
        """Create testing environment configuration."""
        return Settings(
            environment=Environment.TESTING,
            debug=False,
            database={'url': 'sqlite:///:memory:', 'echo': False},
            logging={'level': LogLevel.WARNING, 'file_enabled': False},
            storage={
                'models_dir': Path('./test_models'),
                'corpus_dir': Path('./test_corpus'),
                'logs_dir': Path('./test_logs'),
                'temp_dir': Path('./test_temp')
            },
            **overrides
        )
    
    @staticmethod
    def create_production_config(**overrides) -> Settings:
        """Create production environment configuration."""
        # Ensure critical production settings
        production_overrides = {
            'environment': Environment.PRODUCTION,
            'debug': False,
            'security': {
                'secret_key': os.getenv('SECRET_KEY', 'MUST_BE_SET_IN_PRODUCTION'),
                'allowed_hosts': os.getenv('ALLOWED_HOSTS', '').split(','),
                'rate_limit_requests': 60
            },
            'logging': {
                'level': LogLevel.INFO,
                'structured_logging': True,
                'file_enabled': True
            },
            'database': {
                'url': os.getenv('DATABASE_URL', 'postgresql://localhost/robo_poet_prod'),
                'echo': False,
                'pool_size': 20,
                'max_overflow': 30
            }
        }
        production_overrides.update(overrides)
        
        return Settings(**production_overrides)
    
    @staticmethod
    def create_from_environment() -> Settings:
        """Create configuration based on ENVIRONMENT variable."""
        env = os.getenv('ENVIRONMENT', 'development').lower()
        
        if env == 'production':
            return ConfigurationFactory.create_production_config()
        elif env == 'testing':
            return ConfigurationFactory.create_testing_config()
        else:
            return ConfigurationFactory.create_development_config()


class LoggingFactory:
    """Factory for setting up application logging."""
    
    @staticmethod
    def setup_logging(settings: Settings) -> None:
        """Setup logging configuration based on settings."""
        log_config = {
            'level': getattr(logging, settings.logging.level.value),
            'format': settings.logging.format,
            'handlers': []
        }
        
        # Console handler
        if settings.logging.console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_config['level'])
            console_formatter = logging.Formatter(log_config['format'])
            console_handler.setFormatter(console_formatter)
            log_config['handlers'].append(console_handler)
        
        # File handler
        if settings.logging.file_enabled:
            # Ensure logs directory exists
            settings.storage.logs_dir.mkdir(parents=True, exist_ok=True)
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                filename=settings.storage.logs_dir / 'robo_poet.log',
                maxBytes=settings.logging.file_max_size_mb * 1024 * 1024,
                backupCount=settings.logging.file_backup_count
            )
            file_handler.setLevel(log_config['level'])
            file_formatter = logging.Formatter(log_config['format'])
            file_handler.setFormatter(file_formatter)
            log_config['handlers'].append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_config['level'],
            format=log_config['format'],
            handlers=log_config['handlers'],
            force=True  # Override existing configuration
        )
        
        # Set specific logger levels
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.WARNING)
        
        logger.info(f"Logging configured for {settings.environment.value} environment")


class DatabaseFactory:
    """Factory for database configuration and initialization."""
    
    @staticmethod
    def create_engine_config(settings: Settings) -> Dict[str, Any]:
        """Create SQLAlchemy engine configuration."""
        config = {
            'url': settings.database.url,
            'echo': settings.database.echo,
            'pool_size': settings.database.pool_size,
            'max_overflow': settings.database.max_overflow,
            'pool_timeout': settings.database.pool_timeout
        }
        
        # Environment-specific optimizations
        if settings.is_production():
            config.update({
                'pool_pre_ping': True,
                'pool_recycle': 3600,  # Recycle connections every hour
                'connect_args': {'connect_timeout': 10}
            })
        
        return config
    
    @staticmethod
    def validate_connection(settings: Settings) -> bool:
        """Validate database connection."""
        try:
            from sqlalchemy import create_engine, text
            
            engine_config = DatabaseFactory.create_engine_config(settings)
            engine = create_engine(**engine_config)
            
            with engine.connect() as conn:
                conn.execute(text('SELECT 1'))
            
            logger.info("Database connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False


class GPUFactory:
    """Factory for GPU configuration and initialization."""
    
    @staticmethod
    def configure_gpu(settings: Settings) -> bool:
        """Configure GPU settings for TensorFlow."""
        try:
            import tensorflow as tf
            
            # Set visible devices
            os.environ['CUDA_VISIBLE_DEVICES'] = settings.gpu.visible_devices
            
            # Configure GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, settings.gpu.memory_growth)
                
                # Set memory limit if specified
                if settings.gpu.max_memory_mb:
                    tf.config.experimental.set_memory_limit(
                        gpus[0], 
                        settings.gpu.max_memory_mb
                    )
                
                # Configure mixed precision
                if settings.gpu.mixed_precision:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                
                logger.info(f"GPU configured: {len(gpus)} device(s) available")
                return True
            else:
                logger.warning("No GPU devices found")
                return False
                
        except ImportError:
            logger.warning("TensorFlow not available for GPU configuration")
            return False
        except Exception as e:
            logger.error(f"GPU configuration failed: {e}")
            return False


class ApplicationFactory:
    """Main factory for application initialization."""
    
    @staticmethod
    def create_application(
        environment: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Container:
        """Create and configure the complete application."""
        
        # Create configuration
        if environment:
            os.environ['ENVIRONMENT'] = environment
        
        if environment == 'testing':
            settings = ConfigurationFactory.create_testing_config(**(config_overrides or {}))
        elif environment == 'production':
            settings = ConfigurationFactory.create_production_config(**(config_overrides or {}))
        else:
            settings = ConfigurationFactory.create_development_config(**(config_overrides or {}))
        
        # Setup logging
        LoggingFactory.setup_logging(settings)
        
        # Configure GPU
        GPUFactory.configure_gpu(settings)
        
        # Validate database
        if not DatabaseFactory.validate_connection(settings):
            raise RuntimeError("Database connection failed")
        
        # Initialize dependency container
        container = initialize_container(settings)
        
        # Wire dependencies
        wire_container(container)
        
        logger.info(f"Application initialized for {settings.environment.value} environment")
        return container
    
    @staticmethod
    def create_for_testing(**overrides) -> Container:
        """Create application configured for testing."""
        return ApplicationFactory.create_application('testing', overrides)
    
    @staticmethod
    def create_for_development(**overrides) -> Container:
        """Create application configured for development."""
        return ApplicationFactory.create_application('development', overrides)
    
    @staticmethod
    def create_for_production(**overrides) -> Container:
        """Create application configured for production."""
        return ApplicationFactory.create_application('production', overrides)


# Convenience functions
def setup_development_environment() -> Container:
    """Setup complete development environment."""
    return ApplicationFactory.create_for_development()


def setup_testing_environment() -> Container:
    """Setup complete testing environment.""" 
    return ApplicationFactory.create_for_testing()


def setup_production_environment() -> Container:
    """Setup complete production environment."""
    return ApplicationFactory.create_for_production()


# Environment detection
def get_current_environment() -> str:
    """Get current environment from various sources."""
    # Check environment variable
    env = os.getenv('ENVIRONMENT')
    if env:
        return env.lower()
    
    # Check for common CI/CD indicators
    if any(key in os.environ for key in ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI']):
        return 'testing'
    
    # Check for production indicators
    if any(key in os.environ for key in ['KUBERNETES_SERVICE_HOST', 'HEROKU_APP_NAME']):
        return 'production'
    
    # Default to development
    return 'development'


def auto_configure() -> Container:
    """Automatically configure application based on environment detection."""
    env = get_current_environment()
    logger.info(f"Auto-detected environment: {env}")
    return ApplicationFactory.create_application(env)