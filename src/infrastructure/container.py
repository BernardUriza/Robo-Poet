"""
Dependency injection container using dependency-injector.

Manages application dependencies and their lifecycle.
"""

import logging
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import Settings, get_cached_settings
from src.infrastructure.persistence.sqlalchemy_models import Base
from src.infrastructure.persistence.unit_of_work import SQLAlchemyUnitOfWork
from src.infrastructure.persistence.repositories.corpus_repository import SQLAlchemyCorpusRepository
from src.infrastructure.persistence.repositories.model_repository import SQLAlchemyModelRepository
from src.infrastructure.persistence.repositories.event_repository import SQLAlchemyEventRepository
from src.application.services import TrainingService, GenerationService
from src.application.services.telares.telares_service import TelaresDetectionService
from src.application.services.telares.training_service import TelaresTrainingService
from src.application.queries.query_handlers import TrainingQueryHandler
from src.application.event_handlers.training_event_handlers import TrainingEventHandler, GenerationEventHandler
from src.application.message_bus import MessageBus, create_message_bus


logger = logging.getLogger(__name__)


class Container(containers.DeclarativeContainer):
    """Dependency injection container."""
    
    # Configuration
    config = providers.Singleton(get_cached_settings)
    
    # Database
    database_engine = providers.Singleton(
        create_engine,
        config.provided.database.url,
        echo=config.provided.database.echo,
        pool_size=config.provided.database.pool_size,
        max_overflow=config.provided.database.max_overflow,
        pool_timeout=config.provided.database.pool_timeout
    )
    
    database_session_factory = providers.Singleton(
        sessionmaker,
        bind=database_engine,
        autocommit=False,
        autoflush=False
    )
    
    # Repositories
    corpus_repository_factory = providers.Factory(
        SQLAlchemyCorpusRepository,
        session=providers.Callable(database_session_factory)
    )
    
    model_repository_factory = providers.Factory(
        SQLAlchemyModelRepository,
        session=providers.Callable(database_session_factory)
    )
    
    event_repository_factory = providers.Factory(
        SQLAlchemyEventRepository,
        session=providers.Callable(database_session_factory)
    )
    
    # Unit of Work
    unit_of_work_factory = providers.Factory(
        SQLAlchemyUnitOfWork,
        session_factory=database_session_factory
    )
    
    # Application Services
    training_service = providers.Factory(
        TrainingService,
        uow=unit_of_work_factory
    )
    
    generation_service = providers.Factory(
        GenerationService,
        uow=unit_of_work_factory
    )
    
    # Telares Services
    telares_detection_service = providers.Factory(
        TelaresDetectionService
    )
    
    telares_training_service = providers.Factory(
        TelaresTrainingService
    )
    
    # Query Handlers
    training_query_handler = providers.Factory(
        TrainingQueryHandler,
        uow=unit_of_work_factory
    )
    
    # Event Handlers
    training_event_handler = providers.Factory(
        TrainingEventHandler,
        uow=unit_of_work_factory
    )
    
    generation_event_handler = providers.Factory(
        GenerationEventHandler,
        uow=unit_of_work_factory
    )
    
    # Message Bus
    message_bus = providers.Singleton(
        create_message_bus,
        training_service=training_service,
        generation_service=generation_service,
        query_handler=training_query_handler,
        event_handler=training_event_handler
    )
    
    # Logging
    logger_factory = providers.Factory(
        logging.getLogger
    )


def get_container() -> Container:
    """Get the dependency injection container."""
    return Container()


def initialize_container(settings: Settings = None) -> Container:
    """Initialize and configure the dependency injection container."""
    if settings:
        # Override default settings
        container = Container()
        container.config.override(settings)
    else:
        container = get_container()
    
    # Initialize database
    engine = container.database_engine()
    Base.metadata.create_all(bind=engine)
    
    logger.info("Dependency injection container initialized")
    return container


def wire_container(container: Container, modules: list = None) -> None:
    """Wire the container with specified modules."""
    if modules is None:
        modules = [
            'src.api',
            'src.cli',
            'src.application.services',
            'src.application.queries',
            'src.application.event_handlers'
        ]
    
    try:
        container.wire(modules=modules)
        logger.info(f"Container wired with modules: {modules}")
    except Exception as e:
        logger.warning(f"Failed to wire some modules: {e}")


class ServiceLocator:
    """Service locator pattern for dependency resolution."""
    
    def __init__(self, container: Container):
        self.container = container
    
    def get_training_service(self) -> TrainingService:
        """Get training service instance."""
        return self.container.training_service()
    
    def get_generation_service(self) -> GenerationService:
        """Get generation service instance."""
        return self.container.generation_service()
    
    def get_telares_detection_service(self) -> TelaresDetectionService:
        """Get telares detection service instance."""
        return self.container.telares_detection_service()
    
    def get_telares_training_service(self) -> TelaresTrainingService:
        """Get telares training service instance."""
        return self.container.telares_training_service()
    
    def get_query_handler(self) -> TrainingQueryHandler:
        """Get query handler instance."""
        return self.container.training_query_handler()
    
    def get_message_bus(self) -> MessageBus:
        """Get message bus instance."""
        return self.container.message_bus()
    
    def get_unit_of_work(self) -> SQLAlchemyUnitOfWork:
        """Get unit of work instance."""
        return self.container.unit_of_work_factory()
    
    def get_settings(self) -> Settings:
        """Get application settings."""
        return self.container.config()


# Global service locator
_service_locator: ServiceLocator = None


def get_service_locator() -> ServiceLocator:
    """Get the global service locator."""
    global _service_locator
    if _service_locator is None:
        container = initialize_container()
        _service_locator = ServiceLocator(container)
    return _service_locator


def reset_service_locator() -> None:
    """Reset the global service locator."""
    global _service_locator
    _service_locator = None


# Dependency injection decorators
def inject_training_service(func):
    """Decorator to inject training service."""
    return inject(func)


def inject_generation_service(func):
    """Decorator to inject generation service."""
    return inject(func)


def inject_query_handler(func):
    """Decorator to inject query handler."""
    return inject(func)


def inject_message_bus(func):
    """Decorator to inject message bus."""
    return inject(func)


# Factory functions for common patterns
class ServiceFactory:
    """Factory for creating service instances."""
    
    def __init__(self, container: Container):
        self.container = container
    
    def create_training_service(self) -> TrainingService:
        """Create a new training service instance."""
        return self.container.training_service()
    
    def create_generation_service(self) -> GenerationService:
        """Create a new generation service instance."""
        return self.container.generation_service()
    
    def create_message_bus(self) -> MessageBus:
        """Create a new message bus instance."""
        return self.container.message_bus()
    
    def create_unit_of_work(self) -> SQLAlchemyUnitOfWork:
        """Create a new unit of work instance."""
        return self.container.unit_of_work_factory()


def get_service_factory() -> ServiceFactory:
    """Get service factory instance."""
    container = get_container()
    return ServiceFactory(container)


# Context managers for dependency scoping
class DatabaseScope:
    """Context manager for database session scope."""
    
    def __init__(self, container: Container):
        self.container = container
        self.session = None
    
    def __enter__(self):
        self.session = self.container.database_session_factory()()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
            self.session.close()


class ServiceScope:
    """Context manager for service scope."""
    
    def __init__(self, container: Container):
        self.container = container
        self.services = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup any resources if needed
        self.services.clear()
    
    def get_service(self, service_type: type):
        """Get service instance with caching."""
        if service_type not in self.services:
            if service_type == TrainingService:
                self.services[service_type] = self.container.training_service()
            elif service_type == GenerationService:
                self.services[service_type] = self.container.generation_service()
            elif service_type == MessageBus:
                self.services[service_type] = self.container.message_bus()
            else:
                raise ValueError(f"Unknown service type: {service_type}")
        
        return self.services[service_type]