"""
Unit of Work implementation using SQLAlchemy.

Provides transaction management and repository coordination.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
import logging

from src.infrastructure.persistence.sqlalchemy_models import Base
from src.infrastructure.persistence.repositories.corpus_repository import SQLAlchemyCorpusRepository
from src.infrastructure.persistence.repositories.model_repository import SQLAlchemyModelRepository
from src.infrastructure.persistence.repositories.event_repository import SQLAlchemyEventRepository

logger = logging.getLogger(__name__)


class SQLAlchemyUnitOfWork:
    """SQLAlchemy implementation of Unit of Work pattern."""
    
    def __init__(self, connection_string: str = "sqlite:///robopoet.db"):
        """
        Initialize Unit of Work with database connection.
        
        Args:
            connection_string: Database connection string
        """
        self.engine = create_engine(
            connection_string,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600    # Recycle connections after 1 hour
        )
        self.session_factory = sessionmaker(bind=self.engine)
        self.session: Optional[Session] = None
        
        # Repository instances (initialized in __enter__)
        self.corpus: Optional[SQLAlchemyCorpusRepository] = None
        self.models: Optional[SQLAlchemyModelRepository] = None
        self.events: Optional[SQLAlchemyEventRepository] = None
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def __enter__(self):
        """Enter the unit of work context."""
        self.session = self.session_factory()
        
        # Initialize repositories with the session
        self.corpus = SQLAlchemyCorpusRepository(self.session)
        self.models = SQLAlchemyModelRepository(self.session)
        self.events = SQLAlchemyEventRepository(self.session)
        
        logger.debug("Unit of Work session started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the unit of work context."""
        try:
            if exc_type is not None:
                # Exception occurred, rollback
                self.rollback()
                logger.warning(f"Unit of Work rolled back due to exception: {exc_val}")
            else:
                # No exception, commit
                self.commit()
                logger.debug("Unit of Work committed successfully")
        finally:
            self.session.close()
            logger.debug("Unit of Work session closed")
    
    def commit(self) -> None:
        """Commit all changes in this unit of work."""
        if not self.session:
            raise RuntimeError("No active session. Use within context manager.")
        
        try:
            self.session.commit()
            logger.debug("Transaction committed")
        except SQLAlchemyError as e:
            logger.error(f"Error committing transaction: {e}")
            self.rollback()
            raise
    
    def rollback(self) -> None:
        """Rollback all changes in this unit of work."""
        if not self.session:
            raise RuntimeError("No active session. Use within context manager.")
        
        try:
            self.session.rollback()
            logger.debug("Transaction rolled back")
        except SQLAlchemyError as e:
            logger.error(f"Error rolling back transaction: {e}")
            raise
    
    def flush(self) -> None:
        """Flush pending changes without committing."""
        if not self.session:
            raise RuntimeError("No active session. Use within context manager.")
        
        try:
            self.session.flush()
            logger.debug("Session flushed")
        except SQLAlchemyError as e:
            logger.error(f"Error flushing session: {e}")
            raise
    
    def refresh(self, instance):
        """Refresh an instance from the database."""
        if not self.session:
            raise RuntimeError("No active session. Use within context manager.")
        
        self.session.refresh(instance)
    
    @property
    def is_active(self) -> bool:
        """Check if the unit of work has an active session."""
        return self.session is not None and self.session.is_active


class UnitOfWorkManager:
    """Manager for creating and configuring Unit of Work instances."""
    
    def __init__(self, connection_string: str = "sqlite:///robopoet.db"):
        self.connection_string = connection_string
    
    def create(self) -> SQLAlchemyUnitOfWork:
        """Create a new Unit of Work instance."""
        return SQLAlchemyUnitOfWork(self.connection_string)
    
    @classmethod
    def for_testing(cls) -> SQLAlchemyUnitOfWork:
        """Create Unit of Work for testing with in-memory database."""
        return SQLAlchemyUnitOfWork("sqlite:///:memory:")
    
    @classmethod
    def for_production(cls, host: str, database: str, username: str, password: str) -> SQLAlchemyUnitOfWork:
        """Create Unit of Work for production with PostgreSQL."""
        connection_string = f"postgresql://{username}:{password}@{host}/{database}"
        return SQLAlchemyUnitOfWork(connection_string)