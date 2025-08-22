"""
Message bus for coordinating commands, queries, and events.

The message bus implements the Mediator pattern to decouple
application components and enable cross-cutting concerns.
"""

import logging
from typing import Dict, List, Callable, Any, Type, Union
from dataclasses import dataclass

from src.domain.events.training_events import DomainEvent
from src.application.commands.training_commands import (
    CreateCorpusCommand, TrainModelCommand, StopTrainingCommand,
    ArchiveModelCommand, DeleteCorpusCommand, UpdateCorpusCommand
)
from src.application.queries.training_queries import (
    GetModelByIdQuery, GetCorpusByIdQuery, ListModelsQuery, ListCorpusesQuery,
    GetTrainingMetricsQuery, GetModelsByCorpusQuery, SearchModelsQuery,
    SearchCorpusesQuery, GetModelStatisticsQuery, GetCorpusStatisticsQuery,
    GetRecentActivityQuery, GetModelComparisonQuery
)


logger = logging.getLogger(__name__)


@dataclass
class MessageContext:
    """Context information for message processing."""
    user_id: str = "system"
    request_id: str = ""
    correlation_id: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MessageBus:
    """
    Message bus for handling commands, queries, and events.
    
    Implements the Mediator pattern to decouple application components
    and enable cross-cutting concerns like logging, validation, and monitoring.
    """
    
    def __init__(self):
        self._command_handlers: Dict[Type, Callable] = {}
        self._query_handlers: Dict[Type, Callable] = {}
        self._event_handlers: Dict[Type, List[Callable]] = {}
        self._middleware: List[Callable] = []
    
    def register_command_handler(self, command_type: Type, handler: Callable) -> None:
        """Register a command handler."""
        if command_type in self._command_handlers:
            raise ValueError(f"Command handler for {command_type} already registered")
        
        self._command_handlers[command_type] = handler
        logger.debug(f"Registered command handler for {command_type.__name__}")
    
    def register_query_handler(self, query_type: Type, handler: Callable) -> None:
        """Register a query handler."""
        if query_type in self._query_handlers:
            raise ValueError(f"Query handler for {query_type} already registered")
        
        self._query_handlers[query_type] = handler
        logger.debug(f"Registered query handler for {query_type.__name__}")
    
    def register_event_handler(self, event_type: Type, handler: Callable) -> None:
        """Register an event handler (multiple handlers allowed per event)."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Registered event handler for {event_type.__name__}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware for cross-cutting concerns."""
        self._middleware.append(middleware)
        logger.debug(f"Added middleware: {middleware.__name__}")
    
    def handle_command(self, command: Any, context: MessageContext = None) -> Any:
        """Handle a command and return the result."""
        if context is None:
            context = MessageContext()
        
        command_type = type(command)
        
        if command_type not in self._command_handlers:
            raise ValueError(f"No handler registered for command {command_type.__name__}")
        
        logger.info(
            f"Handling command: {command_type.__name__}",
            extra={
                'command_type': command_type.__name__,
                'user_id': context.user_id,
                'request_id': context.request_id
            }
        )
        
        try:
            # Apply middleware
            for middleware in self._middleware:
                middleware('command', command, context)
            
            # Execute command handler
            handler = self._command_handlers[command_type]
            result = handler(command)
            
            logger.info(f"Command {command_type.__name__} completed successfully")
            return result
            
        except Exception as e:
            logger.error(
                f"Command {command_type.__name__} failed: {str(e)}",
                extra={
                    'command_type': command_type.__name__,
                    'error': str(e),
                    'user_id': context.user_id,
                    'request_id': context.request_id
                }
            )
            raise
    
    def handle_query(self, query: Any, context: MessageContext = None) -> Any:
        """Handle a query and return the result."""
        if context is None:
            context = MessageContext()
        
        query_type = type(query)
        
        if query_type not in self._query_handlers:
            raise ValueError(f"No handler registered for query {query_type.__name__}")
        
        logger.debug(
            f"Handling query: {query_type.__name__}",
            extra={
                'query_type': query_type.__name__,
                'user_id': context.user_id,
                'request_id': context.request_id
            }
        )
        
        try:
            # Apply middleware
            for middleware in self._middleware:
                middleware('query', query, context)
            
            # Execute query handler
            handler = self._query_handlers[query_type]
            result = handler(query)
            
            logger.debug(f"Query {query_type.__name__} completed successfully")
            return result
            
        except Exception as e:
            logger.error(
                f"Query {query_type.__name__} failed: {str(e)}",
                extra={
                    'query_type': query_type.__name__,
                    'error': str(e),
                    'user_id': context.user_id,
                    'request_id': context.request_id
                }
            )
            raise
    
    def publish_event(self, event: DomainEvent, context: MessageContext = None) -> None:
        """Publish a domain event to all registered handlers."""
        if context is None:
            context = MessageContext()
        
        event_type = type(event)
        
        logger.info(
            f"Publishing event: {event_type.__name__}",
            extra={
                'event_type': event_type.__name__,
                'event_id': getattr(event, 'id', 'unknown'),
                'user_id': context.user_id,
                'request_id': context.request_id
            }
        )
        
        # Get handlers for this event type
        handlers = self._event_handlers.get(event_type, [])
        
        if not handlers:
            logger.warning(f"No handlers registered for event {event_type.__name__}")
            return
        
        # Execute all handlers
        for handler in handlers:
            try:
                # Apply middleware
                for middleware in self._middleware:
                    middleware('event', event, context)
                
                handler(event)
                logger.debug(f"Event handler {handler.__name__} completed for {event_type.__name__}")
                
            except Exception as e:
                logger.error(
                    f"Event handler {handler.__name__} failed for {event_type.__name__}: {str(e)}",
                    extra={
                        'event_type': event_type.__name__,
                        'handler': handler.__name__,
                        'error': str(e),
                        'user_id': context.user_id,
                        'request_id': context.request_id
                    }
                )
                # Continue with other handlers even if one fails
    
    def get_registered_handlers(self) -> Dict[str, Any]:
        """Get information about registered handlers for debugging."""
        return {
            'commands': list(self._command_handlers.keys()),
            'queries': list(self._query_handlers.keys()),
            'events': {
                event_type: len(handlers) 
                for event_type, handlers in self._event_handlers.items()
            },
            'middleware': len(self._middleware)
        }


def create_message_bus(training_service, generation_service, query_handler, event_handler) -> MessageBus:
    """
    Factory function to create and configure a message bus with all handlers.
    
    This function wires up all the command, query, and event handlers
    to create a fully configured message bus.
    """
    bus = MessageBus()
    
    # Register command handlers
    bus.register_command_handler(CreateCorpusCommand, training_service.create_corpus)
    bus.register_command_handler(TrainModelCommand, training_service.train_model)
    bus.register_command_handler(StopTrainingCommand, training_service.stop_training)
    bus.register_command_handler(ArchiveModelCommand, training_service.archive_model)
    bus.register_command_handler(DeleteCorpusCommand, training_service.delete_corpus)
    bus.register_command_handler(UpdateCorpusCommand, training_service.update_corpus)
    
    # Register query handlers
    bus.register_query_handler(GetModelByIdQuery, query_handler.handle_get_model_by_id)
    bus.register_query_handler(GetCorpusByIdQuery, query_handler.handle_get_corpus_by_id)
    bus.register_query_handler(ListModelsQuery, query_handler.handle_list_models)
    bus.register_query_handler(ListCorpusesQuery, query_handler.handle_list_corpuses)
    bus.register_query_handler(GetTrainingMetricsQuery, query_handler.handle_get_training_metrics)
    bus.register_query_handler(SearchModelsQuery, query_handler.handle_search_models)
    bus.register_query_handler(GetModelStatisticsQuery, query_handler.handle_get_model_statistics)
    bus.register_query_handler(GetRecentActivityQuery, query_handler.handle_get_recent_activity)
    
    # Register event handlers
    from src.domain.events.training_events import (
        TrainingStarted, TrainingCompleted, TrainingFailed,
        CorpusCreated, ModelArchived
    )
    
    bus.register_event_handler(TrainingStarted, event_handler.handle_training_started)
    bus.register_event_handler(TrainingCompleted, event_handler.handle_training_completed)
    bus.register_event_handler(TrainingFailed, event_handler.handle_training_failed)
    bus.register_event_handler(CorpusCreated, event_handler.handle_corpus_created)
    bus.register_event_handler(ModelArchived, event_handler.handle_model_archived)
    
    # Add default middleware
    bus.add_middleware(_logging_middleware)
    bus.add_middleware(_validation_middleware)
    
    logger.info("Message bus created and configured with all handlers")
    return bus


def _logging_middleware(message_type: str, message: Any, context: MessageContext) -> None:
    """Middleware for structured logging."""
    logger.debug(
        f"Processing {message_type}: {type(message).__name__}",
        extra={
            'message_type': message_type,
            'message_class': type(message).__name__,
            'user_id': context.user_id,
            'request_id': context.request_id,
            'correlation_id': context.correlation_id
        }
    )


def _validation_middleware(message_type: str, message: Any, context: MessageContext) -> None:
    """Middleware for message validation."""
    if message_type == 'command' and hasattr(message, 'validate'):
        try:
            message.validate()
        except Exception as e:
            logger.error(f"Command validation failed: {str(e)}")
            raise ValueError(f"Invalid command: {str(e)}")