"""
Logging Integration Utilities

Provides easy integration of structured logging and error handling
throughout the Robo-Poet codebase with minimal code changes.
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
import inspect

from .structured_logger import get_logger, LogFormat, CorrelationManager, LogContext, PerformanceMetric
from ..core.exceptions import ErrorContextManager, get_global_error_handler, RoboPoetError


def setup_structured_logging(
    app_name: str = "robo_poet",
    log_format: LogFormat = LogFormat.CONSOLE,
    log_file: Optional[str] = None,
    enable_performance_tracking: bool = True
):
    """
    Setup structured logging for the entire application.
    
    Args:
        app_name: Application name for logger hierarchy
        log_format: Log output format
        log_file: Optional log file path
        enable_performance_tracking: Whether to enable performance metrics
    """
    # Configure root logger
    logger = get_logger(
        name=app_name,
        log_format=log_format,
        output_file=log_file,
        enable_metrics=enable_performance_tracking
    )
    
    logger.info(f"Structured logging initialized for {app_name}")
    return logger


def logged_operation(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    track_performance: bool = True,
    handle_errors: bool = True
):
    """
    Decorator to automatically log operations with correlation tracking.
    
    Args:
        component: Component name (defaults to module name)
        operation: Operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        track_performance: Whether to track performance metrics
        handle_errors: Whether to automatically handle errors
    """
    def decorator(func: Callable):
        # Extract component and operation names if not provided
        func_component = component or func.__module__.split('.')[-1]
        func_operation = operation or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"robo_poet.{func_component}")
            
            # Create operation context
            with logger.operation_context(func_operation, func_component) as ctx:
                # Log function arguments if requested
                if log_args:
                    # Get parameter names
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    # Filter out sensitive arguments
                    safe_args = {
                        k: v for k, v in bound_args.arguments.items()
                        if not any(sensitive in k.lower() for sensitive in ['password', 'token', 'secret', 'key'])
                    }
                    
                    logger.debug(f"Calling {func_operation} with args: {safe_args}", ctx)
                
                try:
                    if handle_errors:
                        # Use error context manager
                        with ErrorContextManager(func_component, func_operation) as error_ctx:
                            result = func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Log result if requested
                    if log_result and result is not None:
                        # Truncate large results
                        result_str = str(result)
                        if len(result_str) > 200:
                            result_str = result_str[:200] + "..."
                        logger.debug(f"{func_operation} returned: {result_str}", ctx)
                    
                    return result
                    
                except Exception as e:
                    if handle_errors and isinstance(e, RoboPoetError):
                        # Let error handler attempt recovery
                        error_handler = get_global_error_handler()
                        recovered = error_handler.handle_error(e, attempt_recovery=False)  # Just log, don't recover
                    raise
        
        return wrapper
    return decorator


def error_boundary(
    component: str,
    operation: str,
    fallback_result: Any = None,
    suppress_errors: bool = False,
    attempt_recovery: bool = True
):
    """
    Decorator to create an error boundary with automatic recovery attempts.
    
    Args:
        component: Component name for error context
        operation: Operation name for error context
        fallback_result: Result to return if error occurs and is suppressed
        suppress_errors: Whether to suppress errors and return fallback
        attempt_recovery: Whether to attempt automatic error recovery
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with ErrorContextManager(component, operation) as ctx:
                    return func(*args, **kwargs)
            
            except RoboPoetError as e:
                if attempt_recovery:
                    error_handler = get_global_error_handler()
                    recovered = error_handler.handle_error(e)
                    
                    if recovered:
                        # Retry operation after recovery
                        try:
                            return func(*args, **kwargs)
                        except:
                            pass  # Recovery didn't work, proceed with normal error handling
                
                if suppress_errors:
                    logger = get_logger(f"robo_poet.{component}")
                    logger.warning(f"Error suppressed in {operation}, returning fallback", exception=e)
                    return fallback_result
                else:
                    raise
            
            except Exception as e:
                # Convert non-RoboPoet errors
                from ..core.exceptions import SystemError, ErrorContext
                
                error_context = ErrorContext(
                    component=component,
                    operation=operation,
                    parameters={},
                    system_state={},
                    suggestions=[f"Review {component}.{operation} implementation"]
                )
                
                system_error = SystemError(
                    f"Unexpected error in {component}.{operation}: {str(e)}",
                    context=error_context,
                    cause=e
                )
                
                if suppress_errors:
                    logger = get_logger(f"robo_poet.{component}")
                    logger.error(f"Error suppressed in {operation}, returning fallback", exception=system_error)
                    return fallback_result
                else:
                    raise system_error
        
        return wrapper
    return decorator


def with_correlation(correlation_id: Optional[str] = None):
    """
    Decorator to ensure correlation ID is set for the duration of function execution.
    
    Args:
        correlation_id: Specific correlation ID to use (generates one if None)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with CorrelationManager.correlation_context(correlation_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def performance_monitor(operation: str, component: Optional[str] = None):
    """
    Decorator to monitor and log performance metrics.
    
    Args:
        operation: Operation name for metrics
        component: Component name (defaults to module name)
    """
    def decorator(func: Callable):
        func_component = component or func.__module__.split('.')[-1]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"robo_poet.{func_component}")
            
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Create performance metric
                metric = PerformanceMetric(
                    operation=operation,
                    duration_ms=duration_ms
                )
                
                # Log performance
                logger.log_performance(metric)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.warning(f"Operation {operation} failed after {duration_ms:.1f}ms")
                raise
        
        return wrapper
    return decorator


class LoggingMixin:
    """
    Mixin class to add structured logging capabilities to any class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(f"robo_poet.{self.__class__.__name__.lower()}")
    
    def log_operation(self, operation: str, message: str, **metadata):
        """Log an operation with metadata."""
        context = LogContext(
            correlation_id=CorrelationManager.get_correlation_id(),
            component=self.__class__.__name__,
            operation=operation,
            metadata=metadata
        )
        self._logger.info(message, context)
    
    def log_error(self, operation: str, error: Exception, **metadata):
        """Log an error with context."""
        context = LogContext(
            correlation_id=CorrelationManager.get_correlation_id(),
            component=self.__class__.__name__,
            operation=operation,
            metadata=metadata
        )
        self._logger.error(f"Error in {operation}: {str(error)}", context, exception=error)
    
    def operation_context(self, operation: str, **metadata):
        """Create operation context for the class."""
        return self._logger.operation_context(operation, self.__class__.__name__, **metadata)


def quick_setup(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Quick setup for structured logging with sensible defaults.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    import logging
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Setup with console format and optional file
    logger = setup_structured_logging(
        app_name="robo_poet",
        log_format=LogFormat.CONSOLE,
        log_file=log_file
    )
    
    # Set log level
    logger.logger.setLevel(numeric_level)
    
    logger.info(f"Quick setup complete - logging level: {log_level}")
    
    return logger


# Usage examples
if __name__ == "__main__":
    # Quick setup
    logger = quick_setup("DEBUG", "robo_poet.log")
    
    # Example decorated function
    @logged_operation(component="demo", operation="test_function", log_args=True)
    @performance_monitor("demo_operation")
    def example_function(name: str, value: int) -> str:
        """Example function with logging."""
        import time
        time.sleep(0.1)  # Simulate work
        return f"Hello {name}, value is {value}"
    
    # Example error boundary
    @error_boundary("demo", "risky_operation", fallback_result="fallback", suppress_errors=True)
    def risky_function():
        """Function that might fail."""
        raise ValueError("Something went wrong")
    
    # Test the decorators
    with CorrelationManager.correlation_context():
        result = example_function("test", 42)
        print(f"Result: {result}")
        
        fallback = risky_function()
        print(f"Fallback: {fallback}")
    
    # Show final statistics
    print("\nFinal logger statistics:")
    print(f"Error summary: {logger.get_error_summary()}")
    print(f"Performance summary: {logger.get_performance_summary()}")