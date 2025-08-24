"""
Structured Logging System with Correlation IDs and Centralized Error Reporting.

Provides enhanced logging capabilities with:
- Correlation ID tracking across requests/operations
- Structured JSON logging for production environments
- Performance metrics collection
- Error aggregation and reporting
- Integration with existing RoboPoetError hierarchy
"""

import logging
import json
import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import sys

from ..core.exceptions import RoboPoetError, ErrorSeverity, ErrorCategory


class LogFormat(str, Enum):
    """Log output formats."""
    CONSOLE = "console"
    JSON = "json"
    DETAILED = "detailed"


@dataclass
class LogContext:
    """Context information for structured logging."""
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        if self.metadata:
            data.update(self.metadata)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class PerformanceMetric:
    """Performance measurement data."""
    operation: str
    duration_ms: float
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class CorrelationManager:
    """Thread-safe correlation ID management."""
    
    _local = threading.local()
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get current correlation ID or create new one."""
        if not hasattr(cls._local, 'correlation_id'):
            cls._local.correlation_id = str(uuid.uuid4())[:8]
        return cls._local.correlation_id
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear correlation ID for current thread."""
        if hasattr(cls._local, 'correlation_id'):
            delattr(cls._local, 'correlation_id')
    
    @classmethod
    @contextmanager
    def correlation_context(cls, correlation_id: Optional[str] = None):
        """Context manager for correlation ID scope."""
        old_id = getattr(cls._local, 'correlation_id', None)
        try:
            if correlation_id:
                cls.set_correlation_id(correlation_id)
            else:
                cls.set_correlation_id(str(uuid.uuid4())[:8])
            yield cls.get_correlation_id()
        finally:
            if old_id:
                cls.set_correlation_id(old_id)
            else:
                cls.clear_correlation_id()


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, format_type: LogFormat = LogFormat.JSON, include_context: bool = True):
        super().__init__()
        self.format_type = format_type
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record according to specified format."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add correlation ID
        if self.include_context:
            try:
                log_data['correlation_id'] = CorrelationManager.get_correlation_id()
            except:
                pass  # Don't fail logging if correlation ID unavailable
        
        # Add exception information
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 
                          'relativeCreated', 'thread', 'threadName', 'processName', 
                          'process', 'getMessage', 'stack_info', 'exc_info', 'exc_text'}:
                extra_fields[key] = value
        
        if extra_fields:
            log_data.update(extra_fields)
        
        # Format according to type
        if self.format_type == LogFormat.JSON:
            return json.dumps(log_data, default=str, ensure_ascii=False)
        elif self.format_type == LogFormat.CONSOLE:
            correlation = f"[{log_data.get('correlation_id', 'N/A')}]" if self.include_context else ""
            return f"{log_data['timestamp']} {log_data['level']:<8} {correlation} {log_data['logger']}: {log_data['message']}"
        else:  # DETAILED
            return f"{log_data['timestamp']} [{log_data['level']}] {log_data['logger']}:{log_data['line']} - {log_data['message']}"


class ErrorAggregator:
    """Aggregates and reports error patterns."""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def add_error(self, error: RoboPoetError, context: Optional[LogContext] = None):
        """Add error to aggregation."""
        with self._lock:
            error_key = f"{error.__class__.__name__}:{error.category.value}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'error_type': error.__class__.__name__,
                'error_message': error.message,
                'category': error.category.value,
                'severity': error.severity.value,
                'count': self.error_counts[error_key]
            }
            
            if context:
                error_data['context'] = context.to_dict()
            
            if error.context:
                error_data['error_context'] = error.context.to_dict()
            
            self.errors.append(error_data)
            
            # Maintain size limit
            if len(self.errors) > self.max_errors:
                self.errors = self.errors[-self.max_errors:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of aggregated errors."""
        with self._lock:
            return {
                'total_errors': len(self.errors),
                'error_counts': self.error_counts.copy(),
                'top_errors': sorted(
                    self.error_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10],
                'last_updated': datetime.now().isoformat()
            }
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent errors."""
        with self._lock:
            return self.errors[-count:] if self.errors else []


class StructuredLogger:
    """Enhanced structured logger with correlation tracking and metrics."""
    
    def __init__(
        self, 
        name: str,
        log_format: LogFormat = LogFormat.CONSOLE,
        level: int = logging.INFO,
        include_context: bool = True,
        enable_metrics: bool = True,
        output_file: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.include_context = include_context
        self.enable_metrics = enable_metrics
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = StructuredFormatter(log_format, include_context)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_formatter = StructuredFormatter(LogFormat.JSON, include_context)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Error aggregator
        self.error_aggregator = ErrorAggregator()
        
        # Performance metrics storage
        self.performance_metrics: List[PerformanceMetric] = []
    
    def _log_with_context(self, level: int, message: str, context: Optional[LogContext] = None, 
                         exception: Optional[Exception] = None, **kwargs):
        """Log with optional context and exception."""
        extra = {}
        
        if context:
            extra.update(context.to_dict())
        
        extra.update(kwargs)
        
        if exception:
            self.logger.log(level, message, extra=extra, exc_info=True)
            
            # Add to error aggregator if it's a RoboPoetError
            if isinstance(exception, RoboPoetError):
                self.error_aggregator.add_error(exception, context)
        else:
            self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, context, **kwargs)
    
    def info(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, context, **kwargs)
    
    def warning(self, message: str, context: Optional[LogContext] = None, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, context, **kwargs)
    
    def error(self, message: str, context: Optional[LogContext] = None, 
              exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception."""
        self._log_with_context(logging.ERROR, message, context, exception, **kwargs)
    
    def critical(self, message: str, context: Optional[LogContext] = None, 
                 exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception."""
        self._log_with_context(logging.CRITICAL, message, context, exception, **kwargs)
    
    def log_performance(self, metric: PerformanceMetric, context: Optional[LogContext] = None):
        """Log performance metric."""
        if self.enable_metrics:
            self.performance_metrics.append(metric)
            
            # Log as info with structured data
            self.info(
                f"Performance: {metric.operation} completed in {metric.duration_ms:.1f}ms",
                context,
                **metric.to_dict()
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get aggregated error summary."""
        return self.error_aggregator.get_error_summary()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.performance_metrics:
            return {'total_operations': 0}
        
        by_operation = {}
        total_duration = 0
        
        for metric in self.performance_metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = {
                    'count': 0,
                    'total_duration_ms': 0,
                    'avg_duration_ms': 0,
                    'max_duration_ms': 0,
                    'min_duration_ms': float('inf')
                }
            
            op_stats = by_operation[metric.operation]
            op_stats['count'] += 1
            op_stats['total_duration_ms'] += metric.duration_ms
            op_stats['max_duration_ms'] = max(op_stats['max_duration_ms'], metric.duration_ms)
            op_stats['min_duration_ms'] = min(op_stats['min_duration_ms'], metric.duration_ms)
            total_duration += metric.duration_ms
        
        # Calculate averages
        for op_stats in by_operation.values():
            op_stats['avg_duration_ms'] = op_stats['total_duration_ms'] / op_stats['count']
            if op_stats['min_duration_ms'] == float('inf'):
                op_stats['min_duration_ms'] = 0
        
        return {
            'total_operations': len(self.performance_metrics),
            'total_duration_ms': total_duration,
            'operations': by_operation
        }
    
    @contextmanager
    def operation_context(self, operation: str, component: Optional[str] = None, **metadata):
        """Context manager for tracking operation performance and logging."""
        context = LogContext(
            correlation_id=CorrelationManager.get_correlation_id(),
            component=component,
            operation=operation,
            metadata=metadata
        )
        
        start_time = time.time()
        self.info(f"Starting {operation}", context)
        
        try:
            yield context
            
            # Log successful completion
            duration_ms = (time.time() - start_time) * 1000
            self.info(f"Completed {operation} in {duration_ms:.1f}ms", context)
            
            # Record performance metric
            if self.enable_metrics:
                metric = PerformanceMetric(
                    operation=operation,
                    duration_ms=duration_ms
                )
                self.log_performance(metric, context)
                
        except Exception as e:
            # Log error with context
            duration_ms = (time.time() - start_time) * 1000
            self.error(
                f"Failed {operation} after {duration_ms:.1f}ms: {str(e)}", 
                context, 
                exception=e
            )
            raise


# Global logger instance factory
_loggers: Dict[str, StructuredLogger] = {}

def get_logger(
    name: str,
    log_format: LogFormat = LogFormat.CONSOLE,
    level: int = logging.INFO,
    include_context: bool = True,
    enable_metrics: bool = True,
    output_file: Optional[str] = None
) -> StructuredLogger:
    """Get or create structured logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(
            name=name,
            log_format=log_format,
            level=level,
            include_context=include_context,
            enable_metrics=enable_metrics,
            output_file=output_file
        )
    return _loggers[name]


# Utility functions for common patterns
def log_error_with_context(
    logger: StructuredLogger,
    error: RoboPoetError,
    operation: str,
    component: str,
    **metadata
):
    """Utility to log RoboPoetError with full context."""
    context = LogContext(
        correlation_id=CorrelationManager.get_correlation_id(),
        component=component,
        operation=operation,
        metadata=metadata
    )
    
    logger.error(
        f"Error in {component}.{operation}: {error.message}",
        context=context,
        exception=error,
        error_category=error.category.value,
        error_severity=error.severity.value
    )


def create_performance_decorator(logger: StructuredLogger):
    """Create performance tracking decorator."""
    def decorator(operation_name: str):
        def wrapper(func: Callable):
            def wrapped(*args, **kwargs):
                with logger.operation_context(operation_name, component=func.__module__):
                    return func(*args, **kwargs)
            return wrapped
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    # Create logger with different formats
    console_logger = get_logger("test.console", LogFormat.CONSOLE)
    json_logger = get_logger("test.json", LogFormat.JSON, output_file="test.log")
    
    # Test basic logging
    with CorrelationManager.correlation_context():
        context = LogContext(
            correlation_id=CorrelationManager.get_correlation_id(),
            component="TestComponent",
            operation="test_operation",
            user_id="user123"
        )
        
        console_logger.info("Test message", context)
        json_logger.info("JSON test message", context)
        
        # Test performance tracking
        with console_logger.operation_context("test_performance") as ctx:
            time.sleep(0.1)  # Simulate work
        
        # Test error handling
        try:
            from ..core.exceptions import SystemError
            raise SystemError("Test system error")
        except SystemError as e:
            console_logger.error("System error occurred", context, exception=e)
        
        # Show summaries
        print("\nError Summary:", console_logger.get_error_summary())
        print("\nPerformance Summary:", console_logger.get_performance_summary())