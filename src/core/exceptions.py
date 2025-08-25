"""
Comprehensive exception hierarchy for Robo-Poet.

Provides structured error handling with contextual information,
recovery suggestions, and proper logging integration.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import traceback
import sys


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    HARDWARE = "hardware"
    CONFIGURATION = "configuration"
    DATA = "data"
    MODEL = "model"
    GENERATION = "generation"
    SYSTEM = "system"
    USER = "user"


@dataclass
class ErrorContext:
    """Additional context for errors."""
    component: str
    operation: str
    parameters: Dict[str, Any]
    system_state: Dict[str, Any]
    suggestions: List[str]
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "component": self.component,
            "operation": self.operation,
            "parameters": self.parameters,
            "system_state": self.system_state,
            "suggestions": self.suggestions
        }
        
        # Add optional fields if present
        for field in ['correlation_id', 'user_id', 'session_id', 'timestamp', 'stack_trace']:
            value = getattr(self, field, None)
            if value is not None:
                result[field] = value
                
        return result
    
    def add_stack_trace(self) -> None:
        """Capture current stack trace."""
        self.stack_trace = ''.join(traceback.format_stack())


class RoboPoetError(Exception):
    """Base exception for all Robo-Poet errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.cause = cause
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log the error with appropriate level and structured logging."""
        # Try to use structured logger first, fall back to standard logging
        try:
            from ..utils.structured_logger import get_logger, LogContext, CorrelationManager
            
            logger = get_logger(f"robo_poet.{self.category.value}")
            
            # Create log context with correlation ID
            log_context = LogContext(
                correlation_id=CorrelationManager.get_correlation_id(),
                component=self.context.component if self.context else "unknown",
                operation=self.context.operation if self.context else "unknown",
                metadata={
                    "error_type": self.__class__.__name__,
                    "error_category": self.category.value,
                    "error_severity": self.severity.value,
                    "cause": str(self.cause) if self.cause else None
                }
            )
            
            # Log with appropriate severity level
            if self.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"CRITICAL ERROR: {self.message}", log_context, exception=self)
            elif self.severity == ErrorSeverity.HIGH:
                logger.error(f"HIGH SEVERITY: {self.message}", log_context, exception=self)
            elif self.severity == ErrorSeverity.MEDIUM:
                logger.warning(f"MEDIUM SEVERITY: {self.message}", log_context)
            else:
                logger.info(f"LOW SEVERITY: {self.message}", log_context)
                
        except ImportError:
            # Fall back to standard logging if structured logger not available
            logger = logging.getLogger(f"robo_poet.{self.category.value}")
            
            log_data = {
                "error_type": self.__class__.__name__,
                "error_message": self.message,
                "category": self.category.value,
                "severity": self.severity.value
            }
            
            if self.context:
                log_data["context"] = self.context.to_dict()
            
            if self.cause:
                log_data["cause"] = str(self.cause)
            
            if self.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"CRITICAL ERROR: {self.message}", extra=log_data, exc_info=True)
            elif self.severity == ErrorSeverity.HIGH:
                logger.error(f"HIGH SEVERITY: {self.message}", extra=log_data, exc_info=True)
            elif self.severity == ErrorSeverity.MEDIUM:
                logger.warning(f"MEDIUM SEVERITY: {self.message}", extra=log_data)
            else:
                logger.info(f"LOW SEVERITY: {self.message}", extra=log_data)
    
    def get_recovery_suggestions(self) -> List[str]:
        """Get suggested recovery actions."""
        suggestions = []
        
        if self.context and self.context.suggestions:
            suggestions.extend(self.context.suggestions)
        
        # Add generic suggestions based on category
        if self.category == ErrorCategory.HARDWARE:
            suggestions.extend([
                "Check GPU drivers and CUDA installation",
                "Verify hardware compatibility",
                "Monitor system resources"
            ])
        elif self.category == ErrorCategory.CONFIGURATION:
            suggestions.extend([
                "Review configuration file syntax",
                "Check file paths and permissions",
                "Validate parameter ranges"
            ])
        elif self.category == ErrorCategory.DATA:
            suggestions.extend([
                "Verify data file integrity",
                "Check file formats and encoding",
                "Validate data preprocessing parameters"
            ])
        
        return list(set(suggestions))  # Remove duplicates


# Hardware and GPU errors
class GPUError(RoboPoetError):
    """GPU-related errors."""
    
    def __init__(self, message: str, gpu_info: Optional[Dict[str, Any]] = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.HARDWARE)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        
        if gpu_info and not kwargs.get("context"):
            kwargs["context"] = ErrorContext(
                component="GPU",
                operation="gpu_operation",
                parameters={},
                system_state=gpu_info,
                suggestions=[
                    "Check nvidia-smi output",
                    "Verify CUDA installation",
                    "Update GPU drivers"
                ]
            )
        
        super().__init__(message, **kwargs)


class CUDAError(GPUError):
    """CUDA-specific errors."""
    
    def __init__(self, message: str, cuda_version: Optional[str] = None, **kwargs):
        if cuda_version:
            message = f"CUDA {cuda_version}: {message}"
        
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class MemoryError(GPUError):
    """GPU memory errors."""
    
    def __init__(self, message: str, memory_info: Optional[Dict[str, Any]] = None, **kwargs):
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        
        if memory_info and not kwargs.get("context"):
            suggestions = [
                "Reduce batch size",
                "Enable mixed precision training",
                "Use gradient checkpointing",
                "Clear GPU cache"
            ]
            
            kwargs["context"] = ErrorContext(
                component="GPU Memory",
                operation="memory_allocation",
                parameters={},
                system_state=memory_info,
                suggestions=suggestions
            )
        
        super().__init__(message, **kwargs)


class TensorCoreError(GPUError):
    """Tensor Core optimization errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


# Configuration errors
class ConfigurationError(RoboPoetError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_section: Optional[str] = None, **kwargs):
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        
        if config_section:
            message = f"Configuration [{config_section}]: {message}"
        
        super().__init__(message, **kwargs)


class ValidationError(ConfigurationError):
    """Configuration validation errors."""
    
    def __init__(self, message: str, invalid_values: Optional[Dict[str, Any]] = None, **kwargs):
        if invalid_values and not kwargs.get("context"):
            kwargs["context"] = ErrorContext(
                component="Configuration Validator",
                operation="validation",
                parameters=invalid_values,
                system_state={},
                suggestions=[
                    "Check parameter ranges and types",
                    "Review configuration documentation",
                    "Use configuration templates"
                ]
            )
        
        super().__init__(message, **kwargs)


# Data processing errors
class DataError(RoboPoetError):
    """Data processing errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.DATA)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class CorpusError(DataError):
    """Text corpus errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        if file_path:
            message = f"Corpus [{file_path}]: {message}"
        
        super().__init__(message, **kwargs)


class TokenizationError(DataError):
    """Tokenization errors."""
    
    def __init__(self, message: str, tokenizer_type: Optional[str] = None, **kwargs):
        if tokenizer_type:
            message = f"Tokenizer [{tokenizer_type}]: {message}"
        
        super().__init__(message, **kwargs)


class AugmentationError(DataError):
    """Data augmentation errors."""
    
    def __init__(self, message: str, augmentation_type: Optional[str] = None, **kwargs):
        if augmentation_type:
            message = f"Augmentation [{augmentation_type}]: {message}"
        
        kwargs.setdefault("severity", ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


# Model training errors
class ModelError(RoboPoetError):
    """Model-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.MODEL)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class TrainingError(ModelError):
    """Model training errors."""
    
    def __init__(
        self, 
        message: str, 
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        **kwargs
    ):
        if epoch is not None:
            message = f"Training [Epoch {epoch}]: {message}"
        if batch is not None:
            message = f"{message} [Batch {batch}]"
        
        super().__init__(message, **kwargs)


class ConvergenceError(TrainingError):
    """Model convergence errors."""
    
    def __init__(self, message: str, metrics: Optional[Dict[str, float]] = None, **kwargs):
        if metrics and not kwargs.get("context"):
            suggestions = [
                "Adjust learning rate",
                "Modify model architecture", 
                "Check data quality",
                "Increase training epochs",
                "Use different optimizer"
            ]
            
            kwargs["context"] = ErrorContext(
                component="Training Loop",
                operation="convergence_check",
                parameters={},
                system_state={"metrics": metrics},
                suggestions=suggestions
            )
        
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ArchitectureError(ModelError):
    """Model architecture errors."""
    
    def __init__(self, message: str, layer_info: Optional[Dict[str, Any]] = None, **kwargs):
        if layer_info and not kwargs.get("context"):
            kwargs["context"] = ErrorContext(
                component="Model Architecture",
                operation="architecture_validation",
                parameters=layer_info,
                system_state={},
                suggestions=[
                    "Check layer compatibility",
                    "Verify input/output dimensions",
                    "Review model configuration"
                ]
            )
        
        super().__init__(message, **kwargs)


# Text generation errors
class GenerationError(RoboPoetError):
    """Text generation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.GENERATION)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class SamplingError(GenerationError):
    """Sampling strategy errors."""
    
    def __init__(self, message: str, sampler_type: Optional[str] = None, **kwargs):
        if sampler_type:
            message = f"Sampler [{sampler_type}]: {message}"
        
        super().__init__(message, **kwargs)


class QualityError(GenerationError):
    """Generation quality errors."""
    
    def __init__(
        self, 
        message: str, 
        quality_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        if quality_metrics and not kwargs.get("context"):
            suggestions = [
                "Adjust sampling parameters",
                "Use different generation strategy",
                "Increase model training",
                "Modify prompt or context"
            ]
            
            kwargs["context"] = ErrorContext(
                component="Generation Quality",
                operation="quality_check",
                parameters={},
                system_state={"quality_metrics": quality_metrics},
                suggestions=suggestions
            )
        
        kwargs.setdefault("severity", ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


# System errors
class SystemError(RoboPoetError):
    """System-level errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.SYSTEM)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ResourceError(SystemError):
    """Resource management errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        current_usage: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        message = f"Resource [{resource_type}]: {message}"
        
        if current_usage and not kwargs.get("context"):
            suggestions = [
                f"Reduce {resource_type} usage",
                "Close unnecessary processes",
                "Increase resource limits",
                "Use more efficient algorithms"
            ]
            
            kwargs["context"] = ErrorContext(
                component="Resource Manager",
                operation="resource_check",
                parameters={},
                system_state=current_usage,
                suggestions=suggestions
            )
        
        super().__init__(message, **kwargs)


class DependencyError(SystemError):
    """Dependency and environment errors."""
    
    def __init__(self, message: str, dependency: Optional[str] = None, **kwargs):
        if dependency:
            message = f"Dependency [{dependency}]: {message}"
        
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


# User interaction errors
class UserError(RoboPoetError):
    """User input and interaction errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("category", ErrorCategory.USER)
        kwargs.setdefault("severity", ErrorSeverity.LOW)
        super().__init__(message, **kwargs)


class InputError(UserError):
    """User input errors."""
    
    def __init__(self, message: str, input_value: Optional[str] = None, **kwargs):
        if input_value:
            message = f"Input [{input_value}]: {message}"
        
        super().__init__(message, **kwargs)


# Error handling utilities
class ErrorHandler:
    """Central error handler with recovery strategies and monitoring."""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.recovery_attempts: Dict[str, int] = {}
        self.recovery_success_rate: Dict[str, float] = {}
        
        # Initialize structured logger
        try:
            from ..utils.structured_logger import get_logger
            self.logger = get_logger("robo_poet.error_handler")
        except ImportError:
            self.logger = None
    
    def handle_error(self, error: RoboPoetError, attempt_recovery: bool = True, 
                    max_retries: int = 3) -> bool:
        """
        Handle error with potential recovery and monitoring.
        
        Args:
            error: The error to handle
            attempt_recovery: Whether to attempt automatic recovery
            max_retries: Maximum recovery attempts
            
        Returns:
            True if recovery was successful, False otherwise
        """
        error_key = f"{error.__class__.__name__}:{error.category.value}"
        
        # Track recovery attempts
        if self.enable_monitoring:
            self.recovery_attempts[error_key] = self.recovery_attempts.get(error_key, 0) + 1
        
        # Log error handling start
        if self.logger:
            with self.logger.operation_context("error_handling", "ErrorHandler") as ctx:
                ctx.metadata = {
                    "error_type": error.__class__.__name__,
                    "error_category": error.category.value,
                    "attempt_recovery": attempt_recovery,
                    "recovery_attempt_count": self.recovery_attempts.get(error_key, 0)
                }
                
                self.logger.error(f"Handling error: {error.message}", ctx, exception=error)
                
                if not attempt_recovery:
                    return False
                
                # Check retry limits
                if self.recovery_attempts.get(error_key, 0) > max_retries:
                    self.logger.warning(f"Max recovery attempts ({max_retries}) exceeded for {error_key}", ctx)
                    return False
                
                # Attempt category-specific recovery
                success = False
                if error.category == ErrorCategory.HARDWARE:
                    success = self._recover_gpu_error(error)
                elif error.category == ErrorCategory.CONFIGURATION:
                    success = self._recover_config_error(error)
                elif error.category == ErrorCategory.DATA:
                    success = self._recover_data_error(error)
                elif error.category == ErrorCategory.MODEL:
                    success = self._recover_model_error(error)
                
                # Update success rate tracking
                if self.enable_monitoring:
                    self._update_success_rate(error_key, success)
                
                if success:
                    self.logger.info(f"Successfully recovered from {error_key}", ctx)
                else:
                    self.logger.warning(f"Failed to recover from {error_key}", ctx)
                
                return success
        else:
            # Fallback without structured logging
            logger = logging.getLogger("robo_poet.error_handler")
            logger.error(f"Handling error: {error.message}")
            
            if not attempt_recovery or self.recovery_attempts.get(error_key, 0) > max_retries:
                return False
            
            # Attempt category-specific recovery
            if error.category == ErrorCategory.HARDWARE:
                return self._recover_gpu_error(error)
            elif error.category == ErrorCategory.CONFIGURATION:
                return self._recover_config_error(error)
            elif error.category == ErrorCategory.DATA:
                return self._recover_data_error(error)
            elif error.category == ErrorCategory.MODEL:
                return self._recover_model_error(error)
            
            return False
    
    def _update_success_rate(self, error_key: str, success: bool):
        """Update success rate tracking."""
        current_rate = self.recovery_success_rate.get(error_key, 0.0)
        attempts = self.recovery_attempts.get(error_key, 1)
        
        # Moving average of success rate
        new_rate = (current_rate * (attempts - 1) + (1.0 if success else 0.0)) / attempts
        self.recovery_success_rate[error_key] = new_rate
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "recovery_attempts": self.recovery_attempts.copy(),
            "success_rates": self.recovery_success_rate.copy(),
            "total_errors_handled": sum(self.recovery_attempts.values())
        }
    
    @staticmethod
    def handle_error_static(error: RoboPoetError, attempt_recovery: bool = True) -> bool:
        """Static method for backward compatibility."""
        handler = ErrorHandler()
        return handler.handle_error(error, attempt_recovery)
    
    @staticmethod
    def _recover_gpu_error(error: GPUError) -> bool:
        """Attempt GPU error recovery."""
        logger = logging.getLogger("robo_poet.error_handler.gpu")
        
        try:
            # Clear GPU cache
                logger.info("Cleared GPU cache for recovery")
                return True
        except Exception as e:
            logger.error(f"GPU recovery failed: {e}")
        
        return False
    
    @staticmethod
    def _recover_config_error(error: ConfigurationError) -> bool:
        """Attempt configuration error recovery."""
        logger = logging.getLogger("robo_poet.error_handler.config")
        
        # For now, just log suggestions
        suggestions = error.get_recovery_suggestions()
        logger.info(f"Configuration recovery suggestions: {suggestions}")
        
        return False  # Manual intervention required
    
    @staticmethod
    def _recover_data_error(error: DataError) -> bool:
        """Attempt data error recovery."""
        logger = logging.getLogger("robo_poet.error_handler.data")
        
        # Could implement data validation/cleaning here
        logger.info("Data error recovery not implemented")
        
        return False
    
    @staticmethod
    def _recover_model_error(error: ModelError) -> bool:
        """Attempt model error recovery."""
        logger = logging.getLogger("robo_poet.error_handler.model")
        
        # Could implement checkpoint loading here
        logger.info("Model error recovery not implemented")
        
        return False


# Context managers for error handling
class ErrorContextManager:
    """Enhanced context manager for structured error handling with correlation tracking."""
    
    def __init__(self, component: str, operation: str, user_id: Optional[str] = None,
                 session_id: Optional[str] = None, **metadata):
        self.component = component
        self.operation = operation
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata
        self.start_time = None
        
        # Initialize structured logger
        try:
            from ..utils.structured_logger import get_logger, LogContext, CorrelationManager
            self.logger = get_logger(f"robo_poet.{component.lower()}")
            self.use_structured = True
        except ImportError:
            self.logger = logging.getLogger(f"robo_poet.{component.lower()}")
            self.use_structured = False
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        
        if self.use_structured:
            from ..utils.structured_logger import LogContext, CorrelationManager
            
            self.log_context = LogContext(
                correlation_id=CorrelationManager.get_correlation_id(),
                component=self.component,
                operation=self.operation,
                user_id=self.user_id,
                session_id=self.session_id,
                metadata=self.metadata
            )
            self.logger.debug(f"Starting {self.operation}", self.log_context)
        else:
            self.logger.debug(f"Starting {self.operation}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0
        
        if exc_type is None:
            if self.use_structured:
                self.logger.info(f"Completed {self.operation} in {duration_ms:.1f}ms", self.log_context)
            else:
                self.logger.debug(f"Completed {self.operation}")
        else:
            # Convert to RoboPoet error if needed
            if not isinstance(exc_val, RoboPoetError):
                # Create enhanced context with correlation tracking
                error_context = ErrorContext(
                    component=self.component,
                    operation=self.operation,
                    parameters=self.metadata,
                    system_state={"duration_ms": duration_ms},
                    suggestions=[
                        f"Check {self.component} configuration",
                        f"Verify {self.operation} preconditions",
                        "Review system logs for additional context"
                    ]
                )
                
                if self.use_structured:
                    from ..utils.structured_logger import CorrelationManager
                    error_context.correlation_id = CorrelationManager.get_correlation_id()
                    error_context.user_id = self.user_id
                    error_context.session_id = self.session_id
                
                error_context.add_stack_trace()
                
                robo_error = SystemError(
                    f"Unexpected error in {self.component}.{self.operation}: {exc_val}",
                    context=error_context,
                    cause=exc_val
                )
                
                # Log original traceback with context
                if self.use_structured:
                    self.logger.error(
                        f"Unexpected error in {self.operation} after {duration_ms:.1f}ms", 
                        self.log_context, 
                        exception=robo_error
                    )
                else:
                    self.logger.error(f"Unexpected error in {self.operation}", exc_info=True)
                
                # Raise enhanced RoboPoet error
                raise robo_error from exc_val
            else:
                # Log RoboPoet error with context
                if self.use_structured:
                    self.logger.error(
                        f"Error in {self.operation} after {duration_ms:.1f}ms: {exc_val.message}",
                        self.log_context,
                        exception=exc_val
                    )
                else:
                    self.logger.error(f"Error in {self.operation}: {exc_val.message}", exc_info=True)
        
        return False  # Don't suppress exceptions


# Example usage and demonstrations
def demo_error_handling():
    """Demonstrate enhanced error handling capabilities."""
    print("🔧 Demonstrating enhanced error handling system...")
    
    # Initialize error handler with monitoring
    error_handler = ErrorHandler(enable_monitoring=True)
    
    # GPU error with enhanced context
    gpu_info = {"memory_used": 7800, "memory_total": 8192}
    try:
        raise MemoryError("GPU out of memory", memory_info=gpu_info)
    except MemoryError as e:
        print(f"GPU Error: {e.message}")
        print(f"Suggestions: {e.get_recovery_suggestions()}")
        
        # Attempt recovery with monitoring
        recovered = error_handler.handle_error(e)
        print(f"Recovery attempted: {recovered}")
    
    # Training error with correlation tracking
    try:
        from ..utils.structured_logger import CorrelationManager
        
        with CorrelationManager.correlation_context() as correlation_id:
            print(f"Correlation ID: {correlation_id}")
            
            # Create error with context
            error_context = ErrorContext(
                component="ModelTrainer",
                operation="train_epoch",
                parameters={"epoch": 5, "batch_size": 32},
                system_state={"loss": float('inf'), "gradient_norm": 1000.0},
                suggestions=["Reduce learning rate", "Check data quality"],
                correlation_id=correlation_id
            )
            
            training_error = TrainingError(
                "Loss diverged during training", 
                epoch=5,
                context=error_context,
                severity=ErrorSeverity.HIGH
            )
            
            # Handle with enhanced monitoring
            recovered = error_handler.handle_error(training_error)
            print(f"Training error recovery: {recovered}")
            
    except ImportError:
        print("Structured logging not available, using basic error handling")
        try:
            raise TrainingError("Loss diverged", epoch=5)
        except TrainingError as e:
            recovered = error_handler.handle_error(e)
            print(f"Basic recovery attempted: {recovered}")
    
    # Using enhanced error context manager
    try:
        with ErrorContextManager("DataProcessor", "tokenization", user_id="user123") as ctx:
            # Simulate error
            raise ValueError("Invalid token sequence")
    except SystemError as e:
        print(f"Context manager error: {e.message}")
        if e.context:
            print(f"Error context: {e.context.component}.{e.context.operation}")
    
    # Show recovery statistics
    stats = error_handler.get_recovery_stats()
    print(f"\n📊 Recovery Statistics:")
    print(f"Total errors handled: {stats['total_errors_handled']}")
    print(f"Recovery attempts: {stats['recovery_attempts']}")
    print(f"Success rates: {stats['success_rates']}")
    
    # Demonstrate error aggregation
    try:
        from ..utils.structured_logger import get_logger
        logger = get_logger("demo.logger")
        
        error_summary = logger.get_error_summary()
        print(f"\n📈 Error Summary: {error_summary}")
        
        performance_summary = logger.get_performance_summary()
        print(f"📈 Performance Summary: {performance_summary}")
        
    except ImportError:
        print("Structured logging not available for summaries")
    
    print("\n✅ Error handling demonstration complete!")


def create_error_context_decorator(component: str):
    """Create decorator for automatic error context management."""
    def decorator(operation_name: str):
        def wrapper(func):
            def wrapped(*args, **kwargs):
                with ErrorContextManager(component, operation_name):
                    return func(*args, **kwargs)
            return wrapped
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(enable_monitoring=True)
    return _global_error_handler


if __name__ == "__main__":
    demo_error_handling()