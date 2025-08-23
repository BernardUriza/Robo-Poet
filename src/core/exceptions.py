"""
Comprehensive exception hierarchy for Robo-Poet.

Provides structured error handling with contextual information,
recovery suggestions, and proper logging integration.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "component": self.component,
            "operation": self.operation,
            "parameters": self.parameters,
            "system_state": self.system_state,
            "suggestions": self.suggestions
        }


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
        """Log the error with appropriate level."""
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
            logger.critical(f"CRITICAL ERROR: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {self.message}", extra=log_data)
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
    """Central error handler with recovery strategies."""
    
    @staticmethod
    def handle_error(error: RoboPoetError, attempt_recovery: bool = True) -> bool:
        """
        Handle error with potential recovery.
        
        Args:
            error: The error to handle
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger = logging.getLogger("robo_poet.error_handler")
        
        logger.error(f"Handling error: {error.message}")
        
        if not attempt_recovery:
            return False
        
        # Attempt category-specific recovery
        if error.category == ErrorCategory.HARDWARE:
            return ErrorHandler._recover_gpu_error(error)
        elif error.category == ErrorCategory.CONFIGURATION:
            return ErrorHandler._recover_config_error(error)
        elif error.category == ErrorCategory.DATA:
            return ErrorHandler._recover_data_error(error)
        elif error.category == ErrorCategory.MODEL:
            return ErrorHandler._recover_model_error(error)
        
        return False
    
    @staticmethod
    def _recover_gpu_error(error: GPUError) -> bool:
        """Attempt GPU error recovery."""
        logger = logging.getLogger("robo_poet.error_handler.gpu")
        
        try:
            # Clear GPU cache
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
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
    """Context manager for structured error handling."""
    
    def __init__(self, component: str, operation: str):
        self.component = component
        self.operation = operation
        self.logger = logging.getLogger(f"robo_poet.{component.lower()}")
    
    def __enter__(self):
        self.logger.debug(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.debug(f"Completed {self.operation}")
        else:
            # Convert to RoboPoet error if needed
            if not isinstance(exc_val, RoboPoetError):
                robo_error = SystemError(
                    f"Unexpected error in {self.component}: {exc_val}",
                    cause=exc_val
                )
                # Log original traceback
                self.logger.error(f"Unexpected error in {self.operation}", exc_info=True)
                # Raise RoboPoet error
                raise robo_error from exc_val
        
        return False  # Don't suppress exceptions


# Example usage
def demo_error_handling():
    """Demonstrate error handling capabilities."""
    
    # GPU error with context
    gpu_info = {"memory_used": 7800, "memory_total": 8192}
    try:
        raise MemoryError("GPU out of memory", memory_info=gpu_info)
    except MemoryError as e:
        print(f"GPU Error: {e.message}")
        print(f"Suggestions: {e.get_recovery_suggestions()}")
    
    # Training error with recovery
    try:
        raise TrainingError("Loss diverged", epoch=5)
    except TrainingError as e:
        recovered = ErrorHandler.handle_error(e)
        print(f"Recovery attempted: {recovered}")
    
    # Using error context
    try:
        with ErrorContext("DataProcessor", "tokenization"):
            # Simulate error
            raise ValueError("Invalid token")
    except SystemError as e:
        print(f"Context error: {e.message}")


if __name__ == "__main__":
    demo_error_handling()