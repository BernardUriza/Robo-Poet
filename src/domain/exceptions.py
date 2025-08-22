"""
Domain-specific exceptions.

Custom exceptions that represent business rule violations.
"""


class RoboPoetDomainException(Exception):
    """Base exception for all domain-related errors."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__


class CorpusValidationError(RoboPoetDomainException):
    """Raised when corpus validation fails."""
    pass


class ModelConfigurationError(RoboPoetDomainException):
    """Raised when model configuration is invalid."""
    pass


class TrainingStateError(RoboPoetDomainException):
    """Raised when training operation is invalid for current state."""
    pass


class GenerationError(RoboPoetDomainException):
    """Raised when text generation fails."""
    pass


class ModelNotReadyError(RoboPoetDomainException):
    """Raised when attempting to use model that's not ready."""
    pass


class ResourceNotFoundError(RoboPoetDomainException):
    """Raised when required resource is not found."""
    pass


class InvalidParametersError(RoboPoetDomainException):
    """Raised when parameters are invalid."""
    pass


class BusinessRuleViolationError(RoboPoetDomainException):
    """Raised when a business rule is violated."""
    pass


class ConcurrencyError(RoboPoetDomainException):
    """Raised when concurrent access causes conflicts."""
    pass