"""
Tests unitarios para el sistema de excepciones del core.

Prueba el sistema de excepciones estructuradas y manejo de errores.
"""

import pytest
import logging
from unittest.mock import Mock, patch
from io import StringIO

# Setup sys.path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.exceptions import (
    RoboPoetError, ErrorSeverity, ErrorCategory, ErrorContext,
    GPUError, CUDAError, MemoryError, TensorCoreError,
    ConfigurationError, ValidationError,
    DataError, CorpusError, TokenizationError, AugmentationError,
    ModelError, TrainingError, ConvergenceError, ArchitectureError,
    GenerationError, SamplingError, QualityError,
    SystemError, ResourceError, DependencyError,
    UserError, InputError,
    ErrorHandler
)


class TestErrorEnums:
    """Tests para los enums de error."""
    
    def test_error_severity_values(self):
        """Test valores de ErrorSeverity."""
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.CRITICAL == "critical"
    
    def test_error_category_values(self):
        """Test valores de ErrorCategory."""
        assert ErrorCategory.HARDWARE == "hardware"
        assert ErrorCategory.CONFIGURATION == "configuration"
        assert ErrorCategory.DATA == "data"
        assert ErrorCategory.MODEL == "model"
        assert ErrorCategory.GENERATION == "generation"
        assert ErrorCategory.SYSTEM == "system"
        assert ErrorCategory.USER == "user"


class TestErrorContext:
    """Tests para ErrorContext."""
    
    def test_create_error_context(self):
        """Test crear contexto de error."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            parameters={"param1": "value1"},
            system_state={"gpu_memory": "8GB"},
            suggestions=["Check GPU", "Restart system"]
        )
        
        assert context.component == "TestComponent"
        assert context.operation == "test_operation"
        assert context.parameters["param1"] == "value1"
        assert context.system_state["gpu_memory"] == "8GB"
        assert "Check GPU" in context.suggestions
    
    def test_error_context_to_dict(self):
        """Test conversión de contexto a diccionario."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            parameters={"param1": "value1"},
            system_state={"state1": "active"},
            suggestions=["suggestion1"]
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["component"] == "TestComponent"
        assert context_dict["operation"] == "test_operation"
        assert context_dict["parameters"]["param1"] == "value1"
        assert context_dict["system_state"]["state1"] == "active"
        assert context_dict["suggestions"] == ["suggestion1"]


class TestRoboPoetError:
    """Tests para la clase base RoboPoetError."""
    
    def test_create_basic_error(self):
        """Test crear error básico."""
        error = RoboPoetError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.category == ErrorCategory.SYSTEM
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context is None
        assert error.cause is None
    
    def test_create_error_with_all_parameters(self):
        """Test crear error con todos los parámetros."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            parameters={},
            system_state={},
            suggestions=["Test suggestion"]
        )
        
        cause = ValueError("Original error")
        
        error = RoboPoetError(
            message="Detailed error message",
            category=ErrorCategory.HARDWARE,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            cause=cause
        )
        
        assert error.message == "Detailed error message"
        assert error.category == ErrorCategory.HARDWARE
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context == context
        assert error.cause == cause
    
    def test_error_logging(self):
        """Test logging automático de errores."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            error = RoboPoetError(
                "Test logging error",
                severity=ErrorSeverity.CRITICAL
            )
            
            # Verificar que se llamó al logger
            mock_get_logger.assert_called_with("robo_poet.system")
            mock_logger.critical.assert_called_once()
    
    def test_get_recovery_suggestions(self):
        """Test obtener sugerencias de recuperación."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            parameters={},
            system_state={},
            suggestions=["Custom suggestion"]
        )
        
        # Test error de hardware
        gpu_error = RoboPoetError(
            "GPU error",
            category=ErrorCategory.HARDWARE,
            context=context
        )
        
        suggestions = gpu_error.get_recovery_suggestions()
        assert "Custom suggestion" in suggestions
        assert "Check GPU drivers and CUDA installation" in suggestions
        
        # Test error de configuración
        config_error = RoboPoetError(
            "Config error",
            category=ErrorCategory.CONFIGURATION
        )
        
        suggestions = config_error.get_recovery_suggestions()
        assert "Review configuration file syntax" in suggestions


class TestSpecificErrors:
    """Tests para errores específicos."""
    
    def test_gpu_error(self):
        """Test GPUError."""
        gpu_info = {"name": "RTX 2000 Ada", "memory": "8GB"}
        
        error = GPUError("GPU failure", gpu_info=gpu_info)
        
        assert error.category == ErrorCategory.HARDWARE
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.system_state == gpu_info
        assert "Check nvidia-smi output" in error.context.suggestions
    
    def test_cuda_error(self):
        """Test CUDAError."""
        error = CUDAError("CUDA runtime error", cuda_version="12.2")
        
        assert "CUDA 12.2" in error.message
        assert error.severity == ErrorSeverity.CRITICAL
    
    def test_memory_error(self):
        """Test MemoryError."""
        memory_info = {"used": "7.5GB", "total": "8GB"}
        
        error = MemoryError("Out of memory", memory_info=memory_info)
        
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.system_state == memory_info
        assert "Reduce batch size" in error.context.suggestions
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config", config_section="gpu")
        
        assert "Configuration [gpu]" in error.message
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.MEDIUM
    
    def test_validation_error(self):
        """Test ValidationError."""
        invalid_values = {"learning_rate": -0.1, "batch_size": 0}
        
        error = ValidationError("Validation failed", invalid_values=invalid_values)
        
        assert error.context.parameters == invalid_values
        assert "Check parameter ranges and types" in error.context.suggestions
    
    def test_corpus_error(self):
        """Test CorpusError."""
        error = CorpusError("File not found", file_path="/test/file.txt")
        
        assert "Corpus [/test/file.txt]" in error.message
        assert error.category == ErrorCategory.DATA
    
    def test_training_error(self):
        """Test TrainingError."""
        error = TrainingError("Training failed", epoch=5, batch=100)
        
        assert "Training [Epoch 5]" in error.message
        assert "[Batch 100]" in error.message
        assert error.category == ErrorCategory.MODEL
    
    def test_convergence_error(self):
        """Test ConvergenceError."""
        metrics = {"loss": 2.5, "accuracy": 0.2}
        
        error = ConvergenceError("Model not converging", metrics=metrics)
        
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.system_state["metrics"] == metrics
        assert "Adjust learning rate" in error.context.suggestions
    
    def test_sampling_error(self):
        """Test SamplingError."""
        error = SamplingError("Sampling failed", sampler_type="nucleus")
        
        assert "Sampler [nucleus]" in error.message
        assert error.category == ErrorCategory.GENERATION
    
    def test_resource_error(self):
        """Test ResourceError."""
        usage = {"cpu": "95%", "memory": "7.8GB"}
        
        error = ResourceError("High resource usage", "memory", current_usage=usage)
        
        assert "Resource [memory]" in error.message
        assert error.context.system_state == usage
        assert "Reduce memory usage" in error.context.suggestions
    
    def test_dependency_error(self):
        """Test DependencyError."""
        error = DependencyError("Missing dependency", dependency="tensorflow")
        
        assert "Dependency [tensorflow]" in error.message
        assert error.severity == ErrorSeverity.CRITICAL
    
    def test_input_error(self):
        """Test InputError."""
        error = InputError("Invalid input", input_value="invalid_value")
        
        assert "Input [invalid_value]" in error.message
        assert error.category == ErrorCategory.USER
        assert error.severity == ErrorSeverity.LOW


class TestErrorHandler:
    """Tests para ErrorHandler."""
    
    def test_handle_error_no_recovery(self):
        """Test manejo de error sin recuperación."""
        error = RoboPoetError("Test error")
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = ErrorHandler.handle_error(error, attempt_recovery=False)
            
            assert result is False
            mock_logger.error.assert_called_once()
    
    def test_handle_gpu_error_recovery(self):
        """Test recuperación de error de GPU."""
        gpu_error = GPUError("GPU error")
        
        with patch('tensorflow.config.list_physical_devices') as mock_list_devices:
            with patch('tensorflow.keras.backend.clear_session') as mock_clear:
                mock_list_devices.return_value = [Mock()]  # Simular GPU disponible
                
                result = ErrorHandler.handle_error(gpu_error, attempt_recovery=True)
                
                assert result is True
                mock_clear.assert_called_once()
    
    def test_handle_gpu_error_no_tensorflow(self):
        """Test recuperación de error de GPU sin TensorFlow."""
        gpu_error = GPUError("GPU error")
        
        with patch('tensorflow.config.list_physical_devices', side_effect=ImportError):
            result = ErrorHandler.handle_error(gpu_error, attempt_recovery=True)
            
            assert result is False
    
    def test_handle_config_error_recovery(self):
        """Test recuperación de error de configuración."""
        config_error = ConfigurationError("Config error")
        
        result = ErrorHandler.handle_error(config_error, attempt_recovery=True)
        
        # Los errores de configuración requieren intervención manual
        assert result is False
    
    def test_handle_unknown_category_error(self):
        """Test manejo de error de categoría desconocida."""
        # Crear error con categoría no manejada específicamente
        unknown_error = RoboPoetError("Unknown error", category=ErrorCategory.USER)
        
        result = ErrorHandler.handle_error(unknown_error, attempt_recovery=True)
        
        # Categorías no manejadas específicamente devuelven False
        assert result is False


class TestErrorContextManager:
    """Tests para el context manager de errores."""
    
    def test_error_context_manager_success(self):
        """Test context manager con operación exitosa."""
        from core.exceptions import ErrorContextManager
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with ErrorContextManager("TestComponent", "test_operation"):
                pass  # Operación exitosa
            
            # Verificar logging de inicio y finalización
            assert mock_logger.debug.call_count == 2
    
    def test_error_context_manager_with_robo_poet_error(self):
        """Test context manager con RoboPoetError."""
        from core.exceptions import ErrorContextManager
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            original_error = ValidationError("Test validation error")
            
            with pytest.raises(ValidationError):
                with ErrorContextManager("TestComponent", "test_operation"):
                    raise original_error
            
            # No debe convertir RoboPoetError
            mock_logger.debug.assert_called()
    
    def test_error_context_manager_with_system_error(self):
        """Test context manager con error de sistema."""
        from core.exceptions import ErrorContextManager
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(SystemError):
                with ErrorContextManager("TestComponent", "test_operation"):
                    raise ValueError("System error")
            
            # Debe loggear el error original
            mock_logger.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])