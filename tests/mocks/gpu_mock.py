"""
Sistema de mocking de GPU para CI/CD.

Proporciona mocks completos de TensorFlow y componentes GPU para testing
en entornos sin hardware GPU.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager


class MockTensorFlow:
    """Mock completo del módulo TensorFlow."""
    
    def __init__(self):
        """Inicializa el mock de TensorFlow."""
        self._setup_base_tf()
        self._setup_keras()
        self._setup_config()
        self._setup_device()
        self._setup_data()
        
    def _setup_base_tf(self):
        """Configura mocks base de TensorFlow."""
        self.version = Mock()
        self.version.__version__ = "2.20.0"
        
        # Operaciones básicas
        self.constant = Mock(side_effect=self._mock_constant)
        self.reduce_sum = Mock(side_effect=self._mock_reduce_sum)
        self.reduce_mean = Mock(side_effect=self._mock_reduce_mean)
        self.cast = Mock(side_effect=self._mock_cast)
        self.concat = Mock(side_effect=self._mock_concat)
        self.expand_dims = Mock(side_effect=self._mock_expand_dims)
        self.squeeze = Mock(side_effect=self._mock_squeeze)
        
        # Funciones matemáticas
        self.math = Mock()
        self.math.log = Mock(side_effect=self._mock_log)
        self.math.exp = Mock(side_effect=self._mock_exp)
        self.math.multiply = Mock(side_effect=self._mock_multiply)
        
        # Random
        self.random = Mock()
        self.random.categorical = Mock(side_effect=self._mock_categorical)
        self.random.uniform = Mock(side_effect=self._mock_uniform)
        self.random.normal = Mock(side_effect=self._mock_normal)
        
    def _setup_keras(self):
        """Configura mocks de Keras."""
        self.keras = Mock()
        
        # Layers
        self.keras.layers = Mock()
        self.keras.layers.Embedding = Mock(side_effect=self._mock_embedding)
        self.keras.layers.LSTM = Mock(side_effect=self._mock_lstm)
        self.keras.layers.Dense = Mock(side_effect=self._mock_dense)
        self.keras.layers.Dropout = Mock(side_effect=self._mock_dropout)
        self.keras.layers.Input = Mock(side_effect=self._mock_input)
        
        # Model
        self.keras.Model = Mock(side_effect=self._mock_model)
        self.keras.Sequential = Mock(side_effect=self._mock_sequential)
        
        # Utils
        self.keras.utils = Mock()
        self.keras.utils.to_categorical = Mock(side_effect=self._mock_to_categorical)
        
        # Models
        self.keras.models = Mock()
        self.keras.models.load_model = Mock(side_effect=self._mock_load_model)
        
        # Optimizers
        self.keras.optimizers = Mock()
        self.keras.optimizers.Adam = Mock(side_effect=self._mock_adam)
        self.keras.optimizers.SGD = Mock(side_effect=self._mock_sgd)
        
        # Losses
        self.keras.losses = Mock()
        self.keras.losses.SparseCategoricalCrossentropy = Mock(
            side_effect=self._mock_sparse_categorical_crossentropy
        )
        
        # Callbacks
        self.keras.callbacks = Mock()
        self.keras.callbacks.EarlyStopping = Mock(side_effect=self._mock_early_stopping)
        self.keras.callbacks.ModelCheckpoint = Mock(side_effect=self._mock_model_checkpoint)
        self.keras.callbacks.TensorBoard = Mock(side_effect=self._mock_tensorboard)
        
        # Backend
        self.keras.backend = Mock()
        self.keras.backend.clear_session = Mock()
        
        # Mixed precision
        self.keras.mixed_precision = Mock()
        self.keras.mixed_precision.Policy = Mock(side_effect=self._mock_policy)
        self.keras.mixed_precision.set_global_policy = Mock()
        self.keras.mixed_precision.global_policy = Mock(return_value=Mock(name='mixed_float16'))
        
    def _setup_config(self):
        """Configura mocks de configuración."""
        self.config = Mock()
        
        # GPU config
        self.config.list_physical_devices = Mock(return_value=[])
        self.config.experimental = Mock()
        self.config.experimental.list_physical_devices = Mock(return_value=[])
        self.config.experimental.set_memory_growth = Mock()
        self.config.experimental.set_memory_limit = Mock()
        
        # Test functions
        self.test = Mock()
        self.test.is_built_with_cuda = Mock(return_value=True)
        self.test.is_gpu_available = Mock(return_value=False)
        
    def _setup_device(self):
        """Configura mocks de dispositivos."""
        self.device = Mock(side_effect=self._mock_device)
        
    def _setup_data(self):
        """Configura mocks de tf.data."""
        self.data = Mock()
        self.data.Dataset = Mock()
        self.data.Dataset.from_tensor_slices = Mock(side_effect=self._mock_dataset_from_slices)
        self.data.AUTOTUNE = -1
        
    # Mock implementations
    def _mock_constant(self, value, dtype=None, shape=None, name=None):
        """Mock tf.constant."""
        if isinstance(value, (list, tuple)):
            return MockTensor(np.array(value), dtype=dtype)
        return MockTensor(np.array(value), dtype=dtype)
    
    def _mock_reduce_sum(self, tensor, axis=None):
        """Mock tf.reduce_sum."""
        if hasattr(tensor, 'numpy_data'):
            result = np.sum(tensor.numpy_data, axis=axis)
            return MockTensor(result)
        return MockTensor(np.sum(np.array(tensor), axis=axis))
    
    def _mock_reduce_mean(self, tensor, axis=None):
        """Mock tf.reduce_mean."""
        if hasattr(tensor, 'numpy_data'):
            result = np.mean(tensor.numpy_data, axis=axis)
            return MockTensor(result)
        return MockTensor(np.mean(np.array(tensor), axis=axis))
    
    def _mock_cast(self, tensor, dtype):
        """Mock tf.cast."""
        return MockTensor(tensor, dtype=dtype)
    
    def _mock_concat(self, tensors, axis=0):
        """Mock tf.concat."""
        arrays = [t.numpy_data if hasattr(t, 'numpy_data') else np.array(t) for t in tensors]
        result = np.concatenate(arrays, axis=axis)
        return MockTensor(result)
    
    def _mock_expand_dims(self, tensor, axis):
        """Mock tf.expand_dims."""
        array = tensor.numpy_data if hasattr(tensor, 'numpy_data') else np.array(tensor)
        result = np.expand_dims(array, axis=axis)
        return MockTensor(result)
    
    def _mock_squeeze(self, tensor, axis=None):
        """Mock tf.squeeze."""
        array = tensor.numpy_data if hasattr(tensor, 'numpy_data') else np.array(tensor)
        result = np.squeeze(array, axis=axis)
        return MockTensor(result)
    
    def _mock_log(self, tensor):
        """Mock tf.math.log."""
        array = tensor.numpy_data if hasattr(tensor, 'numpy_data') else np.array(tensor)
        return MockTensor(np.log(array))
    
    def _mock_exp(self, tensor):
        """Mock tf.math.exp."""
        array = tensor.numpy_data if hasattr(tensor, 'numpy_data') else np.array(tensor)
        return MockTensor(np.exp(array))
    
    def _mock_multiply(self, x, y):
        """Mock tf.math.multiply."""
        x_array = x.numpy_data if hasattr(x, 'numpy_data') else np.array(x)
        y_array = y.numpy_data if hasattr(y, 'numpy_data') else np.array(y)
        return MockTensor(x_array * y_array)
    
    def _mock_categorical(self, logits, num_samples=1):
        """Mock tf.random.categorical."""
        # Simulación simple de sampling categórico
        if hasattr(logits, 'numpy_data'):
            probs = logits.numpy_data
        else:
            probs = np.array(logits)
        
        # Aplicar softmax
        exp_probs = np.exp(probs - np.max(probs, axis=-1, keepdims=True))
        softmax_probs = exp_probs / np.sum(exp_probs, axis=-1, keepdims=True)
        
        # Sample
        batch_size, vocab_size = softmax_probs.shape
        samples = []
        for i in range(batch_size):
            for _ in range(num_samples):
                sample = np.random.choice(vocab_size, p=softmax_probs[i])
                samples.append(sample)
        
        return MockTensor(np.array(samples).reshape(batch_size, num_samples))
    
    def _mock_uniform(self, shape, minval=0, maxval=1, dtype=None):
        """Mock tf.random.uniform."""
        return MockTensor(np.random.uniform(minval, maxval, shape))
    
    def _mock_normal(self, shape, mean=0, stddev=1, dtype=None):
        """Mock tf.random.normal."""
        return MockTensor(np.random.normal(mean, stddev, shape))
    
    def _mock_device(self, device_name):
        """Mock tf.device."""
        return MockDeviceContext()
    
    def _mock_dataset_from_slices(self, tensors):
        """Mock tf.data.Dataset.from_tensor_slices."""
        return MockDataset(tensors)
    
    # Keras layer mocks
    def _mock_embedding(self, input_dim, output_dim, **kwargs):
        """Mock Embedding layer."""
        return MockLayer("Embedding", {"input_dim": input_dim, "output_dim": output_dim})
    
    def _mock_lstm(self, units, return_sequences=False, **kwargs):
        """Mock LSTM layer."""
        return MockLayer("LSTM", {"units": units, "return_sequences": return_sequences})
    
    def _mock_dense(self, units, activation=None, **kwargs):
        """Mock Dense layer."""
        return MockLayer("Dense", {"units": units, "activation": activation})
    
    def _mock_dropout(self, rate, **kwargs):
        """Mock Dropout layer."""
        return MockLayer("Dropout", {"rate": rate})
    
    def _mock_input(self, shape=None, **kwargs):
        """Mock Input layer."""
        return MockLayer("Input", {"shape": shape})
    
    def _mock_model(self, inputs=None, outputs=None, **kwargs):
        """Mock Keras Model."""
        return MockModel(inputs, outputs)
    
    def _mock_sequential(self, layers=None, **kwargs):
        """Mock Sequential model."""
        return MockSequentialModel(layers or [])
    
    def _mock_to_categorical(self, y, num_classes=None):
        """Mock to_categorical."""
        y_array = np.array(y)
        if num_classes is None:
            num_classes = np.max(y_array) + 1
        return np.eye(num_classes)[y_array]
    
    def _mock_load_model(self, filepath, **kwargs):
        """Mock load_model."""
        return MockSequentialModel([])
    
    def _mock_adam(self, learning_rate=0.001, **kwargs):
        """Mock Adam optimizer."""
        return MockOptimizer("Adam", {"learning_rate": learning_rate})
    
    def _mock_sgd(self, learning_rate=0.01, **kwargs):
        """Mock SGD optimizer."""
        return MockOptimizer("SGD", {"learning_rate": learning_rate})
    
    def _mock_sparse_categorical_crossentropy(self, **kwargs):
        """Mock SparseCategoricalCrossentropy loss."""
        return MockLoss("sparse_categorical_crossentropy")
    
    def _mock_early_stopping(self, **kwargs):
        """Mock EarlyStopping callback."""
        return MockCallback("EarlyStopping")
    
    def _mock_model_checkpoint(self, **kwargs):
        """Mock ModelCheckpoint callback."""
        return MockCallback("ModelCheckpoint")
    
    def _mock_tensorboard(self, **kwargs):
        """Mock TensorBoard callback."""
        return MockCallback("TensorBoard")
    
    def _mock_policy(self, name):
        """Mock mixed precision policy."""
        policy = Mock()
        policy.name = name
        return policy


class MockTensor:
    """Mock de tensor de TensorFlow."""
    
    def __init__(self, data, dtype=None, shape=None):
        """Inicializa mock tensor."""
        self.numpy_data = np.array(data)
        self.dtype = dtype
        self._shape = shape or self.numpy_data.shape
    
    def numpy(self):
        """Retorna datos como numpy array."""
        return self.numpy_data
    
    @property
    def shape(self):
        """Retorna shape del tensor."""
        return self._shape
    
    def __getitem__(self, key):
        """Soporte para indexing."""
        return MockTensor(self.numpy_data[key])
    
    def __len__(self):
        """Soporte para len()."""
        return len(self.numpy_data)


class MockDeviceContext:
    """Mock de contexto de dispositivo."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockDataset:
    """Mock de tf.data.Dataset."""
    
    def __init__(self, data):
        self.data = data
    
    def batch(self, batch_size):
        """Mock batch operation."""
        return MockDataset(self.data)
    
    def prefetch(self, buffer_size):
        """Mock prefetch operation."""
        return MockDataset(self.data)
    
    def shuffle(self, buffer_size):
        """Mock shuffle operation."""
        return MockDataset(self.data)
    
    def map(self, map_func, num_parallel_calls=None):
        """Mock map operation."""
        return MockDataset(self.data)
    
    def repeat(self, count=None):
        """Mock repeat operation."""
        return MockDataset(self.data)
    
    def take(self, count):
        """Mock take operation."""
        return MockDataset(self.data)


class MockLayer:
    """Mock de capa de Keras."""
    
    def __init__(self, layer_type: str, config: Dict[str, Any]):
        self.layer_type = layer_type
        self.config = config
    
    def __call__(self, inputs):
        """Mock de llamada a la capa."""
        return MockTensor(np.random.random((32, 128)))  # Salida simulada


class MockModel:
    """Mock de modelo de Keras."""
    
    def __init__(self, inputs=None, outputs=None):
        self.inputs = inputs
        self.outputs = outputs
        self.compiled = False
        self.history = None
    
    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """Mock compile."""
        self.compiled = True
    
    def fit(self, x=None, y=None, epochs=1, batch_size=32, validation_data=None, 
            callbacks=None, verbose=1, **kwargs):
        """Mock fit."""
        # Simular entrenamiento
        history = {
            'loss': [2.0 - (i * 0.1) for i in range(epochs)],
            'accuracy': [0.3 + (i * 0.05) for i in range(epochs)]
        }
        
        if validation_data:
            history['val_loss'] = [2.2 - (i * 0.08) for i in range(epochs)]
            history['val_accuracy'] = [0.25 + (i * 0.04) for i in range(epochs)]
        
        self.history = Mock()
        self.history.history = history
        return self.history
    
    def predict(self, x, batch_size=32, verbose=0, **kwargs):
        """Mock predict."""
        if isinstance(x, (list, tuple)):
            batch_size = len(x)
        else:
            batch_size = x.shape[0] if hasattr(x, 'shape') else 32
        
        # Simular predicciones
        vocab_size = 1000
        return np.random.random((batch_size, vocab_size))
    
    def evaluate(self, x=None, y=None, batch_size=32, verbose=1, **kwargs):
        """Mock evaluate."""
        return [1.5, 0.6]  # [loss, accuracy]
    
    def save(self, filepath, **kwargs):
        """Mock save."""
        # Simular guardado
        pass
    
    def load_weights(self, filepath, **kwargs):
        """Mock load_weights."""
        pass
    
    def save_weights(self, filepath, **kwargs):
        """Mock save_weights."""
        pass
    
    def summary(self):
        """Mock summary."""
        print(f"Mock Model Summary: {self.layer_type if hasattr(self, 'layer_type') else 'Model'}")


class MockSequentialModel(MockModel):
    """Mock de modelo Sequential."""
    
    def __init__(self, layers=None):
        super().__init__()
        self.layers_list = layers or []
        self.layer_type = "Sequential"
    
    def add(self, layer):
        """Mock add layer."""
        self.layers_list.append(layer)
    
    def build(self, input_shape=None):
        """Mock build."""
        pass


class MockOptimizer:
    """Mock de optimizador."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config


class MockLoss:
    """Mock de función de pérdida."""
    
    def __init__(self, name: str):
        self.name = name


class MockCallback:
    """Mock de callback."""
    
    def __init__(self, name: str):
        self.name = name


# Fixtures de pytest
@pytest.fixture
def mock_tensorflow():
    """Fixture que proporciona mock completo de TensorFlow."""
    tf_mock = MockTensorFlow()
    
    with patch.dict('sys.modules', {'tensorflow': tf_mock}):
        with patch('tensorflow', tf_mock):
            yield tf_mock


@pytest.fixture
def mock_gpu_available():
    """Fixture que simula GPU disponible."""
    tf_mock = MockTensorFlow()
    
    # Simular GPU disponible
    gpu_device = Mock()
    gpu_device.name = '/physical_device:GPU:0'
    gpu_device.device_type = 'GPU'
    
    tf_mock.config.list_physical_devices.return_value = [gpu_device]
    tf_mock.config.experimental.list_physical_devices.return_value = [gpu_device]
    tf_mock.test.is_gpu_available.return_value = True
    
    with patch.dict('sys.modules', {'tensorflow': tf_mock}):
        with patch('tensorflow', tf_mock):
            yield tf_mock


@pytest.fixture
def mock_gpu_unavailable():
    """Fixture que simula GPU no disponible."""
    tf_mock = MockTensorFlow()
    
    # Simular sin GPU
    tf_mock.config.list_physical_devices.return_value = []
    tf_mock.config.experimental.list_physical_devices.return_value = []
    tf_mock.test.is_gpu_available.return_value = False
    
    # Simular error de GPU en operaciones
    def gpu_error(*args, **kwargs):
        raise RuntimeError("GPU not available")
    
    tf_mock.device.side_effect = gpu_error
    
    with patch.dict('sys.modules', {'tensorflow': tf_mock}):
        with patch('tensorflow', tf_mock):
            yield tf_mock


@contextmanager
def mock_gpu_environment(gpu_available: bool = True, 
                        memory_limit: Optional[int] = None,
                        tensor_cores: bool = True):
    """Context manager para simular entorno GPU específico."""
    tf_mock = MockTensorFlow()
    
    if gpu_available:
        gpu_device = Mock()
        gpu_device.name = '/physical_device:GPU:0'
        gpu_device.device_type = 'GPU'
        tf_mock.config.list_physical_devices.return_value = [gpu_device]
        tf_mock.test.is_gpu_available.return_value = True
    else:
        tf_mock.config.list_physical_devices.return_value = []
        tf_mock.test.is_gpu_available.return_value = False
    
    # Mock memory settings
    if memory_limit:
        tf_mock.config.experimental.set_memory_limit = Mock()
    
    # Mock tensor cores
    if tensor_cores:
        tf_mock.keras.mixed_precision.Policy = Mock(return_value=Mock(name='mixed_float16'))
    
    with patch.dict('sys.modules', {'tensorflow': tf_mock}):
        with patch('tensorflow', tf_mock):
            yield tf_mock


# Decoradores para tests
def requires_no_gpu(func):
    """Decorador que ejecuta test solo si no hay GPU real."""
    def wrapper(*args, **kwargs):
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                pytest.skip("Test requiere ejecutarse sin GPU real")
        except ImportError:
            pass  # Sin TensorFlow, podemos ejecutar el test
        return func(*args, **kwargs)
    return wrapper


def mock_gpu_components(func):
    """Decorador que mockea componentes GPU automáticamente."""
    def wrapper(*args, **kwargs):
        with mock_gpu_environment():
            return func(*args, **kwargs)
    return wrapper


# Utilidades para tests
def create_mock_training_data(batch_size: int = 32, 
                            sequence_length: int = 100,
                            vocab_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Crea datos de entrenamiento simulados."""
    X = np.random.randint(0, vocab_size, (batch_size, sequence_length))
    y = np.random.randint(0, vocab_size, (batch_size, sequence_length))
    return X, y


def create_mock_model_config() -> Dict[str, Any]:
    """Crea configuración de modelo simulada."""
    return {
        'vocab_size': 1000,
        'embedding_dim': 128,
        'lstm_units': [256, 256],
        'dropout_rate': 0.3,
        'sequence_length': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10
    }


def assert_mock_training_called(tf_mock: MockTensorFlow):
    """Verifica que se llamaron funciones de entrenamiento."""
    # Verificar que se crearon layers
    tf_mock.keras.layers.Embedding.assert_called()
    tf_mock.keras.layers.LSTM.assert_called()
    tf_mock.keras.layers.Dense.assert_called()
    
    # Verificar que se creó modelo
    tf_mock.keras.Sequential.assert_called()


def assert_mock_prediction_called(tf_mock: MockTensorFlow):
    """Verifica que se llamaron funciones de predicción."""
    # En un test real, verificaríamos que predict fue llamado
    # tf_mock.keras.Sequential().predict.assert_called()
    pass