"""
Configuración compartida para tests de pytest.

Proporciona fixtures y configuración común para toda la suite de tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import sys
import os

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_PATH))

# Import mocks
from tests.mocks.gpu_mock import MockTensorFlow


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configuración global para todos los tests."""
    # Configurar variables de entorno para tests
    os.environ['ROBO_POET_LOG_LEVEL'] = 'DEBUG'
    os.environ['ROBO_POET_DEBUG'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forzar CPU para tests
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir logs de TensorFlow
    
    yield
    
    # Cleanup
    for key in ['ROBO_POET_LOG_LEVEL', 'ROBO_POET_DEBUG', 'CUDA_VISIBLE_DEVICES']:
        os.environ.pop(key, None)


@pytest.fixture
def temp_dir():
    """Proporciona directorio temporal para tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_corpus_file(temp_dir):
    """Archivo de corpus de ejemplo para tests."""
    content = """
    The art of power requires understanding human nature and psychology.
    Knowledge without action is useless, but action without knowledge is dangerous.
    In the realm of artificial intelligence, we seek to understand patterns in data.
    These patterns reveal the underlying structure of language and communication.
    The greatest innovations come from combining different fields of knowledge.
    Like a poet who crafts verses, an AI must learn the rhythm of human expression.
    Every word carries meaning beyond its literal interpretation in context.
    Context shapes understanding, and understanding shapes our perception of reality.
    Machine learning models learn from examples to generalize to new situations.
    The challenge is to create systems that can understand nuance and subtlety.
    """
    
    corpus_file = temp_dir / "test_corpus.txt"
    corpus_file.write_text(content.strip())
    return corpus_file


@pytest.fixture
def sample_model_config():
    """Configuración de modelo de ejemplo para tests."""
    return {
        'vocab_size': 1000,
        'embedding_dim': 64,
        'lstm_units': [128, 128],
        'dropout_rate': 0.2,
        'sequence_length': 50,
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 2
    }


@pytest.fixture
def mock_tensorflow_session():
    """Sesión de TensorFlow mockeada para tests."""
    tf_mock = MockTensorFlow()
    
    with pytest.MonkeyPatch().context() as m:
        m.setattr('sys.modules.tensorflow', tf_mock)
        yield tf_mock


@pytest.fixture
def mock_repositories():
    """Repositorios mockeados para tests de servicios."""
    corpus_repo = Mock()
    model_repo = Mock()
    event_repo = Mock()
    
    return {
        'corpus_repository': corpus_repo,
        'model_repository': model_repo,
        'event_repository': event_repo
    }


@pytest.fixture
def mock_unit_of_work(mock_repositories):
    """Unit of Work mockeado."""
    from unittest.mock import AsyncMock
    
    uow = Mock()
    uow.corpus_repository = mock_repositories['corpus_repository']
    uow.model_repository = mock_repositories['model_repository'] 
    uow.event_repository = mock_repositories['event_repository']
    
    uow.commit = AsyncMock()
    uow.rollback = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=None)
    
    return uow


@pytest.fixture
def disable_gpu():
    """Deshabilita GPU para tests específicos."""
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    yield
    
    if original_cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)


# Markers para categorizar tests
def pytest_configure(config):
    """Configuración de markers para pytest."""
    config.addinivalue_line(
        "markers", "unit: marca tests como tests unitarios"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests como tests de integración"
    )
    config.addinivalue_line(
        "markers", "e2e: marca tests como tests end-to-end"
    )
    config.addinivalue_line(
        "markers", "gpu: marca tests que requieren GPU"
    )
    config.addinivalue_line(
        "markers", "slow: marca tests que tardan en ejecutarse"
    )
    config.addinivalue_line(
        "markers", "mock: marca tests que usan mocks"
    )


# Hooks para control de ejecución
def pytest_collection_modifyitems(config, items):
    """Modifica items de test según configuración."""
    # Skip tests GPU si no hay GPU disponible
    skip_gpu = pytest.mark.skip(reason="GPU no disponible")
    
    for item in items:
        if "gpu" in item.keywords:
            # Check if GPU is actually available
            try:
                import tensorflow as tf
                if not tf.config.list_physical_devices('GPU'):
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)


# Fixtures de limpieza
@pytest.fixture(autouse=True)
def cleanup_imports():
    """Limpia imports después de cada test."""
    yield
    
    # Limpiar imports específicos que podrían causar problemas
    modules_to_clean = [
        'tensorflow', 'tf', 'keras',
        'src.orchestrator', 'src.model', 'src.data_processor'
    ]
    
    for module in modules_to_clean:
        if module in sys.modules:
            # No removemos completamente para evitar problemas
            pass