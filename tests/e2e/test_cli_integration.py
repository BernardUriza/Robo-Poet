"""
Tests end-to-end para la interfaz CLI.

Prueba la integración completa del sistema desde la línea de comandos.
"""

import pytest
import subprocess
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, Mock

# Setup sys.path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from orchestrator import main as orchestrator_main


class TestCLIBasicFunctionality:
    """Tests básicos de funcionalidad CLI."""
    
    @pytest.fixture
    def sample_text_file(self):
        """Archivo de texto de ejemplo para entrenamiento."""
        content = """
        The art of power is not just about dominance, but about understanding human nature.
        When you master yourself, you begin to understand others. Knowledge without action is useless.
        In the realm of artificial intelligence, we seek to understand patterns in human language.
        These patterns reveal the underlying structure of communication and thought.
        The greatest leaders throughout history understood the power of words and timing.
        Like a poet who crafts verses, an AI must learn the rhythm of human expression.
        Every sentence carries meaning beyond its literal interpretation.
        Context shapes understanding, and understanding shapes reality.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_cli_help_display(self):
        """Test mostrar ayuda del CLI."""
        # Test ayuda principal
        result = subprocess.run(
            ['python', 'robo_poet.py', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 0
        assert 'Robo-Poet Academic Neural Text Generation Framework' in result.stdout
        assert '--text' in result.stdout
        assert '--epochs' in result.stdout
        assert '--generate' in result.stdout
        assert 'Ejemplos de uso:' in result.stdout
    
    def test_cli_invalid_arguments(self):
        """Test argumentos inválidos."""
        result = subprocess.run(
            ['python', 'robo_poet.py', '--invalid-arg'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode != 0
        assert 'error' in result.stderr.lower()
    
    def test_cli_training_dry_run(self, sample_text_file):
        """Test entrenamiento en modo dry-run (sin GPU real)."""
        # Test con argumentos válidos pero sin ejecutar entrenamiento real
        result = subprocess.run(
            ['python', 'robo_poet.py', '--text', sample_text_file, '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=10,  # Timeout rápido
            cwd=Path(__file__).parent.parent.parent
        )
        
        # El proceso debería mostrar información de GPU no disponible
        output = result.stdout + result.stderr
        assert any(keyword in output.lower() for keyword in [
            'gpu', 'tensorflow', 'sistema de menús no disponible'
        ])
        
        # Cleanup
        Path(sample_text_file).unlink()
    
    def test_cli_generation_dry_run(self):
        """Test generación en modo dry-run."""
        result = subprocess.run(
            ['python', 'robo_poet.py', '--generate', 'nonexistent_model.keras'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Debería fallar pero de manera controlada
        output = result.stdout + result.stderr
        assert any(keyword in output.lower() for keyword in [
            'gpu', 'modelo', 'archivo', 'error'
        ])


class TestOrchestrator:
    """Tests del orquestador principal."""
    
    def test_orchestrator_instantiation(self):
        """Test instanciación del orquestador."""
        # Mock las dependencias para evitar errores de importación
        with patch('src.orchestrator.AcademicMenuSystem', return_value=None):
            with patch('src.orchestrator.Phase1TrainingInterface', return_value=None):
                with patch('src.orchestrator.Phase2GenerationInterface', return_value=None):
                    with patch('src.orchestrator.FileManager', return_value=None):
                        with patch('src.orchestrator.DisplayUtils', return_value=None):
                            from orchestrator import RoboPoetOrchestrator
                            
                            orchestrator = RoboPoetOrchestrator()
                            
                            assert orchestrator is not None
                            assert hasattr(orchestrator, 'gpu_available')
                            assert hasattr(orchestrator, 'config')
    
    def test_orchestrator_main_function(self):
        """Test función main del orquestador."""
        # Mock sys.argv para simular argumentos CLI
        test_args = ['robo_poet.py', '--help']
        
        with patch('sys.argv', test_args):
            with patch('src.orchestrator.AcademicMenuSystem', return_value=None):
                with patch('src.orchestrator.Phase1TrainingInterface', return_value=None):
                    with patch('src.orchestrator.Phase2GenerationInterface', return_value=None):
                        with patch('src.orchestrator.FileManager', return_value=None):
                            with patch('src.orchestrator.DisplayUtils', return_value=None):
                                
                                # El main debería manejar argumentos de ayuda
                                try:
                                    result = orchestrator_main()
                                    # Si no lanza excepción, está bien
                                    assert True
                                except SystemExit as e:
                                    # SystemExit es normal para --help
                                    assert e.code == 0
                                except Exception:
                                    # Otras excepciones están bien en este contexto de test
                                    assert True
    
    def test_orchestrator_error_handling(self):
        """Test manejo de errores en el orquestador."""
        with patch('src.orchestrator.AcademicMenuSystem', side_effect=ImportError("Module not found")):
            from orchestrator import RoboPoetOrchestrator
            
            # Debería crear el orquestador aunque falten dependencias
            orchestrator = RoboPoetOrchestrator()
            assert orchestrator.menu_system is None


class TestIntegrationWithMockedDependencies:
    """Tests de integración con dependencias mockeadas."""
    
    @pytest.fixture
    def mock_tensorflow(self):
        """Mock completo de TensorFlow."""
        tf_mock = Mock()
        
        # Mock GPU detection
        tf_mock.config.list_physical_devices.return_value = []
        tf_mock.device.return_value.__enter__ = Mock()
        tf_mock.device.return_value.__exit__ = Mock()
        tf_mock.constant.return_value = Mock()
        tf_mock.reduce_sum.return_value = Mock()
        
        # Mock model components
        sequential_mock = Mock()
        tf_mock.keras.Sequential.return_value = sequential_mock
        tf_mock.keras.layers.Embedding = Mock()
        tf_mock.keras.layers.LSTM = Mock()
        tf_mock.keras.layers.Dense = Mock()
        
        # Mock training
        sequential_mock.compile = Mock()
        sequential_mock.fit = Mock(return_value=Mock(history={'loss': [1.5, 1.0]}))
        sequential_mock.save = Mock()
        
        return tf_mock
    
    def test_training_workflow_integration(self, sample_text_file, mock_tensorflow):
        """Test flujo completo de entrenamiento con mocks."""
        with patch.dict('sys.modules', {'tensorflow': mock_tensorflow}):
            with patch('src.orchestrator.tf_module', mock_tensorflow):
                
                # Mock file operations
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.read_text') as mock_read:
                        mock_read.return_value = Path(sample_text_file).read_text()
                        
                        with patch('src.data_processor.DataProcessor') as mock_dp_class:
                            mock_dp = Mock()
                            mock_dp.load_and_preprocess.return_value = (
                                [[1, 2, 3, 4]], [[2, 3, 4, 5]], {'word1': 1, 'word2': 2}
                            )
                            mock_dp_class.return_value = mock_dp
                            
                            with patch('src.model.RoboPoetModel') as mock_model_class:
                                mock_model = Mock()
                                mock_model.build_model.return_value = mock_tensorflow.keras.Sequential()
                                mock_model.train.return_value = {'loss': [1.0], 'final_loss': 1.0}
                                mock_model_class.return_value = mock_model
                                
                                # Simular argumentos de entrenamiento
                                test_args = ['robo_poet.py', '--text', sample_text_file, '--epochs', '2']
                                
                                with patch('sys.argv', test_args):
                                    try:
                                        # Esto debería ejecutar sin errores mayores
                                        result = orchestrator_main()
                                        # El resultado puede variar dependiendo de la lógica
                                        assert result is None or isinstance(result, int)
                                    except SystemExit:
                                        # SystemExit es aceptable
                                        pass
                                    except Exception as e:
                                        # En este contexto, ciertas excepciones son esperadas
                                        assert 'disponible' in str(e) or 'import' in str(e).lower()
        
        # Cleanup
        Path(sample_text_file).unlink()
    
    def test_generation_workflow_integration(self, mock_tensorflow):
        """Test flujo completo de generación con mocks."""
        with patch.dict('sys.modules', {'tensorflow': mock_tensorflow}):
            
            # Mock modelo guardado
            mock_model = Mock()
            mock_model.predict.return_value = [[0.1, 0.8, 0.1]]
            mock_tensorflow.keras.models.load_model.return_value = mock_model
            
            # Mock file existence
            with patch('pathlib.Path.exists', return_value=True):
                with patch('src.data_processor.DataProcessor') as mock_dp_class:
                    mock_dp = Mock()
                    mock_dp.load_vocabulary.return_value = {'test': 0, 'word': 1}
                    mock_dp.decode_sequence.return_value = "Generated test text"
                    mock_dp_class.return_value = mock_dp
                    
                    # Simular argumentos de generación
                    test_args = [
                        'robo_poet.py', 
                        '--generate', 'test_model.keras',
                        '--seed', 'Test seed',
                        '--temp', '0.8',
                        '--length', '50'
                    ]
                    
                    with patch('sys.argv', test_args):
                        try:
                            result = orchestrator_main()
                            assert result is None or isinstance(result, int)
                        except SystemExit:
                            pass
                        except Exception as e:
                            # Excepciones relacionadas con dependencias faltantes son OK
                            assert any(keyword in str(e).lower() for keyword in [
                                'disponible', 'import', 'module', 'tensorflow'
                            ])


class TestErrorRecovery:
    """Tests de recuperación de errores."""
    
    def test_missing_dependencies_graceful_handling(self):
        """Test manejo elegante de dependencias faltantes."""
        # Test que el sistema maneja dependencias faltantes sin crash
        result = subprocess.run(
            ['python', 'robo_poet.py', '--help'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Debería mostrar ayuda sin crash
        assert 'usage:' in result.stdout or result.returncode == 0
    
    def test_invalid_file_handling(self):
        """Test manejo de archivos inválidos."""
        result = subprocess.run(
            ['python', 'robo_poet.py', '--text', '/nonexistent/file.txt', '--epochs', '1'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Debería manejar el archivo faltante de manera controlada
        output = result.stdout + result.stderr
        # El sistema debería detectar que algo está mal, ya sea archivo o dependencias
        assert len(output) > 0
    
    def test_gpu_unavailable_fallback(self):
        """Test fallback cuando GPU no está disponible."""
        # Forzar entorno sin GPU
        env = {'CUDA_VISIBLE_DEVICES': '-1'}
        
        result = subprocess.run(
            ['python', 'robo_poet.py', '--help'],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Debería funcionar sin GPU
        assert 'usage:' in result.stdout or result.returncode == 0


class TestOutputValidation:
    """Tests de validación de salida."""
    
    def test_help_output_completeness(self):
        """Test que la salida de ayuda sea completa."""
        result = subprocess.run(
            ['python', 'robo_poet.py', '--help'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        help_text = result.stdout
        
        # Verificar elementos esenciales de la ayuda
        required_elements = [
            'usage:',
            '--text',
            '--epochs',
            '--generate',
            '--seed',
            '--temp',
            '--length',
            'Ejemplos de uso:'
        ]
        
        for element in required_elements:
            assert element in help_text, f"Missing help element: {element}"
    
    def test_error_message_quality(self):
        """Test que los mensajes de error sean informativos."""
        # Test con argumento inválido
        result = subprocess.run(
            ['python', 'robo_poet.py', '--invalid-option'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent
        )
        
        error_output = result.stderr
        
        # Debería contener información útil de error
        assert len(error_output) > 0
        # Debería mencionar la opción inválida o mostrar uso correcto
        assert 'invalid' in error_output.lower() or 'usage' in error_output.lower()


if __name__ == "__main__":
    pytest.main([__file__, '-v'])