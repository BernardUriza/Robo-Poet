#!/usr/bin/env python3
"""
Robo-Poet Academic Framework Orchestrator

Main orchestrator for the modular academic text generation framework.
Coordinates all components and provides unified entry point.

Author: ML Academic Framework
Version: 2.1 - Modular Architecture
Hardware: Optimized for NVIDIA RTX 2000 Ada
Platform: WSL2 + Kali Linux + Windows 11
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List

# Conditional numpy import for testing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️ Numpy no disponible - algunas funcionalidades limitadas")

# Import core modules safely
import os
import json

# Configure environment for potential GPU use
conda_prefix = os.getenv('CONDA_PREFIX', '/usr/local')
if conda_prefix != '/usr/local':
    os.environ['CUDA_HOME'] = conda_prefix
    os.environ['LD_LIBRARY_PATH'] = f'{conda_prefix}/lib:{conda_prefix}/lib64:{os.environ.get("LD_LIBRARY_PATH", "")}'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # PyTorch environment optimizations

# Configure PyTorch environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Import with fallbacks
gpu_available = False
torch_module = None

# Try to import PyTorch
try:
    import torch
    torch_module = torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"🔥 PyTorch GPU available: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("⚠️ PyTorch no disponible - modo CPU solamente")

try:
    from core.unified_config import get_config
except ImportError:
    # Fallback to basic config
    def get_config():
        from types import SimpleNamespace
        return SimpleNamespace(
            gpu=SimpleNamespace(mixed_precision=True, memory_growth=True),
            model=SimpleNamespace(batch_size=32, epochs=10),
            system=SimpleNamespace(debug=True)
        )

# GPU detection with PyTorch
try:
    if torch_module and torch_module.cuda.is_available():
        gpu_available = True
        print(f"🎯 PyTorch GPU funcionando correctamente: {torch_module.cuda.get_device_name(0)}")
    else:
        gpu_available = False
        if torch_module:
            print("⚠️ PyTorch disponible pero GPU no detectada")
        else:
            print("⚠️ PyTorch no disponible")
except Exception as e:
    print(f"⚠️ GPU no disponible, continuando sin GPU: {e}")
    gpu_available = False

# Import framework components with fallbacks
try:
    from interface.menu_system import AcademicMenuSystem
except ImportError as e:
    print(f"⚠️ Menu system not available: {e}")
    AcademicMenuSystem = None

try:
    from interface.phase1_training import Phase1TrainingInterface
except ImportError as e:
    print(f"⚠️ Phase1 interface not available: {e}")
    Phase1TrainingInterface = None

try:
    from interface.phase2_generation import Phase2GenerationInterface
except ImportError as e:
    print(f"⚠️ Phase2 interface not available: {e}")
    Phase2GenerationInterface = None

try:
    from utils.file_manager import FileManager
except ImportError as e:
    print(f"⚠️ File manager not available: {e}")
    FileManager = None

try:
    from utils.display_utils import DisplayUtils
except ImportError as e:
    print(f"⚠️ Display utils not available: {e}")
    DisplayUtils = None

try:
    from utils.input_validator import InputValidator
except ImportError as e:
    print(f"⚠️ Input validator not available: {e}")
    InputValidator = None

# Import PyTorch model components
try:
    from model_pytorch import create_model, RoboPoetModel
    MODEL_TYPE = "PyTorch GPT"
    print("🚀 Using PyTorch GPT model (modern transformer architecture)")
except ImportError as e:
    print(f"❌ PyTorch model not available: {e}")
    print("   Please ensure PyTorch is installed and robo-poet-pytorch directory exists")
    create_model = None
    RoboPoetModel = None
    MODEL_TYPE = "PyTorch GPT (Not Available)"


class RoboPoetOrchestrator:
    """Main orchestrator for the Robo-Poet Academic Framework."""
    
    def __init__(self):
        """Initialize orchestrator with all components."""
        self.config = get_config()
        
        # Show current model type
        print(f"🤖 Model System: {MODEL_TYPE}")
        
        # Academic Performance GPU Requirement
        if torch_module and not torch_module.cuda.is_available():
            print("🎓 ACADEMIC PERFORMANCE WARNING:")
            print("   📚 GPU/CUDA not available - academic benchmarks require GPU")
            print("   🔧 Install CUDA-enabled PyTorch for optimal performance")
        elif torch_module and torch_module.cuda.is_available():
            print(f"🔥 Academic Performance Mode: GPU Available")
            print(f"   🎮 GPU: {torch_module.cuda.get_device_name(0)}")
        
        # Initialize components with fallbacks
        self.menu_system = AcademicMenuSystem() if AcademicMenuSystem else None
        self.phase1_interface = Phase1TrainingInterface(self.config) if Phase1TrainingInterface else None
        self.phase2_interface = Phase2GenerationInterface(self.config) if Phase2GenerationInterface else None
        self.file_manager = FileManager() if FileManager else None
        self.display = DisplayUtils() if DisplayUtils else None
        
        # GPU availability
        self.gpu_available = gpu_available
        self.torch_module = torch_module
    
    def _safe_display(self, method_name, *args, **kwargs):
        """Safely call display methods with fallback."""
        if self.display and hasattr(self.display, method_name):
            return getattr(self.display, method_name)(*args, **kwargs)
        else:
            # Fallback to simple print
            if method_name == 'show_error':
                print(f"❌ {args[0] if args else 'Error'}")
            elif method_name == 'show_warning':
                print(f"⚠️ {args[0] if args else 'Warning'}")
            elif method_name == 'pause_for_user':
                input("Presiona Enter para continuar...")
            else:
                print(f"ℹ️ {args[0] if args else 'Info'}")
    
    def _safe_file_manager(self, method_name, *args, **kwargs):
        """Safely call file manager methods with fallback."""
        if self.file_manager and hasattr(self.file_manager, method_name):
            return getattr(self.file_manager, method_name)(*args, **kwargs)
        else:
            print(f"❌ File manager no disponible para {method_name}")
            return None
    
    def run_interactive_mode(self) -> int:
        """Run the main interactive academic interface."""
        if not self.menu_system:
            print("❌ Sistema de menús no disponible. Por favor instala las dependencias faltantes.")
            return 1
            
        try:
            while True:
                self.menu_system.show_header()
                choice = self.menu_system.show_main_menu()
                
                if choice == '1':
                    # Phase 1: Intensive Training
                    if not self.phase1_interface:
                        print("❌ Interfaz de entrenamiento no disponible.")
                        continue
                        
                    if not self.gpu_available:
                        self._safe_display('show_warning',
                            "GPU no disponible. Se recomienda GPU para entrenamiento eficiente."
                        )
                        
                        # Simple confirmation without InputValidator if not available
                        response = input("¿Continuar de todas formas? (y/N): ").lower().strip()
                        if response not in ('y', 'yes', 's', 'si'):
                            continue
                    
                    self.phase1_interface.run_intensive_training()
                
                elif choice == '2':
                    # Phase 2: Text Generation
                    if not self.phase2_interface:
                        print("❌ Interfaz de generación no disponible.")
                        continue
                    self.phase2_interface.run_generation_studio()
                
                elif choice == '3':
                    # View Available Models
                    self._show_available_models()
                
                elif choice == '4':
                    # HOSPITAL - Cirugía de Gates (NEW)
                    self._run_gate_surgery()
                
                elif choice == '5':
                    # ANÁLISIS - Gradient Flow Analysis (NEW)
                    self._run_gradient_analysis()
                
                elif choice == '6':
                    # Monitor Training Progress
                    self._monitor_training_progress()
                
                elif choice == '7':
                    # Clean All Models
                    self._clean_all_models()
                
                elif choice == '8':
                    # Test Suite Módulo 2
                    self._run_test_suite()
                
                elif choice == 'A':
                    # Ver Logs y Archivos Generados
                    self._view_logs_and_files()
                
                elif choice == 'B':
                    # Explorar Visualizaciones y Gráficos
                    self._explore_visualizations()
                
                elif choice == 'C':
                    # Attention Mechanism Demo & Validation
                    self._run_attention_demos()
                
                elif choice == 'D':
                    # Dataset Preprocessing Pipeline
                    self._run_dataset_preprocessing()
                
                elif choice == '9':
                    # System Configuration and Status
                    self.menu_system.show_system_status()
                
                elif choice == '0':
                    # Exit
                    self.menu_system.show_exit_message()
                    return 0
                
                else:
                    print("❌ Opción inválida. Por favor selecciona 0-9, A-B.")
                    self.display.pause_for_user()
        
        except KeyboardInterrupt:
            print("\n\n🎯 Sistema interrumpido por usuario")
            self.menu_system.show_exit_message()
            return 0
        except Exception as e:
            self.display.show_error(f"Error crítico en orchestrator: {e}")
            return 1
    
    def run_direct_training(self, text_file: str, epochs: int, model_name: str) -> int:
        """Run direct training mode (CLI) with mandatory model name."""
        try:
            if not self.gpu_available:
                print("⚠️ GPU no disponible - entrenamiento será lento en CPU")
            
            print(f"🚀 Modo directo: Entrenando modelo '{model_name}' con {text_file} por {epochs} épocas")
            
            # Validate parameters
            if not Path(text_file).exists():
                self.display.show_error(f"Archivo no encontrado: {text_file}")
                return 1
            
            valid, message = self.file_manager.validate_text_file(text_file)
            if not valid:
                self.display.show_error(message)
                return 1
            
            # Import training modules
            from data_processor import TextProcessor
            from model import LSTMTextGenerator, ModelTrainer, ModelManager
            import json
            from datetime import datetime
            
            print("✅ Archivo válido, iniciando entrenamiento...")
            
            # Prepare data
            processor = TextProcessor(
                sequence_length=self.config.model.sequence_length,
                step_size=3  # Fixed step size for sliding window
            )
            
            # Load and prepare multi-corpus text
            X_onehot, y_onehot = processor.prepare_data("corpus", max_length=500_000)
            
            # Convert one-hot to integer encoding for embedding layer
            X = np.argmax(X_onehot, axis=-1)  # Shape: (samples, sequence_length)
            y = np.argmax(y_onehot, axis=-1)  # Shape: (samples,)
            
            # Build model
            model_builder = LSTMTextGenerator(
                vocab_size=processor.vocab_size,
                sequence_length=self.config.model.sequence_length,
                lstm_units=self.config.model.lstm_units[0] if isinstance(self.config.model.lstm_units, list) else self.config.model.lstm_units,
                variational_dropout_rate=self.config.model.dropout_rate
            )
            model = model_builder.build_model()
            
            # Train model - GPU MANDATORY, no CPU fallback
            if not self.gpu_available:
                print("\n🔴 SISTEMA TERMINADO: GPU es obligatoria para este proyecto académico")
                import sys
                sys.exit(1)
            
            device = '/GPU:0'  # FIXED: Always GPU, no fallback
            trainer = ModelTrainer(model, device)
            
            history = trainer.train(
                X, y,
                batch_size=self.config.model.batch_size,
                epochs=epochs,
                validation_split=self.config.data.validation_split
            )
            
            # Save model and metadata with user-provided name
            model_filename = f"{model_name}.keras"
            model_path = self.file_manager.models_dir / model_filename
            
            ModelManager.save_model(model, str(model_path))
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_file': model_filename,
                'training_file': text_file,
                'epochs': epochs,
                'final_epoch': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_accuracy': float(history.history['accuracy'][-1]),
                'vocab_size': processor.vocab_size,
                'sequence_length': self.config.model.sequence_length,
                'lstm_units': self.config.model.lstm_units,
                'dropout_rate': self.config.model.dropout_rate,
                'batch_size': self.config.model.batch_size,
                'char_to_idx': processor.token_to_idx,  # Legacy name kept for compatibility
                'idx_to_char': processor.idx_to_token,  # Legacy name kept for compatibility
                'token_to_idx': processor.token_to_idx,
                'idx_to_token': processor.idx_to_token,
                'training_start_time': datetime.now().isoformat(),
                'training_end_time': datetime.now().isoformat(),
                'gpu_used': self.gpu_available
            }
            
            metadata_path = self.file_manager.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Entrenamiento completado exitosamente")
            print(f"🏷️  Modelo '{model_name}' guardado")
            print(f"💾 Archivo: {model_path}")
            print(f"📋 Metadata: {metadata_path}")
            print(f"🎨 Ahora puedes usar: python robo_poet.py --generate {model_filename}")
            
            return 0
            
        except Exception as e:
            self.display.show_error(f"Error en entrenamiento directo: {e}")
            return 1
    
    def run_corpus_training(self, epochs: int, model_name: str) -> int:
        """
        Run multi-corpus training mode - automatically uses all texts in corpus/.
        This is the new preferred method that replaces single-file training.
        """
        return self.run_direct_training("corpus", epochs, model_name)
    
    def run_direct_generation(self, model_path: str, seed: str = "The power of", 
                             temperature: float = 0.8, length: int = 200) -> int:
        """Run direct generation mode (CLI)."""
        try:
            if not Path(model_path).exists():
                self.display.show_error(f"Modelo no encontrado: {model_path}")
                return 1
            
            print(f"🎨 Generando texto con modelo: {Path(model_path).name}")
            
            # Import generation modules
            from model import ModelManager
            from data_processor import TextGenerator
            import json
            
            # Load model
            model_manager = ModelManager()
            model = model_manager.load_model(model_path)
            
            if not model:
                self.display.show_error("No se pudo cargar el modelo")
                return 1
            
            # Load metadata
            metadata_path = Path(model_path).parent / (Path(model_path).stem + '_metadata.json')
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                # Support both old and new naming conventions
                char_to_idx = metadata.get('token_to_idx', metadata.get('char_to_idx', {}))
                raw_idx_to_char = metadata.get('idx_to_token', metadata.get('idx_to_char', {}))
                idx_to_char = {int(k): v for k, v in raw_idx_to_char.items()}
            else:
                self.display.show_error("Metadata no encontrada - no se puede generar texto")
                return 1
            
            # Generate text
            generator = TextGenerator(model, char_to_idx, idx_to_char, tokenization='word')
            result = generator.generate(seed, length, temperature)
            
            # Display result
            self.display.format_generation_result(
                result, seed, temperature, length, 0, Path(model_path).name
            )
            
            return 0
            
        except Exception as e:
            self.display.show_error(f"Error en generación directa: {e}")
            return 1
    
    def _show_available_models(self) -> None:
        """Show available models with enhanced information."""
        print("\n📊 MODELOS DISPONIBLES")
        print("=" * 60)
        
        models = self.file_manager.list_available_models_enhanced()
        
        if not models:
            print("📭 No hay modelos entrenados disponibles")
            print("💡 Ejecuta FASE 1: Entrenamiento Intensivo para crear modelos")
        else:
            print(f"📈 Total de modelos: {len(models)}")
            print()
            
            for i, model_info in enumerate(models, 1):
                print(f"{i}. ", end="")
                self.display.format_model_info(model_info)
        
        self.display.pause_for_user()
    
    def _monitor_training_progress(self) -> None:
        """Monitor training progress (placeholder for advanced monitoring)."""
        print("\n📈 MONITOREO DE PROGRESO DE ENTRENAMIENTO")
        print("=" * 60)
        print("🔍 Buscando entrenamientos activos...")
        
        # Check for active TensorBoard logs
        if self.file_manager.logs_dir.exists():
            log_files = list(self.file_manager.logs_dir.glob("*"))
            if log_files:
                print(f"📊 Encontrados {len(log_files)} logs de entrenamiento")
                print("💡 Para monitoreo en tiempo real:")
                print("   tensorboard --logdir logs --port 6006")
                print("   Luego abre: http://localhost:6006")
            else:
                print("📭 No hay logs de entrenamiento disponibles")
        else:
            print("📭 Directorio de logs no encontrado")
        
        print("\n💡 HERRAMIENTAS DE MONITOREO:")
        print("   🖥️ GPU: nvidia-smi")
        print("   📊 TensorBoard: tensorboard --logdir logs")
        print("   🔄 Tiempo real: watch nvidia-smi")
        
        self.display.pause_for_user()
    
    def _run_gate_surgery(self) -> None:
        """
        🏥 HOSPITAL - Ejecutar cirugía de emergencia en modelo con gates saturados
        """
        print("\n🏥 HOSPITAL - CIRUGÍA DE GATES LSTM")
        print("=" * 50)
        
        # Listar modelos disponibles
        models = self.file_manager.list_available_models()
        
        if not models:
            print("📭 No hay modelos disponibles para cirugía")
            self.display.pause_for_user()
            return
        
        print("📊 Modelos disponibles:")
        for i, model_path in enumerate(models, 1):
            model_name = Path(model_path).name
            print(f"   {i}. {model_name}")
        
        print("\n🔍 Selecciona el modelo a diagnosticar:")
        try:
            choice = input("   Número (o 'c' para cancelar): ").strip()
            
            if choice.lower() == 'c':
                print("❌ Cirugía cancelada")
                self.display.pause_for_user()
                return
            
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                
                print(f"\n🔬 Modelo seleccionado: {Path(selected_model).name}")
                print("⚠️  La cirugía modificará permanentemente los gates del modelo")
                
                confirm = input("\n¿Proceder con la cirugía? (s/N): ").lower().strip()
                
                if confirm in ('s', 'si', 'y', 'yes'):
                    # Ejecutar cirugía
                    from hospital.emergency_gate_surgery import quick_surgery
                    
                    print("\n🚨 INICIANDO CIRUGÍA DE EMERGENCIA...")
                    operated_model, report = quick_surgery(selected_model)
                    
                    if operated_model and report:
                        print("\n🎉 CIRUGÍA EXITOSA")
                        print("📊 El modelo fue operado y guardado con prefijo 'operated_'")
                        print("📋 Reporte de cirugía guardado en src/hospital/")
                    else:
                        print("\n❌ La cirugía falló")
                else:
                    print("❌ Cirugía cancelada")
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Entrada inválida")
        except Exception as e:
            self.display.show_error(f"Error en cirugía: {e}")
        
        self.display.pause_for_user()
    
    def _run_gradient_analysis(self) -> None:
        """
        🔬 ANÁLISIS - Ejecutar análisis profundo de gradient flow
        """
        print("\n🔬 ANÁLISIS PROFUNDO DE MODELOS LSTM")
        print("=" * 60)
        print("📋 MENÚ DE ANÁLISIS DISPONIBLES:")
        print()
        print("🔬 DIAGNÓSTICOS DE GRADIENTES:")
        print("1. 📈 Gradient Flow Analysis")
        print("   • Detección de vanishing/exploding gradients") 
        print("   • Análisis de propagación por capas")
        print("   • Métricas de Pascanu et al. 2013")
        print()
        print("🏔️  ANÁLISIS DE PAISAJE DE PÉRDIDA:")
        print("2. 🎯 Sharp vs Flat Minima Detection")
        print("   • Análisis de sharpness del mínimo actual")
        print("   • Predicción de capacidad de generalización")
        print("   • Visualización de curvatura Hessiana")
        print()
        print("🚀 ANÁLISIS INTEGRAL:")
        print("3. 🔍 Suite Completa de Análisis")
        print("   • Combina ambos análisis anteriores")
        print("   • Reporte consolidado de diagnóstico")
        print("   • Recomendaciones integradas")
        print()
        print("🧪 EXPERIMENTOS DE ABLACIÓN:")
        print("4. ⚗️ Ablation Study (Componente Impact)")
        print("   • Identifica componentes críticos del modelo")
        print("   • Compara variantes arquitectónicas")
        print("   • Optimización sistemática de hiperparámetros")
        print()
        print("=" * 60)
        
        choice = input("🎯 Selecciona análisis (1-4): ").strip()
        
        if choice == "1":
            self._run_gradient_flow_analysis()
        elif choice == "2":
            self._run_minima_analysis()
        elif choice == "3":
            self._run_gradient_flow_analysis()
            print("\n" + "="*50)
            self._run_minima_analysis()
        elif choice == "4":
            self._run_ablation_study()
        else:
            print("❌ Opción inválida")
            self.display.pause_for_user()
    
    def _run_gradient_flow_analysis(self) -> None:
        """Ejecutar análisis de gradient flow específicamente."""
        print("\n📈 ANÁLISIS DE GRADIENT FLOW")
        print("=" * 50)
        print("🔬 ENTRADA: Modelo LSTM entrenado")
        print("📊 SALIDA: Métricas de gradientes, gráficos y reporte JSON")
        print("⏱️  TIEMPO: ~2-5 minutos según número de batches")
        print("=" * 50)
        
        # Listar modelos disponibles
        models = self.file_manager.list_available_models()
        
        if not models:
            print("📭 No hay modelos disponibles para análisis")
            self.display.pause_for_user()
            return
        
        print("📊 Modelos disponibles:")
        for i, model_path in enumerate(models, 1):
            model_name = Path(model_path).name
            # Marcar modelos operados
            if 'operated' in model_name:
                print(f"   {i}. {model_name} 🏥 (operado)")
            else:
                print(f"   {i}. {model_name}")
        
        print("\n🔍 Selecciona el modelo a analizar:")
        try:
            choice = input("   Número (o 'c' para cancelar): ").strip()
            
            if choice.lower() == 'c':
                print("❌ Análisis cancelado")
                self.display.pause_for_user()
                return
            
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                
                print(f"\n📊 Modelo seleccionado: {Path(selected_model).name}")
                
                # Solicitar número de batches
                batches_input = input("📈 Número de batches a analizar (default: 30): ").strip()
                num_batches = int(batches_input) if batches_input else 30
                
                # Ejecutar análisis
                from analysis.gradient_analyzer_lite import GradientAnalyzerLite
                
                print(f"\n🔬 INICIANDO ANÁLISIS ({num_batches} batches)...")
                analyzer = GradientAnalyzerLite(selected_model)
                results = analyzer.run_complete_analysis(num_batches)
                
                if results:
                    print("\n🎉 ANÁLISIS DE GRADIENTES COMPLETADO")
                    print("=" * 55)
                    
                    # Mostrar resumen de resultados
                    collapse_info = results.get('collapse_analysis', {})
                    pascanu_info = results.get('pascanu_analysis', {})
                    
                    print("📊 DIAGNÓSTICO DE GRADIENTES:")
                    print(f"┌{'─' * 53}┐")
                    
                    if collapse_info.get('earliest_collapse', -1) >= 0:
                        print(f"│ 🔴 Colapso en batch: {collapse_info['earliest_collapse']:<32} │")
                    else:
                        print(f"│ ✅ Sin colapso detectado{' ' * 32} │")
                    
                    vanishing_status = "⚠️  SÍ" if pascanu_info.get('has_vanishing') else "✅ NO"
                    exploding_status = "⚠️  SÍ" if pascanu_info.get('has_exploding') else "✅ NO"
                    
                    print(f"│ 📉 Vanishing gradients: {vanishing_status:<28} │")
                    print(f"│ 📈 Exploding gradients: {exploding_status:<28} │")
                    print(f"└{'─' * 53}┘")
                    
                    # Status general
                    if not pascanu_info.get('has_vanishing') and not pascanu_info.get('has_exploding'):
                        print("\n🎯 ESTADO: Flujo de gradientes ESTABLE - Modelo saludable")
                    else:
                        print("\n⚠️  ESTADO: Problemas detectados - Considerar ajustes")
                    
                    print(f"\n📁 ARCHIVOS GENERADOS:")
                    print(f"   📊 CSV: gradient_tracking_*.csv")
                    print(f"   📈 Visualización: gradient_analysis_*.png")
                    print(f"   📋 Reporte JSON: gradient_analysis_lite_*.json")
                else:
                    print("\n❌ El análisis falló")
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Entrada inválida")
        except Exception as e:
            self.display.show_error(f"Error en análisis: {e}")
        
        self.display.pause_for_user()
    
    def _run_minima_analysis(self) -> None:
        """Ejecutar análisis de Sharp vs Flat Minima."""
        print("\n🏔️ ANÁLISIS DE PAISAJE DE PÉRDIDA (SHARP VS FLAT MINIMA)")
        print("=" * 70)
        print("🔬 ENTRADA: Modelo LSTM entrenado")
        print("📊 SALIDA: Clasificación de sharpness, visualización del paisaje")
        print("⏱️  TIEMPO: ~5-15 minutos según configuración")
        print("🎯 PROPÓSITO: Predecir capacidad de generalización del modelo")
        print("=" * 70)
        
        # Listar modelos disponibles
        models = self.file_manager.list_available_models()
        
        if not models:
            print("📭 No hay modelos disponibles para análisis")
            self.display.pause_for_user()
            return
        
        print("📊 Modelos disponibles:")
        for i, model_path in enumerate(models, 1):
            model_name = Path(model_path).name
            if 'operated' in model_name:
                print(f"   {i}. {model_name} 🏥 (operado)")
            else:
                print(f"   {i}. {model_name}")
        
        print("\n🔍 Selecciona el modelo a analizar:")
        try:
            choice = input("   Número (o 'c' para cancelar): ").strip()
            
            if choice.lower() == 'c':
                print("❌ Análisis cancelado")
                self.display.pause_for_user()
                return
            
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                
                print(f"\n📊 Modelo seleccionado: {Path(selected_model).name}")
                print("🔬 Configurando análisis de paisaje de pérdida...")
                
                # Configuración de análisis
                print("\n⚙️ CONFIGURACIÓN DEL ANÁLISIS:")
                print("┌─────────────────────────────────────────────────────────────┐")
                print("│ 1. 🚀 RÁPIDO    │ 20 direcciones │ 30 muestras  │ ~2-3 min │")
                print("│ 2. 📊 ESTÁNDAR  │ 50 direcciones │ 100 muestras │ ~5-8 min │") 
                print("│ 3. 🔬 PROFUNDO  │ 100 direcciones│ 200 muestras │ ~10-15min│")
                print("└─────────────────────────────────────────────────────────────┘")
                print("💡 Más direcciones = mayor precisión en detección de sharpness")
                
                config_choice = input("Selecciona configuración (1-3, default: 2): ").strip()
                
                # Configurar parámetros según elección
                if config_choice == "1":
                    config = {
                        'num_directions': 20,
                        'num_samples': 30,
                        'hessian_samples': 10,
                        'save_plots': True
                    }
                    print("⚡ Configuración rápida seleccionada")
                elif config_choice == "3":
                    config = {
                        'num_directions': 100,
                        'num_samples': 200,
                        'hessian_samples': 50,
                        'save_plots': True
                    }
                    print("🔬 Configuración profunda seleccionada")
                else:
                    config = {
                        'num_directions': 50,
                        'num_samples': 100,
                        'hessian_samples': 20,
                        'save_plots': True
                    }
                    print("📊 Configuración estándar seleccionada")
                
                # Ejecutar análisis
                from analysis.minima_analyzer import analyze_model_sharpness
                
                print(f"\n🔬 INICIANDO ANÁLISIS DE SHARPNESS...")
                print("⏳ Este proceso puede tomar varios minutos...")
                
                try:
                    results = analyze_model_sharpness(selected_model, config=config)
                    
                    if results:
                        print("\n🎉 ANÁLISIS DE SHARPNESS COMPLETADO")
                        print("=" * 60)
                        
                        # Mostrar resumen de resultados
                        classification = results.get('sharpness_classification', {})
                        
                        print("📊 RESULTADO DEL ANÁLISIS:")
                        print(f"┌{'─' * 58}┐")
                        print(f"│ 🏷️  Categoría: {classification.get('category', 'N/A'):<44} │")
                        print(f"│ 📈 Sharpness: {classification.get('overall_sharpness', 0):<45.4f} │")
                        print(f"└{'─' * 58}┘")
                        print()
                        print(f"💡 INTERPRETACIÓN:")
                        print(f"   {classification.get('interpretation', 'N/A')}")
                        
                        # Mostrar recomendaciones
                        recommendations = results.get('recommendations', [])
                        if recommendations:
                            print("\n💡 RECOMENDACIONES:")
                            for rec in recommendations:
                                print(f"   • {rec}")
                        
                        # Información de archivos generados
                        viz_path = results.get('visualization_path')
                        if viz_path:
                            print(f"\n📁 Archivos generados:")
                            print(f"   📈 Visualización: {viz_path}")
                            print(f"   📋 Reporte JSON: minima_analysis_*.json")
                        
                        # Mostrar métricas técnicas adicionales
                        perturbation = results.get('perturbation_analysis', {})
                        curvature = results.get('curvature_analysis', {})
                        
                        if perturbation:
                            print(f"\n📊 MÉTRICAS TÉCNICAS:")
                            print(f"   📍 Loss baseline: {perturbation.get('baseline_loss', 0):.4f}")
                        
                        if curvature:
                            print(f"   🔢 Max eigenvalue: {curvature.get('max_eigenvalue', 0):.4f}")
                            print(f"   📏 Condition number: {curvature.get('condition_number', 0):.2f}")
                    else:
                        print("\n❌ El análisis de sharpness falló")
                        
                except Exception as e:
                    print(f"\n❌ Error durante análisis: {e}")
                    print("💡 Tip: Asegúrate de que el modelo sea compatible")
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Entrada inválida")
        except Exception as e:
            self.display.show_error(f"Error en análisis de minima: {e}")
        
        self.display.pause_for_user()
    
    def _run_ablation_study(self) -> None:
        """Ejecutar experimentos de ablación sistemática."""
        print("\n🧪 EXPERIMENTOS DE ABLACIÓN SISTEMÁTICA")
        print("=" * 60)
        print("🔬 ENTRADA: Modelo LSTM entrenado")
        print("📊 SALIDA: Análisis comparativo de componentes")
        print("⏱️  TIEMPO: ~15-30 minutos según configuración")
        print("🎯 PROPÓSITO: Identificar componentes críticos del modelo")
        print("=" * 60)
        
        # Listar modelos disponibles
        models = self.file_manager.list_available_models()
        
        if not models:
            print("📭 No hay modelos disponibles para análisis")
            self.display.pause_for_user()
            return
        
        print("📊 Modelos disponibles:")
        for i, model_path in enumerate(models, 1):
            model_name = Path(model_path).name
            if 'operated' in model_name:
                print(f"   {i}. {model_name} 🏥 (operado)")
            else:
                print(f"   {i}. {model_name}")
        
        print("\n🔍 Selecciona el modelo para ablación:")
        try:
            choice = input("   Número (o 'c' para cancelar): ").strip()
            
            if choice.lower() == 'c':
                print("❌ Experimentos cancelados")
                self.display.pause_for_user()
                return
            
            model_idx = int(choice) - 1
            if 0 <= model_idx < len(models):
                selected_model = models[model_idx]
                
                print(f"\n📊 Modelo seleccionado: {Path(selected_model).name}")
                print("🔬 Configurando experimentos de ablación...")
                
                # Selección de tipos de experimento
                print("\n⚙️ TIPOS DE EXPERIMENTO DISPONIBLES:")
                print("┌─────────────────────────────────────────────────────┐")
                print("│ 1. 📐 LSTM Units    │ Impacto del tamaño de capas   │")
                print("│ 2. 🏗️  LSTM Layers   │ Efecto del número de capas    │") 
                print("│ 3. 💧 Dropout Rate  │ Influencia de regularización  │")
                print("│ 4. 📚 Embeddings    │ Dimensiones de representación │")
                print("│ 5. 🔍 Completo      │ Todos los experimentos        │")
                print("└─────────────────────────────────────────────────────┘")
                
                exp_choice = input("Selecciona tipo de experimento (1-5, default: 1): ").strip()
                
                # Configurar experimentos según elección
                if exp_choice == "2":
                    experiment_types = ['lstm_layers']
                    print("🏗️ Experimento: Número de capas LSTM")
                elif exp_choice == "3":
                    experiment_types = ['dropout_rate']
                    print("💧 Experimento: Dropout rate")
                elif exp_choice == "4":
                    experiment_types = ['embedding_dim']
                    print("📚 Experimento: Dimensiones de embedding")
                elif exp_choice == "5":
                    experiment_types = ['lstm_units', 'lstm_layers', 'dropout_rate', 'embedding_dim']
                    print("🔍 Experimento completo: Todos los componentes")
                else:  # Default: lstm_units
                    experiment_types = ['lstm_units']
                    print("📐 Experimento: Tamaño de unidades LSTM")
                
                # Configurar velocidad del experimento
                print("\n⚡ CONFIGURACIÓN DE VELOCIDAD:")
                print("1. 🚀 Rápido (3 épocas, datos sintéticos) - ~5 min")
                print("2. 📊 Estándar (5 épocas, datos sintéticos) - ~15 min")
                print("3. 🔬 Completo (10 épocas, datos reales si disponible) - ~30+ min")
                
                speed_choice = input("Selecciona velocidad (1-3, default: 1): ").strip()
                
                if speed_choice == "2":
                    epochs = 5
                    quick_mode = True
                    print("📊 Configuración estándar seleccionada")
                elif speed_choice == "3":
                    epochs = 10
                    quick_mode = False
                    print("🔬 Configuración completa seleccionada")
                else:
                    epochs = 3
                    quick_mode = True
                    print("🚀 Configuración rápida seleccionada")
                
                # Ejecutar experimentos
                from analysis.ablation_analyzer import AblationExperimentRunner
                
                print(f"\n🧪 INICIANDO EXPERIMENTOS DE ABLACIÓN...")
                print("⏳ Este proceso puede tomar tiempo...")
                print("📊 Se entrenarán múltiples variantes del modelo...")
                
                try:
                    runner = AblationExperimentRunner(selected_model)
                    results = runner.run_ablation_study(
                        experiment_types=experiment_types,
                        epochs=epochs,
                        quick_mode=quick_mode
                    )
                    
                    if results and not results.get('error'):
                        print("\n🎉 EXPERIMENTOS DE ABLACIÓN COMPLETADOS")
                        print("=" * 50)
                        
                        # Mostrar resultados principales
                        comparative = results.get('comparative_analysis', {})
                        best_overall = comparative.get('best_overall', {})
                        
                        if 'by_perplexity' in best_overall:
                            best = best_overall['by_perplexity']
                            print("🏆 MEJOR CONFIGURACIÓN ENCONTRADA:")
                            print(f"┌{'─' * 48}┐")
                            print(f"│ Experimento: {best['experiment']:<32} │")
                            print(f"│ Variante: {best['variant']:<35} │")
                            print(f"│ Perplexity: {best['metrics']['perplexity']:<31.2f} │")
                            print(f"│ Parámetros: {best['metrics']['total_params']:<29,} │")
                            print(f"└{'─' * 48}┘")
                        
                        # Mostrar insights de ablación
                        insights = comparative.get('ablation_insights', {})
                        if insights:
                            print("\n💡 INSIGHTS DE COMPONENTES:")
                            for component, data in insights.items():
                                impact = data['impact_score']
                                if impact > 0.1:
                                    status = "🔴 CRÍTICO"
                                elif impact > 0.05:
                                    status = "🟡 IMPORTANTE"
                                else:
                                    status = "🟢 MENOR"
                                
                                print(f"   {component}: {status} (impacto: {impact:.3f})")
                        
                        # Generar visualización
                        viz_path = runner.generate_visualization(results)
                        
                        print(f"\n📁 ARCHIVOS GENERADOS:")
                        print(f"   📊 Visualización: {viz_path}")
                        print(f"   📋 Reporte JSON: {results.get('results_path', 'N/A')}")
                        
                        print("\n🎯 RECOMENDACIÓN FINAL:")
                        if best_overall.get('by_efficiency'):
                            eff_best = best_overall['by_efficiency']
                            print(f"   Para mejor eficiencia: {eff_best['variant']}")
                        
                    else:
                        error = results.get('error', 'Error desconocido')
                        print(f"\n❌ Experimentos fallaron: {error}")
                        
                except Exception as e:
                    print(f"\n❌ Error durante experimentos: {e}")
                    print("💡 Tip: Asegúrate de que el modelo sea compatible")
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Entrada inválida")
        except Exception as e:
            self.display.show_error(f"Error en experimentos de ablación: {e}")
        
        self.display.pause_for_user()
    
    def _run_test_suite(self) -> None:
        """Ejecutar suite de tests del Módulo 2."""
        print("\n🧪 SUITE DE TESTS MÓDULO 2")
        print("=" * 60)
        print("🎯 SISTEMA COMPLETO DE VALIDACIÓN Y DEMOSTRACIÓN")
        print("=" * 60)
        print()
        print("Esta suite ejecutará automáticamente:")
        print("• 📚 Entrenamiento de modelo de prueba (3 épocas)")
        print("• 📈 Análisis completo de gradientes")  
        print("• 🏔️ Análisis del paisaje de pérdida")
        print("• 🧪 Experimentos de ablación")
        print("• 🏥 Cirugía de emergencia de gates")
        print("• 📋 Generación de reportes consolidados")
        print()
        print("🕒 Tiempo estimado: 10-20 minutos")
        print("📁 Se generarán logs y reportes detallados")
        print()
        
        # Menú de opciones de testing
        print("📋 OPCIONES DE TESTING DISPONIBLES:")
        print("┌─────────────────────────────────────────────────────┐")
        print("│ 1. 🚀 Demo Rápido     │ Tests básicos (~5 min)     │")
        print("│ 2. 🔬 Validación Full │ Todos los tests (~20 min)  │")  
        print("│ 3. 🎯 Tests Selectivos│ Elegir tests específicos   │")
        print("│ 4. 📊 Tests por Bloque│ Ejecutar por categorías    │")
        print("└─────────────────────────────────────────────────────┘")
        print()
        
        choice = input("Selecciona modalidad (1-4): ").strip()
        
        if choice == "1":
            self._run_quick_demo()
        elif choice == "2":
            self._run_full_validation()
        elif choice == "3":
            self._run_selective_tests()
        elif choice == "4":
            self._run_block_tests()
        else:
            print("❌ Opción inválida")
            self.display.pause_for_user()
            return
    
    def _run_quick_demo(self) -> None:
        """Ejecutar demo rápido del sistema."""
        print("\n🚀 DEMO RÁPIDO - VALIDACIÓN BÁSICA")
        print("=" * 50)
        print("Ejecutando: Entrenamiento → Gradientes → Cirugía")
        print("⏱️ Tiempo estimado: 5 minutos")
        print()
        
        confirm = input("¿Proceder con demo rápido? (s/N): ").lower().strip()
        
        if confirm in ('s', 'si', 'y', 'yes'):
            try:
                from testing.module2_test_suite import run_quick_demo
                
                print("\n🎬 INICIANDO DEMO...")
                results = run_quick_demo()
                
                self._show_test_results(results, "Demo Rápido")
                
            except Exception as e:
                print(f"\n❌ Error en demo: {e}")
        else:
            print("❌ Demo cancelado")
        
        self.display.pause_for_user()
    
    def _run_full_validation(self) -> None:
        """Ejecutar validación completa del sistema."""
        print("\n🔬 VALIDACIÓN COMPLETA - TODOS LOS COMPONENTES")
        print("=" * 60)
        print("Ejecutando TODOS los tests del Módulo 2:")
        print("• Entrenamiento + Análisis + Ablación + Cirugía + Reportes")
        print("⏱️ Tiempo estimado: 15-25 minutos")
        print()
        
        confirm = input("¿Proceder con validación completa? (s/N): ").lower().strip()
        
        if confirm in ('s', 'si', 'y', 'yes'):
            try:
                from testing.module2_test_suite import run_full_validation
                
                print("\n🔬 INICIANDO VALIDACIÓN COMPLETA...")
                results = run_full_validation()
                
                self._show_test_results(results, "Validación Completa")
                
            except Exception as e:
                print(f"\n❌ Error en validación: {e}")
        else:
            print("❌ Validación cancelada")
        
        self.display.pause_for_user()
    
    def _run_selective_tests(self) -> None:
        """Ejecutar tests específicos seleccionados por el usuario."""
        print("\n🎯 TESTS SELECTIVOS - SELECCIÓN PERSONALIZADA")
        print("=" * 55)
        
        available_tests = [
            ('training', '📚 Entrenamiento de Modelo'),
            ('gradient_analysis', '📈 Análisis de Gradientes'),
            ('minima_analysis', '🏔️ Análisis de Minima'),
            ('ablation_experiments', '🧪 Experimentos de Ablación'),
            ('emergency_surgery', '🏥 Cirugía de Emergencia'),
            ('report_generation', '📋 Generación de Reportes')
        ]
        
        print("Selecciona los tests a ejecutar (números separados por comas):")
        for i, (test_key, test_name) in enumerate(available_tests, 1):
            print(f"{i}. {test_name}")
        
        selection = input("\nSelección (ej: 1,2,5): ").strip()
        
        try:
            selected_indices = [int(x.strip()) for x in selection.split(',')]
            selected_tests = [available_tests[i-1][0] for i in selected_indices 
                            if 1 <= i <= len(available_tests)]
            
            if selected_tests:
                print(f"\n✅ Tests seleccionados: {len(selected_tests)}")
                for test_key in selected_tests:
                    test_name = dict(available_tests)[test_key]
                    print(f"   • {test_name}")
                
                confirm = input("\n¿Ejecutar tests seleccionados? (s/N): ").lower().strip()
                
                if confirm in ('s', 'si', 'y', 'yes'):
                    from testing.module2_test_suite import run_selected_tests
                    
                    print("\n🎯 EJECUTANDO TESTS SELECCIONADOS...")
                    results = run_selected_tests(selected_tests)
                    
                    self._show_test_results(results, "Tests Selectivos")
                else:
                    print("❌ Ejecución cancelada")
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Formato de selección inválido")
        except Exception as e:
            print(f"❌ Error ejecutando tests: {e}")
        
        self.display.pause_for_user()
    
    def _run_block_tests(self) -> None:
        """Ejecutar tests organizados por bloques funcionales."""
        print("\n📊 TESTS POR BLOQUES - CATEGORÍAS FUNCIONALES")
        print("=" * 55)
        
        test_blocks = {
            'core': {
                'name': '🎯 CORE (Entrenamiento + Cirugía)',
                'tests': ['training', 'emergency_surgery'],
                'description': 'Funcionalidades básicas del sistema'
            },
            'analysis': {
                'name': '🔬 ANÁLISIS (Gradientes + Minima)',
                'tests': ['gradient_analysis', 'minima_analysis'],
                'description': 'Suite de análisis profundo'
            },
            'advanced': {
                'name': '🧪 AVANZADO (Ablación + Reportes)',
                'tests': ['ablation_experiments', 'report_generation'],
                'description': 'Funcionalidades experimentales'
            }
        }
        
        print("Selecciona bloque de tests:")
        for i, (block_key, block_info) in enumerate(test_blocks.items(), 1):
            print(f"{i}. {block_info['name']}")
            print(f"   {block_info['description']}")
            print(f"   Tests: {len(block_info['tests'])}")
        
        choice = input("\nSelecciona bloque (1-3): ").strip()
        
        try:
            if choice == "1":
                selected_block = test_blocks['core']
            elif choice == "2":
                selected_block = test_blocks['analysis']
            elif choice == "3":
                selected_block = test_blocks['advanced']
            else:
                print("❌ Selección inválida")
                self.display.pause_for_user()
                return
            
            print(f"\n✅ Bloque seleccionado: {selected_block['name']}")
            print(f"📋 Tests incluidos: {len(selected_block['tests'])}")
            
            confirm = input("¿Ejecutar este bloque? (s/N): ").lower().strip()
            
            if confirm in ('s', 'si', 'y', 'yes'):
                from testing.module2_test_suite import run_selected_tests
                
                print(f"\n📊 EJECUTANDO BLOQUE: {selected_block['name']}")
                results = run_selected_tests(selected_block['tests'])
                
                self._show_test_results(results, f"Bloque {selected_block['name']}")
            else:
                print("❌ Ejecución cancelada")
                
        except Exception as e:
            print(f"❌ Error ejecutando bloque: {e}")
        
        self.display.pause_for_user()
    
    def _show_test_results(self, results: Dict, test_type: str) -> None:
        """Mostrar resultados consolidados de los tests."""
        print(f"\n🎉 {test_type.upper()} COMPLETADO")
        print("=" * 60)
        
        if results.get('success'):
            stats = results.get('summary_statistics', {})
            print(f"✅ ÉXITO TOTAL")
            print(f"📊 Tests ejecutados: {stats.get('successful_tests', 0)}/{stats.get('total_tests', 0)}")
            print(f"⏱️ Tiempo total: {stats.get('total_execution_time', 0):.2f} segundos")
            print(f"📈 Tasa de éxito: {stats.get('success_rate', 0):.1%}")
            
            # Mostrar resultados individuales destacados
            individual_results = results.get('individual_test_results', {})
            
            if 'training' in individual_results:
                training = individual_results['training']
                if training.get('status') == 'SUCCESS':
                    print(f"\n🎯 MODELO ENTRENADO:")
                    print(f"   Parámetros: {training.get('parameters', 'N/A'):,}")
                    print(f"   Épocas: {training.get('epochs_trained', 'N/A')}")
            
            if 'gradient_analysis' in individual_results:
                gradient = individual_results['gradient_analysis']
                if gradient.get('status') == 'SUCCESS':
                    print(f"\n📈 ANÁLISIS DE GRADIENTES:")
                    vanishing = "Sí" if gradient.get('has_vanishing') else "No"
                    exploding = "Sí" if gradient.get('has_exploding') else "No"
                    print(f"   Vanishing: {vanishing}")
                    print(f"   Exploding: {exploding}")
            
            if 'emergency_surgery' in individual_results:
                surgery = individual_results['emergency_surgery']
                if surgery.get('status') == 'SUCCESS':
                    success = "Exitosa" if surgery.get('surgery_successful') else "Fallo"
                    print(f"\n🏥 CIRUGÍA: {success}")
            
        else:
            print(f"❌ ALGUNOS TESTS FALLARON")
            failed_tests = [name for name, result in results.get('individual_test_results', {}).items() 
                          if result.get('status') != 'SUCCESS']
            print(f"📋 Tests fallidos: {', '.join(failed_tests)}")
        
        # Información de archivos generados
        metadata = results.get('test_suite_metadata', {})
        log_file = metadata.get('log_file')
        if log_file:
            print(f"\n📁 ARCHIVOS GENERADOS:")
            print(f"   📝 Log detallado: {log_file}")
            print(f"   📊 Reportes JSON/TXT en directorio actual")
        
        print("\n💡 Los reportes contienen análisis detallado de todos los tests")
    
    def _view_logs_and_files(self) -> None:
        """Ver logs y archivos generados por el sistema."""
        print("\n📝 EXPLORADOR DE LOGS Y ARCHIVOS GENERADOS")
        print("=" * 60)
        
        try:
            from utils.file_viewer import FileViewer, LogInspector
            
            viewer = FileViewer()
            inspector = LogInspector()
            
            # Mostrar resumen de archivos
            summary = viewer.display_file_summary()
            print(summary)
            
            print("\n📋 OPCIONES DE VISUALIZACIÓN:")
            print("┌─────────────────────────────────────────────────┐")
            print("│ 1. 📄 Ver log más reciente                     │")
            print("│ 2. 📊 Inspeccionar reporte específico          │")
            print("│ 3. 🔍 Buscar por tipo de archivo              │")
            print("│ 4. 📈 Resumen de todos los logs               │")
            print("└─────────────────────────────────────────────────┘")
            print()
            
            choice = input("Selecciona opción (1-4): ").strip()
            
            if choice == "1":
                self._view_latest_log(inspector)
            elif choice == "2":
                self._inspect_specific_report(viewer)
            elif choice == "3":
                self._browse_by_file_type(viewer)
            elif choice == "4":
                self._show_logs_summary(inspector)
            else:
                print("❌ Opción inválida")
                
        except ImportError:
            print("❌ Sistema de visualización no disponible")
            print("💡 Instala las dependencias necesarias")
        except Exception as e:
            print(f"❌ Error accediendo archivos: {e}")
        
        self.display.pause_for_user()
    
    def _view_latest_log(self, inspector: 'LogInspector') -> None:
        """Ver el log más reciente."""
        print("\n📄 LOG MÁS RECIENTE")
        print("-" * 40)
        
        latest_logs = inspector.find_latest_logs()
        
        if not latest_logs:
            print("📭 No se encontraron logs")
            return
        
        latest = latest_logs[0]
        print(f"📝 Archivo: {latest['name']}")
        print(f"🕒 Fecha: {latest['modified_human']}")
        print(f"📏 Tamaño: {latest['size_human']}")
        
        if 'log_type' in latest:
            print(f"🏷️ Tipo: {latest['log_type']}")
        
        print("\n" + "="*50)
        
        log_data = inspector.viewer.read_log_file(latest['path'], tail_lines=50)
        
        if log_data['success']:
            print("📋 ÚLTIMAS 50 LÍNEAS:")
            print("-" * 30)
            print(log_data['content'])
            
            analysis = log_data['analysis']
            print(f"\n📊 ESTADÍSTICAS:")
            print(f"   Total líneas: {analysis['total_lines']}")
            print(f"   ❌ Errores: {len(analysis['errors'])}")
            print(f"   ⚠️  Warnings: {len(analysis['warnings'])}")
            print(f"   ✅ Éxitos: {len(analysis['success'])}")
        else:
            print(f"❌ Error leyendo log: {log_data['error']}")
    
    def _inspect_specific_report(self, viewer: 'FileViewer') -> None:
        """Inspeccionar un reporte específico."""
        print("\n📊 INSPECTOR DE REPORTES")
        print("-" * 40)
        
        files = viewer.scan_generated_files()
        reports = files.get('reports', [])
        
        if not reports:
            print("📭 No se encontraron reportes")
            return
        
        print("📋 Reportes disponibles:")
        for i, report in enumerate(reports[:10], 1):
            report_type = report.get('report_type', 'Unknown')
            print(f"{i}. {report['name']} - {report_type} ({report['modified_human']})")
        
        try:
            selection = input(f"\nSelecciona reporte (1-{min(10, len(reports))}): ").strip()
            idx = int(selection) - 1
            
            if 0 <= idx < len(reports):
                selected = reports[idx]
                self._show_report_details(selected['path'])
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Entrada inválida")
    
    def _show_report_details(self, report_path: str) -> None:
        """Mostrar detalles de un reporte."""
        print(f"\n📊 DETALLES DEL REPORTE")
        print("=" * 50)
        
        try:
            if report_path.endswith('.json'):
                with open(report_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Mostrar información estructurada según el tipo
                if 'test_suite_metadata' in data:
                    self._show_test_report_details(data)
                elif 'sharpness_classification' in data:
                    self._show_minima_report_details(data)
                elif 'collapse_analysis' in data:
                    self._show_gradient_report_details(data)
                else:
                    # Mostrar JSON genérico
                    import json as json_module
                    print(json_module.dumps(data, indent=2, ensure_ascii=False))
            else:
                # Archivo de texto
                with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                print(content)
                
        except Exception as e:
            print(f"❌ Error leyendo reporte: {e}")
    
    def _show_test_report_details(self, data: Dict) -> None:
        """Mostrar detalles específicos de reporte de tests."""
        metadata = data['test_suite_metadata']
        stats = data['summary_statistics']
        
        print("🧪 REPORTE DE TESTS")
        print(f"🕒 Ejecutado: {metadata['timestamp']}")
        print(f"⏱️ Duración: {stats['total_execution_time']:.2f}s")
        print(f"📊 Tests: {stats['successful_tests']}/{stats['total_tests']}")
        print(f"📈 Éxito: {stats['success_rate']:.1%}")
        
        print(f"\n📋 RESULTADOS INDIVIDUALES:")
        for test_name, result in data['individual_test_results'].items():
            status = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"  {status} {test_name.replace('_', ' ').title()}")
            
            if result['status'] == 'SUCCESS' and 'total_time' in result:
                print(f"      ⏱️ {result['total_time']:.2f}s")
    
    def _show_minima_report_details(self, data: Dict) -> None:
        """Mostrar detalles específicos de análisis de minima."""
        classification = data['sharpness_classification']
        
        print("🏔️ ANÁLISIS DE PAISAJE DE PÉRDIDA")
        print(f"🏷️ Categoría: {classification['category']}")
        print(f"📈 Sharpness: {classification['overall_sharpness']:.4f}")
        print(f"💡 {classification['interpretation']}")
        
        if 'recommendations' in data:
            print(f"\n💡 RECOMENDACIONES:")
            for rec in data['recommendations']:
                print(f"  • {rec}")
    
    def _show_gradient_report_details(self, data: Dict) -> None:
        """Mostrar detalles específicos de análisis de gradientes."""
        collapse = data.get('collapse_analysis', {})
        pascanu = data.get('pascanu_analysis', {})
        
        print("📈 ANÁLISIS DE GRADIENTES")
        print(f"📊 Batches analizados: {data['analysis_metadata']['batches_analyzed']}")
        print(f"⏱️ Duración: {data['analysis_metadata']['duration_minutes']:.2f} min")
        
        print(f"\n📉 DIAGNÓSTICO:")
        print(f"  Vanishing: {'Sí' if pascanu.get('has_vanishing') else 'No'}")
        print(f"  Exploding: {'Sí' if pascanu.get('has_exploding') else 'No'}")
        
        if collapse.get('earliest_collapse', -1) >= 0:
            print(f"  🔴 Colapso en batch: {collapse['earliest_collapse']}")
    
    def _browse_by_file_type(self, viewer: 'FileViewer') -> None:
        """Navegar archivos por tipo."""
        print("\n🔍 NAVEGADOR POR TIPO DE ARCHIVO")
        print("-" * 40)
        
        files = viewer.scan_generated_files()
        
        print("📂 Categorías disponibles:")
        categories = list(files.keys())
        for i, category in enumerate(categories, 1):
            count = len(files[category])
            category_display = {
                'logs': '📝 Logs',
                'reports': '📊 Reportes', 
                'visualizations': '📈 Visualizaciones',
                'models': '🧠 Modelos'
            }.get(category, category)
            
            print(f"{i}. {category_display} ({count} archivos)")
        
        try:
            selection = input(f"\nSelecciona categoría (1-{len(categories)}): ").strip()
            idx = int(selection) - 1
            
            if 0 <= idx < len(categories):
                selected_category = categories[idx]
                self._show_category_files(files[selected_category], selected_category)
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Entrada inválida")
    
    def _show_category_files(self, file_list: List[Dict], category: str) -> None:
        """Mostrar archivos de una categoría específica."""
        category_display = {
            'logs': '📝 LOGS',
            'reports': '📊 REPORTES', 
            'visualizations': '📈 VISUALIZACIONES',
            'models': '🧠 MODELOS'
        }.get(category, category.upper())
        
        print(f"\n{category_display}")
        print("=" * 50)
        
        if not file_list:
            print("📭 No hay archivos en esta categoría")
            return
        
        for i, file_info in enumerate(file_list, 1):
            print(f"{i}. 📄 {file_info['name']}")
            print(f"   📏 {file_info['size_human']} - 🕒 {file_info['modified_human']}")
            
            # Información específica por tipo
            if category == 'logs' and 'log_type' in file_info:
                print(f"   🏷️ {file_info['log_type']}")
            elif category == 'reports' and 'report_type' in file_info:
                print(f"   📊 {file_info['report_type']}")
            elif category == 'visualizations' and 'viz_type' in file_info:
                print(f"   📈 {file_info['viz_type']}")
            elif category == 'models' and 'model_type' in file_info:
                print(f"   🧠 {file_info['model_type']}")
            
            print()
    
    def _show_logs_summary(self, inspector: 'LogInspector') -> None:
        """Mostrar resumen de todos los logs."""
        print("\n📈 RESUMEN DE LOGS DEL SISTEMA")
        print("=" * 50)
        
        recent_logs = inspector.find_latest_logs()
        
        if not recent_logs:
            print("📭 No se encontraron logs")
            return
        
        total_errors = 0
        total_warnings = 0
        total_success = 0
        
        print("📊 LOGS RECIENTES (máximo 10):")
        print("-" * 40)
        
        for log_info in recent_logs:
            print(f"📝 {log_info['name']}")
            print(f"   🕒 {log_info['modified_human']}")
            
            if 'errors' in log_info:
                errors = log_info['errors']
                warnings = log_info.get('warnings', 0)
                success = log_info.get('success_markers', 0)
                
                print(f"   📊 ❌ {errors} | ⚠️ {warnings} | ✅ {success}")
                
                total_errors += errors
                total_warnings += warnings  
                total_success += success
            
            print()
        
        print("📈 ESTADÍSTICAS TOTALES:")
        print(f"   ❌ Total errores: {total_errors}")
        print(f"   ⚠️ Total warnings: {total_warnings}")
        print(f"   ✅ Total éxitos: {total_success}")
        
        if total_errors == 0:
            print("\n🎉 ¡Sistema funcionando sin errores!")
        elif total_errors < total_success:
            print(f"\n⚠️ Sistema mayormente estable ({total_success-total_errors} más éxitos que errores)")
        else:
            print(f"\n🔴 Atención: {total_errors} errores detectados")
    
    def _explore_visualizations(self) -> None:
        """Explorar visualizaciones y gráficos generados."""
        print("\n📈 EXPLORADOR DE VISUALIZACIONES Y GRÁFICOS")
        print("=" * 60)
        
        try:
            from utils.file_viewer import FileViewer
            
            viewer = FileViewer()
            files = viewer.scan_generated_files()
            visualizations = files.get('visualizations', [])
            
            if not visualizations:
                print("📭 No se encontraron visualizaciones")
                print("💡 Ejecuta análisis para generar gráficos:")
                print("   • python robo_poet.py --analyze modelo.keras")
                print("   • python robo_poet.py --minima modelo.keras")
                print("   • python robo_poet.py --test quick")
                self.display.pause_for_user()
                return
            
            print(f"📊 VISUALIZACIONES DISPONIBLES ({len(visualizations)}):")
            print("=" * 50)
            
            for i, viz in enumerate(visualizations, 1):
                viz_type = viz.get('viz_type', 'Unknown')
                print(f"{i}. 📈 {viz['name']}")
                print(f"   🏷️ Tipo: {viz_type}")
                print(f"   📏 {viz['size_human']} - 🕒 {viz['modified_human']}")
                print()
            
            print("📋 OPCIONES:")
            print("┌─────────────────────────────────────────────────┐")
            print("│ 1. 📄 Ver información detallada de gráfico     │")
            print("│ 2. 🗂️  Organizar por tipo de análisis          │")
            print("│ 3. 💻 Mostrar comandos para abrir imágenes     │")
            print("└─────────────────────────────────────────────────┘")
            print()
            
            choice = input("Selecciona opción (1-3): ").strip()
            
            if choice == "1":
                self._show_visualization_details(visualizations)
            elif choice == "2":
                self._organize_visualizations_by_type(visualizations)
            elif choice == "3":
                self._show_image_open_commands(visualizations)
            else:
                print("❌ Opción inválida")
                
        except ImportError:
            print("❌ Sistema de visualización no disponible")
        except Exception as e:
            print(f"❌ Error accediendo visualizaciones: {e}")
        
        self.display.pause_for_user()
    
    def _show_visualization_details(self, visualizations: List[Dict]) -> None:
        """Mostrar detalles de una visualización específica."""
        print("\n📄 DETALLES DE VISUALIZACIÓN")
        print("-" * 40)
        
        try:
            selection = input(f"Selecciona visualización (1-{len(visualizations)}): ").strip()
            idx = int(selection) - 1
            
            if 0 <= idx < len(visualizations):
                viz = visualizations[idx]
                
                print(f"📈 ARCHIVO: {viz['name']}")
                print(f"🗂️ Ruta: {viz['path']}")
                print(f"📏 Tamaño: {viz['size_human']}")
                print(f"🕒 Creado: {viz['modified_human']}")
                
                # Información específica por tipo
                from utils.file_viewer import FileViewer
                viewer = FileViewer()
                viz_info = viewer.get_visualization_info(viz['path'])
                
                if viz_info['success']:
                    print(f"🏷️ Tipo: {viz_info.get('type', 'Unknown')}")
                    print(f"📋 Descripción: {viz_info.get('description', 'N/A')}")
                    
                    if 'contains' in viz_info:
                        print(f"📊 Contiene:")
                        for item in viz_info['contains']:
                            print(f"   • {item}")
                
                # Comando para abrir
                print(f"\n💻 COMANDOS PARA ABRIR:")
                if os.name == 'nt':  # Windows
                    print(f"   start {viz['path']}")
                else:  # Linux/Mac
                    print(f"   xdg-open {viz['path']}")
                    print(f"   eog {viz['path']}  # GNOME")
                    print(f"   feh {viz['path']}  # Lightweight")
            else:
                print("❌ Selección inválida")
                
        except ValueError:
            print("❌ Entrada inválida")
    
    def _organize_visualizations_by_type(self, visualizations: List[Dict]) -> None:
        """Organizar visualizaciones por tipo de análisis."""
        print("\n🗂️ VISUALIZACIONES POR TIPO")
        print("=" * 50)
        
        # Agrupar por tipo
        by_type = {}
        for viz in visualizations:
            viz_type = viz.get('viz_type', 'Unknown')
            if viz_type not in by_type:
                by_type[viz_type] = []
            by_type[viz_type].append(viz)
        
        for viz_type, viz_list in by_type.items():
            print(f"\n📈 {viz_type.upper()} ({len(viz_list)} archivos):")
            print("-" * 40)
            
            for viz in viz_list:
                print(f"  📄 {viz['name']} - {viz['modified_human']}")
    
    def _show_image_open_commands(self, visualizations: List[Dict]) -> None:
        """Mostrar comandos para abrir todas las imágenes."""
        print("\n💻 COMANDOS PARA ABRIR IMÁGENES")
        print("=" * 50)
        
        if os.name == 'nt':  # Windows
            print("🪟 COMANDOS WINDOWS:")
            for viz in visualizations:
                print(f"start \"{viz['path']}\"")
        else:  # Linux/Mac
            print("🐧 COMANDOS LINUX:")
            for viz in visualizations:
                print(f"xdg-open \"{viz['path']}\"")
                
            print(f"\n🖼️ ABRIR TODAS DE UNA VEZ:")
            paths = ' '.join(f'"{viz["path"]}"' for viz in visualizations)
            print(f"xdg-open {paths}")
    
    def _clean_all_models(self) -> None:
        """Clean all models with enhanced confirmation."""
        print("\n🧹 LIMPIAR TODOS LOS MODELOS")
        print("=" * 50)
        
        models = self.file_manager.list_available_models()
        if not models:
            print("✅ No hay modelos para limpiar")
            self.display.pause_for_user()
            return
        
        print(f"📊 Se encontraron {len(models)} modelos")
        
        # Calculate total size
        total_size = 0
        for model_path in models:
            total_size += Path(model_path).stat().st_size
        
        total_mb = total_size / (1024 * 1024)
        print(f"💾 Espacio total a liberar: {total_mb:.1f} MB")
        
        self.display.show_warning(
            "Esta acción eliminará PERMANENTEMENTE todos los modelos entrenados.\n"
            "   No podrás usar FASE 2 (Generación) hasta entrenar nuevos modelos."
        )
        
        confirm = input("\n🗑️ ¿Confirmar limpieza? (escribe 'ELIMINAR' para confirmar): ").strip()
        
        if confirm != 'ELIMINAR':
            print("❌ Limpieza cancelada")
            self.display.pause_for_user()
            return
        
        # Perform cleanup
        results = self.file_manager.clean_all_models()
        self.display.format_cleanup_results(results)
        self.display.pause_for_user()
    
    def _run_attention_demos(self) -> None:
        """Ejecutar demos y validación del mecanismo de atención."""
        print("\n🎯 ATTENTION MECHANISM DEMO & VALIDATION")
        print("=" * 60)
        print("🎭 Target: Beat LSTM baseline (val_loss = 6.5)")
        print("📐 Implementation: Scaled Dot-Product Attention")
        print()
        
        print("🔬 OPCIONES DISPONIBLES:")
        print("1. 📖 Conceptual Demo (sin dependencias)")
        print("2. 🧪 Validation Suite (requiere TensorFlow)")
        print("3. 📋 Architecture Documentation")
        print("4. 🔙 Volver al menú principal")
        print()
        
        try:
            choice = input("🎯 Selecciona una opción (1-4): ").strip()
            
            if choice == '1':
                print("\n🚀 Ejecutando demo conceptual...")
                import subprocess
                result = subprocess.run([
                    'python', 'demos/demo_attention_concept.py'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print("⚠️ Demo conceptual no disponible (dependencias faltantes)")
                    print("💡 El demo muestra la arquitectura y validaciones matemáticas")
                    
            elif choice == '2':
                print("\n🧪 Ejecutando suite de validación...")
                try:
                    import sys
                    sys.path.insert(0, 'src')
                    from attention.attention_validator import AttentionValidator
                    
                    validator = AttentionValidator(sequence_length=128, d_model=256)
                    results = validator.run_full_validation()
                    
                    if results['summary']['overall_status'] == 'PASSED':
                        print("🎉 ¡Attention mechanism completamente validado!")
                    else:
                        print("⚠️ Validación parcial - revisar logs")
                        
                except ImportError:
                    print("❌ TensorFlow no disponible")
                    print("💡 Instalar con: pip install tensorflow numpy")
                    
            elif choice == '3':
                print("\n📚 ATTENTION ARCHITECTURE DOCUMENTATION")
                print("=" * 50)
                print("📁 Documentación disponible:")
                print("   📖 docs/technical/ATTENTION_IMPLEMENTATION_SUMMARY.md")
                print("   🔧 src/attention/scaled_dot_product_attention.py")
                print("   🧪 src/attention/attention_validator.py")
                print()
                print("🎯 Características clave:")
                print("   ✅ Pure TensorFlow (no pre-built layers)")
                print("   ✅ Shape assertions y gradient tracking")
                print("   ✅ Causal masking para autoregressive generation")
                print("   ✅ Dropout después de softmax")
                print("   ✅ Optimizado para sequence_length=128, d_model=256")
                
            elif choice == '4':
                return
            else:
                print("❌ Opción inválida")
                
        except Exception as e:
            print(f"❌ Error en attention demos: {e}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def _run_dataset_preprocessing(self) -> None:
        """Ejecutar pipeline de preprocesamiento de dataset."""
        print("\n🏗️ DATASET PREPROCESSING PIPELINE")
        print("=" * 60)
        print("🎯 Objetivo: Unificar corpus disperso para mejor convergencia")
        print("🎭 Corpus actual: Shakespeare + Alice (4 archivos)")
        print()
        
        print("🔧 OPCIONES DE PREPROCESAMIENTO:")
        print("1. 🚀 Ejecutar Pipeline Completo (Recomendado)")
        print("2. 📊 Analizar Corpus Actual")
        print("3. 🔍 Validar Dataset Procesado")
        print("4. 🔙 Volver al menú principal")
        print()
        
        try:
            choice = input("🎯 Selecciona una opción (1-4): ").strip()
            
            if choice == '1':
                print("\n🚀 EJECUTANDO PIPELINE COMPLETO...")
                print("=" * 50)
                
                try:
                    import sys
                    sys.path.insert(0, 'src')
                    from data.dataset_preprocessor import DatasetPreprocessor, PreprocessingConfig
                    
                    # Configuración optimizada para Shakespeare & Alice
                    config = PreprocessingConfig(
                        preserve_case=True,
                        preserve_verse_structure=True,
                        max_vocab_size=15000,
                        remove_rare_words=True,
                        rare_word_threshold=2
                    )
                    
                    preprocessor = DatasetPreprocessor(config)
                    result = preprocessor.process_full_pipeline("corpus")
                    
                    if result['success']:
                        print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
                        print(f"📚 Documentos: {result['documents_loaded']}")
                        print(f"📝 Vocabulario: {result['vocabulary_size']:,}")
                        print(f"📊 Corpus: {result['corpus_size']:,} chars")
                        print(f"⏱️ Tiempo: {result['processing_time']:.2f}s")
                        print()
                        print("💡 Dataset unificado disponible en data/processed/")
                        print("🚀 Ahora entrena con: python robo_poet.py --model unified_model")
                    else:
                        print(f"❌ Error en pipeline: {result.get('error', 'Unknown')}")
                        
                except ImportError as e:
                    print(f"❌ Error de importación: {e}")
                    print("💡 Algunos módulos requieren dependencias adicionales")
                    
            elif choice == '2':
                print("\n📊 ANÁLISIS DEL CORPUS ACTUAL")
                print("=" * 50)
                
                from pathlib import Path
                corpus_path = Path("corpus")
                
                if corpus_path.exists():
                    txt_files = list(corpus_path.glob("*.txt"))
                    
                    if txt_files:
                        print(f"✅ Encontrados {len(txt_files)} archivos:")
                        
                        total_size = 0
                        for txt_file in sorted(txt_files):
                            size = txt_file.stat().st_size
                            total_size += size
                            
                            # Análisis básico del contenido
                            try:
                                with open(txt_file, 'r', encoding='utf-8') as f:
                                    content = f.read()[:1000]  # Primera muestra
                                
                                word_count = len(content.split())
                                
                                # Detectar tipo
                                if "shakespeare" in txt_file.name.lower() or "hamlet" in txt_file.name.lower():
                                    doc_type = "🎭 Drama/Poetry"
                                elif "alice" in txt_file.name.lower():
                                    doc_type = "📚 Narrative"
                                else:
                                    doc_type = "📖 General"
                                
                                print(f"   {doc_type} {txt_file.name}: {size:,} bytes, ~{word_count*10:,} words")
                                
                            except Exception as e:
                                print(f"   ❌ {txt_file.name}: Error - {e}")
                        
                        print(f"\n📈 RESUMEN:")
                        print(f"   Total: {total_size:,} bytes ({total_size/1024:.1f} KB)")
                        print(f"   Problema: Archivos dispersos → convergencia lenta")
                        print(f"   Solución: Unificar con marcadores de documento")
                        
                    else:
                        print("❌ No se encontraron archivos .txt en corpus/")
                else:
                    print("❌ Directorio corpus/ no encontrado")
                    
            elif choice == '3':
                print("\n🔍 VALIDACIÓN DE DATASET PROCESADO")
                print("=" * 50)
                
                processed_dir = Path("data/processed")
                if processed_dir.exists():
                    files = list(processed_dir.glob("*.txt")) + list(processed_dir.glob("*.json"))
                    
                    if files:
                        print(f"✅ Dataset procesado encontrado: {len(files)} archivos")
                        
                        for file_path in sorted(files):
                            size = file_path.stat().st_size
                            print(f"   📄 {file_path.name}: {size:,} bytes")
                        
                        # Verificar splits
                        splits_dir = processed_dir / "splits"
                        if splits_dir.exists():
                            splits = list(splits_dir.glob("*.txt"))
                            print(f"   📂 Splits disponibles: {len(splits)}")
                            for split_file in splits:
                                print(f"     📊 {split_file.name}")
                        
                        print("\n💡 Dataset listo para usar con modelo unificado")
                        
                    else:
                        print("❌ No se encontró dataset procesado")
                        print("💡 Ejecuta primero la opción 1 (Pipeline Completo)")
                else:
                    print("❌ Directorio data/processed/ no encontrado")
                    print("💡 Ejecuta primero la opción 1 (Pipeline Completo)")
                    
            elif choice == '4':
                return
            else:
                print("❌ Opción inválida")
                
        except Exception as e:
            print(f"❌ Error en preprocessing: {e}")
        
        input("\n📖 Presiona Enter para continuar...")


def main():
    """Main entry point for the Robo-Poet Academic Framework."""
    parser = argparse.ArgumentParser(
        description="🎓 Robo-Poet Academic Neural Text Generation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python robo_poet.py                                                  # Interfaz académica interactiva
  python robo_poet.py --model mi_modelo --epochs 20                   # Entrenamiento multi-corpus (usa corpus/)
  python robo_poet.py --generate mi_modelo.keras                      # Generación directa
  python robo_poet.py --generate mi_modelo.keras --seed "The power" --temp 0.8
  python robo_poet.py --surgery modelo.keras                          # Cirugía de gates saturados
  python robo_poet.py --analyze modelo.keras --batches 30             # Análisis de gradientes
  python robo_poet.py --minima modelo.keras --config fast             # Análisis de paisaje de pérdida
  python robo_poet.py --ablation modelo.keras --experiments all       # Experimentos de ablación
  python robo_poet.py --test quick                                    # Tests rápidos del Módulo 2
  python robo_poet.py --test selective --test-selection training      # Tests específicos

IMPORTANTE: El sistema ahora usa automáticamente TODOS los archivos .txt en la carpeta 'corpus/'
            para entrenar modelos más ricos y diversos. Simplemente pon tus textos en corpus/
        """
    )
    
    # Training arguments - now uses multi-corpus automatically
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas (default: 20)')
    parser.add_argument('--model', help='Nombre del modelo a entrenar (usa automáticamente corpus/)')
    
    # Generation arguments
    parser.add_argument('--generate', help='Modelo para generación de texto')
    parser.add_argument('--seed', default='The power of', help='Seed para generación (default: "The power of")')
    parser.add_argument('--temp', '--temperature', type=float, default=0.8, 
                       help='Temperature para generación (default: 0.8)')
    parser.add_argument('--length', type=int, default=200, help='Longitud de generación (default: 200)')
    
    # Analysis and repair arguments (NEW)
    parser.add_argument('--surgery', help='Aplicar cirugía de emergencia a modelo con gates saturados')
    parser.add_argument('--analyze', help='Analizar flujo de gradientes del modelo')
    parser.add_argument('--batches', type=int, default=30, help='Batches para análisis (default: 30)')
    parser.add_argument('--minima', help='Analizar paisaje de pérdida (sharp vs flat minima)')
    parser.add_argument('--config', choices=['fast', 'standard', 'deep'], default='standard',
                       help='Configuración de análisis de minima: fast/standard/deep (default: standard)')
    parser.add_argument('--ablation', help='Ejecutar experimentos de ablación sistemática')
    parser.add_argument('--experiments', choices=['lstm_units', 'lstm_layers', 'dropout_rate', 'embedding_dim', 'all'], 
                       default='lstm_units', help='Tipo de experimentos de ablación (default: lstm_units)')
    
    # Testing arguments (NEW)
    parser.add_argument('--test', choices=['quick', 'full', 'selective'], 
                       help='Ejecutar suite de tests del Módulo 2')
    parser.add_argument('--test-selection', nargs='+',
                       choices=['training', 'gradient_analysis', 'minima_analysis', 
                               'ablation_experiments', 'emergency_surgery', 'report_generation'],
                       help='Tests específicos para modo selective')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = RoboPoetOrchestrator()
    
    try:
        # Surgery mode (NEW)
        if args.surgery:
            from hospital.emergency_gate_surgery import quick_surgery
            print("🚨 INICIANDO CIRUGÍA DE EMERGENCIA...")
            operated_model, report = quick_surgery(args.surgery)
            if operated_model:
                print("🎉 Cirugía exitosa - modelo operado guardado")
                return 0
            else:
                print("❌ Cirugía falló")
                return 1
        
        # Gradient Analysis mode (NEW)
        elif args.analyze:
            from analysis.gradient_analyzer_lite import GradientAnalyzerLite
            print("🔬 INICIANDO ANÁLISIS DE GRADIENTES...")
            analyzer = GradientAnalyzerLite(args.analyze)
            results = analyzer.run_complete_analysis(args.batches)
            if results:
                print("🎉 Análisis completo exitoso")
                return 0
            else:
                print("❌ Análisis falló")
                return 1
        
        # Minima Analysis mode (NEW)
        elif args.minima:
            from analysis.minima_analyzer import analyze_model_sharpness
            print("🏔️ INICIANDO ANÁLISIS DE PAISAJE DE PÉRDIDA...")
            
            # Configure analysis based on --config argument
            if args.config == 'fast':
                config = {
                    'num_directions': 20,
                    'num_samples': 30,
                    'hessian_samples': 10,
                    'save_plots': True
                }
                print("⚡ Configuración rápida")
            elif args.config == 'deep':
                config = {
                    'num_directions': 100,
                    'num_samples': 200,
                    'hessian_samples': 50,
                    'save_plots': True
                }
                print("🔬 Configuración profunda")
            else:  # standard
                config = {
                    'num_directions': 50,
                    'num_samples': 100,
                    'hessian_samples': 20,
                    'save_plots': True
                }
                print("📊 Configuración estándar")
            
            try:
                results = analyze_model_sharpness(args.minima, config=config)
                if results:
                    classification = results.get('sharpness_classification', {})
                    print(f"\n🎉 ANÁLISIS COMPLETADO")
                    print(f"🏷️  Categoría: {classification.get('category', 'N/A')}")
                    print(f"📈 Sharpness: {classification.get('overall_sharpness', 0):.4f}")
                    print(f"💡 {classification.get('interpretation', 'N/A')}")
                    return 0
                else:
                    print("❌ Análisis de minima falló")
                    return 1
            except Exception as e:
                print(f"❌ Error en análisis de minima: {e}")
                return 1
        
        # Ablation Study mode (NEW)
        elif args.ablation:
            from analysis.ablation_analyzer import run_quick_ablation_study
            print("🧪 INICIANDO EXPERIMENTOS DE ABLACIÓN...")
            
            # Configure experiment types
            if args.experiments == 'all':
                experiment_types = ['lstm_units', 'lstm_layers', 'dropout_rate', 'embedding_dim']
                print("🔍 Experimentos: Todos los componentes")
            else:
                experiment_types = [args.experiments]
                print(f"📐 Experimento: {args.experiments}")
            
            try:
                results = run_quick_ablation_study(
                    args.ablation, 
                    experiment_types=experiment_types
                )
                
                if results and not results.get('error'):
                    comparative = results.get('comparative_analysis', {})
                    best_overall = comparative.get('best_overall', {})
                    
                    if 'by_perplexity' in best_overall:
                        best = best_overall['by_perplexity']
                        print(f"\n🎉 EXPERIMENTOS COMPLETADOS")
                        print(f"🏆 Mejor configuración: {best['variant']}")
                        print(f"📈 Perplexity: {best['metrics']['perplexity']:.2f}")
                        print(f"📊 Visualización: {results.get('visualization_path', 'N/A')}")
                    return 0
                else:
                    error = results.get('error', 'Error desconocido')
                    print(f"❌ Experimentos fallaron: {error}")
                    return 1
                    
            except Exception as e:
                print(f"❌ Error en experimentos de ablación: {e}")
                return 1
        
        # Testing mode (NEW)
        elif args.test:
            print("🧪 EJECUTANDO SUITE DE TESTS MÓDULO 2...")
            
            if args.test == 'quick':
                from testing.module2_test_suite import run_quick_demo
                print("🚀 Modo: Demo rápido")
                results = run_quick_demo()
                
            elif args.test == 'full':
                from testing.module2_test_suite import run_full_validation
                print("🔬 Modo: Validación completa")
                results = run_full_validation()
                
            elif args.test == 'selective':
                from testing.module2_test_suite import run_selected_tests
                test_selection = args.test_selection or ['training', 'gradient_analysis']
                print(f"🎯 Modo: Tests selectivos - {test_selection}")
                results = run_selected_tests(test_selection)
            
            if results and results.get('success'):
                stats = results.get('summary_statistics', {})
                print(f"\n🎉 TESTS COMPLETADOS EXITOSAMENTE")
                print(f"✅ Éxito: {stats.get('successful_tests', 0)}/{stats.get('total_tests', 0)}")
                print(f"⏱️ Tiempo: {stats.get('total_execution_time', 0):.2f}s")
                return 0
            else:
                print(f"\n❌ ALGUNOS TESTS FALLARON")
                return 1
        
        # Direct training mode - now uses multi-corpus automatically
        elif args.model:
            print(f"🚀 ENTRENAMIENTO MULTI-CORPUS AUTOMÁTICO")
            print(f"   📚 Usando todos los textos de la carpeta 'corpus/'")
            print(f"   🎯 Modelo: {args.model}")
            print(f"   📊 Épocas: {args.epochs}")
            return orchestrator.run_corpus_training(args.epochs, args.model)
        
        # Direct generation mode  
        elif args.generate:
            return orchestrator.run_direct_generation(args.generate, args.seed, args.temp, args.length)
        
        # Interactive mode (default)
        else:
            return orchestrator.run_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n🎯 Proceso interrumpido por usuario")
        return 0
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())