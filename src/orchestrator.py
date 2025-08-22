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
from typing import Optional

# Configure GPU environment FIRST, before any imports
from gpu_detection import configure_gpu_environment, detect_gpu_for_wsl2

# Configure environment
configure_gpu_environment()

# Detect GPU with WSL2 support
gpu_available, tf_module = detect_gpu_for_wsl2()

# Import framework components
from config import get_config
from interface.menu_system import AcademicMenuSystem
from interface.phase1_training import Phase1TrainingInterface
from interface.phase2_generation import Phase2GenerationInterface
from utils.file_manager import FileManager
from utils.display_utils import DisplayUtils
from utils.input_validator import InputValidator


class RoboPoetOrchestrator:
    """Main orchestrator for the Robo-Poet Academic Framework."""
    
    def __init__(self):
        """Initialize orchestrator with all components."""
        self.config = get_config()
        self.menu_system = AcademicMenuSystem()
        self.phase1_interface = Phase1TrainingInterface(self.config)
        self.phase2_interface = Phase2GenerationInterface(self.config)
        self.file_manager = FileManager()
        self.display = DisplayUtils()
        
        # GPU availability
        self.gpu_available = gpu_available
        self.tf_module = tf_module
    
    def run_interactive_mode(self) -> int:
        """Run the main interactive academic interface."""
        try:
            while True:
                self.menu_system.show_header()
                choice = self.menu_system.show_main_menu()
                
                if choice == '1':
                    # Phase 1: Intensive Training
                    if not self.gpu_available:
                        self.display.show_warning(
                            "GPU no disponible. Se recomienda GPU para entrenamiento eficiente."
                        )
                        validator = InputValidator()
                        if not validator.get_confirmation("¿Continuar de todas formas?", False):
                            continue
                    
                    self.phase1_interface.run_intensive_training()
                
                elif choice == '2':
                    # Phase 2: Text Generation
                    self.phase2_interface.run_generation_studio()
                
                elif choice == '3':
                    # View Available Models
                    self._show_available_models()
                
                elif choice == '4':
                    # Monitor Training Progress
                    self._monitor_training_progress()
                
                elif choice == '5':
                    # Clean All Models
                    self._clean_all_models()
                
                elif choice == '6':
                    # System Configuration and Status
                    self.menu_system.show_system_status()
                
                elif choice == '7':
                    # Exit
                    self.menu_system.show_exit_message()
                    return 0
                
                else:
                    print("❌ Opción inválida. Por favor selecciona 1-7.")
                    self.display.pause_for_user()
        
        except KeyboardInterrupt:
            print("\n\n🎯 Sistema interrumpido por usuario")
            self.menu_system.show_exit_message()
            return 0
        except Exception as e:
            self.display.show_error(f"Error crítico en orchestrator: {e}")
            return 1
    
    def run_direct_training(self, text_file: str, epochs: int) -> int:
        """Run direct training mode (CLI)."""
        try:
            if not self.gpu_available:
                print("⚠️ GPU no disponible - entrenamiento será lento en CPU")
            
            print(f"🚀 Modo directo: Entrenando con {text_file} por {epochs} épocas")
            
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
            import tensorflow as tf
            import json
            from datetime import datetime
            
            print("✅ Archivo válido, iniciando entrenamiento...")
            
            # Prepare data
            processor = TextProcessor(
                sequence_length=self.config.model.sequence_length,
                step_size=self.config.training.step_size
            )
            X, y = processor.prepare_data(text_file)
            
            # Build model
            model_builder = LSTMTextGenerator(
                vocab_size=processor.vocab_size,
                sequence_length=self.config.model.sequence_length,
                lstm_units=self.config.model.lstm_units,
                dropout_rate=self.config.model.dropout_rate
            )
            model = model_builder.build_model()
            
            # Train model
            device = '/GPU:0' if self.gpu_available else '/CPU:0'
            trainer = ModelTrainer(model, device)
            
            history = trainer.train(
                X, y,
                batch_size=self.config.training.batch_size,
                epochs=epochs,
                validation_split=self.config.training.validation_split
            )
            
            # Save model and metadata
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"robo_poet_model_{timestamp}.keras"
            model_path = self.file_manager.models_dir / model_filename
            
            ModelManager.save_model(model, str(model_path))
            
            # Save metadata
            metadata = {
                'model_name': model_filename,
                'training_file': text_file,
                'epochs': epochs,
                'final_epoch': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_accuracy': float(history.history['accuracy'][-1]),
                'vocab_size': processor.vocab_size,
                'sequence_length': self.config.model.sequence_length,
                'lstm_units': self.config.model.lstm_units,
                'dropout_rate': self.config.model.dropout_rate,
                'batch_size': self.config.training.batch_size,
                'char_to_idx': processor.char_to_idx,
                'idx_to_char': processor.idx_to_char,
                'training_start_time': datetime.now().isoformat(),
                'training_end_time': datetime.now().isoformat(),
                'gpu_used': self.gpu_available
            }
            
            metadata_path = self.file_manager.models_dir / f"{Path(model_filename).stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Entrenamiento completado exitosamente")
            print(f"💾 Modelo guardado: {model_path}")
            print(f"📋 Metadata guardada: {metadata_path}")
            print(f"🎨 Ahora puedes usar el modo generación")
            
            return 0
            
        except Exception as e:
            self.display.show_error(f"Error en entrenamiento directo: {e}")
            return 1
    
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
                char_to_idx = metadata.get('char_to_idx', {})
                raw_idx_to_char = metadata.get('idx_to_char', {})
                idx_to_char = {int(k): v for k, v in raw_idx_to_char.items()}
            else:
                self.display.show_error("Metadata no encontrada - no se puede generar texto")
                return 1
            
            # Generate text
            generator = TextGenerator(model, char_to_idx, idx_to_char)
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


def main():
    """Main entry point for the Robo-Poet Academic Framework."""
    parser = argparse.ArgumentParser(
        description="🎓 Robo-Poet Academic Neural Text Generation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python robo_poet.py                                    # Interfaz académica interactiva
  python robo_poet.py --text archivo.txt --epochs 20    # Entrenamiento directo
  python robo_poet.py --generate modelo.keras           # Generación directa
  python robo_poet.py --generate modelo.keras --seed "The power" --temp 0.8
        """
    )
    
    # Training arguments
    parser.add_argument('--text', help='Archivo de texto para entrenamiento')
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas (default: 20)')
    
    # Generation arguments
    parser.add_argument('--generate', help='Modelo para generación de texto')
    parser.add_argument('--seed', default='The power of', help='Seed para generación (default: "The power of")')
    parser.add_argument('--temp', '--temperature', type=float, default=0.8, 
                       help='Temperature para generación (default: 0.8)')
    parser.add_argument('--length', type=int, default=200, help='Longitud de generación (default: 200)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = RoboPoetOrchestrator()
    
    try:
        # Direct training mode
        if args.text:
            return orchestrator.run_direct_training(args.text, args.epochs)
        
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