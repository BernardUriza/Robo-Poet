#!/usr/bin/env python3
"""
Robo-Poet: Academic Neural Text Generation Framework

Interfaz académica unificada para entrenamiento y generación de texto con LSTM.
Sistema de dos fases: entrenamiento intensivo y generación pura.

Author: Student ML Researcher
Version: 2.0.0 - Unified Academic Interface
Hardware: Optimized for NVIDIA RTX 2000 Ada
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from config import get_config, GPUConfigurator
from data_processor import TextProcessor, TextGenerator
from model import LSTMTextGenerator, ModelTrainer, ModelManager

class AcademicInterface:
    """Interfaz académica unificada para Robo-Poet."""
    
    def __init__(self):
        """Initialize academic interface."""
        self.model_config, self.system_config = get_config()
        self.current_phase = None
        self.device = None
        
    def show_header(self):
        """Display academic header."""
        print("""
╭─────────────────────────────────────────────────╮
│   🎓 ROBO-POET: INTERFAZ ACADÉMICA v2.0       │
│   Sistema de Entrenamiento y Generación LSTM   │
│   RTX 2000 Ada - Academic Framework           │
╰─────────────────────────────────────────────────╯""")
        
    def show_main_menu(self) -> str:
        """Display main menu and get user choice."""
        print("\n📚 MENÚ PRINCIPAL")
        print("=" * 50)
        print("1. 🔥 FASE 1: Entrenamiento Intensivo (1+ hora)")
        print("2. 🎨 FASE 2: Generación de Texto")
        print("3. 📊 Ver Modelos Disponibles")
        print("4. 📈 Monitorear Progreso de Entrenamiento")
        print("5. ⚙️  Configuración del Sistema")
        print("6. 🚪 Salir")
        print("-" * 50)
        
        while True:
            try:
                choice = input("🎯 Selecciona una opción (1-6): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6']:
                    return choice
                else:
                    print("❌ Opción inválida. Ingresa un número del 1 al 6.")
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                sys.exit(0)
    
    def show_system_status(self):
        """Display current system configuration."""
        print("\n⚙️  CONFIGURACIÓN DEL SISTEMA")
        print("=" * 50)
        
        # Setup GPU info
        gpu_available = GPUConfigurator.setup_gpu()
        device = GPUConfigurator.get_device_strategy()
        
        print(f"💻 Device: {device}")
        print(f"🎯 GPU disponible: {'✅ Sí' if gpu_available else '❌ No'}")
        print(f"📦 Batch size: {self.model_config.batch_size}")
        print(f"🧠 LSTM units: {self.model_config.lstm_units}")
        print(f"📏 Sequence length: {self.model_config.sequence_length}")
        print(f"💧 Dropout rate: {self.model_config.dropout_rate}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def list_available_models(self) -> List[str]:
        """List all available trained models."""
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
            return []
        
        model_files = list(models_dir.glob("*.h5"))
        model_info = []
        
        print("\n📊 MODELOS DISPONIBLES")
        print("=" * 50)
        
        if not model_files:
            print("❌ No se encontraron modelos entrenados")
            print("   Ejecuta FASE 1 primero para entrenar un modelo")
            return []
        
        for i, model_file in enumerate(model_files, 1):
            metadata_file = model_file.with_suffix('.json').name.replace('.json', '_metadata.json')
            metadata_path = models_dir / metadata_file
            
            print(f"\n{i}. 📁 {model_file.name}")
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    print(f"   📊 Vocabulario: {metadata['vocab_size']} caracteres")
                    print(f"   🎯 Épocas: {metadata['epochs_trained']}")
                    print(f"   📉 Loss final: {metadata['final_loss']:.4f}")
                    print(f"   ⏱️  Tiempo: {metadata['training_time']}")
                except:
                    print(f"   ⚠️  Sin metadata disponible")
            else:
                print(f"   ⚠️  Sin metadata disponible")
            
            model_info.append(str(model_file))
        
        return model_info
    
    def monitor_training_progress(self):
        """Monitor current training progress."""
        print("\n📈 MONITOR DE PROGRESO")
        print("=" * 50)
        
        logs_dir = Path("logs")
        models_dir = Path("models")
        
        # Check for active training
        checkpoints = list(models_dir.glob("checkpoint_*.h5")) if models_dir.exists() else []
        
        if checkpoints:
            print(f"🔄 Checkpoints encontrados: {len(checkpoints)}")
            
            # Show latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            mod_time = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
            
            print(f"📅 Último checkpoint: {latest_checkpoint.name}")
            print(f"🕐 Modificado: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check if training might be active
            time_diff = datetime.now() - mod_time
            if time_diff.total_seconds() < 300:  # Less than 5 minutes
                print(f"🟢 Estado: Posiblemente entrenando (actualizado hace {time_diff.total_seconds():.0f}s)")
            else:
                print(f"🟡 Estado: Inactivo (última actualización hace {time_diff})")
        else:
            print(f"❌ No hay entrenamientos en progreso")
        
        # TensorBoard logs
        if logs_dir.exists():
            log_files = list(logs_dir.rglob("*"))
            if log_files:
                print(f"\n📊 TensorBoard logs disponibles: {len(log_files)} archivos")
                print(f"   Para ver gráficos: tensorboard --logdir logs")
        
        input("\n📖 Presiona Enter para continuar...")
        
    def phase1_intensive_training(self):
        """Execute Phase 1: Intensive Training."""
        print("\n🔥 FASE 1: ENTRENAMIENTO INTENSIVO")
        print("=" * 50)
        
        # Get training parameters
        text_file = self.get_text_file_input()
        epochs = self.get_epochs_input()
        
        # Confirm intensive training
        print(f"\n⚠️  CONFIRMACIÓN DE ENTRENAMIENTO INTENSIVO")
        print(f"📁 Archivo: {text_file}")
        print(f"🎯 Épocas: {epochs} (~1+ hora)")
        print(f"💻 Device: {GPUConfigurator.get_device_strategy()}")
        
        confirm = input("\n🤔 ¿Continuar con entrenamiento? (s/N): ").strip().lower()
        if confirm not in ['s', 'si', 'y', 'yes']:
            print("❌ Entrenamiento cancelado")
            return
        
        try:
            # Setup environment
            print("\n🚀 Configurando entorno...")
            gpu_available = GPUConfigurator.setup_gpu()
            device = GPUConfigurator.get_device_strategy()
            
            # Create directories
            Path("models").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            
            # Prepare data
            print("\n📚 Preparando datos...")
            processor = TextProcessor(
                sequence_length=self.model_config.sequence_length,
                step_size=self.model_config.step_size
            )
            
            X, y = processor.prepare_data(
                text_file,
                max_length=self.model_config.max_text_length
            )
            
            print(f"✅ Datos preparados: {len(X):,} secuencias")
            
            # Build model
            print("\n🧠 Construyendo modelo...")
            lstm_generator = LSTMTextGenerator(
                vocab_size=processor.vocab_size,
                sequence_length=self.model_config.sequence_length,
                lstm_units=self.model_config.lstm_units,
                dropout_rate=self.model_config.dropout_rate
            )
            
            model = lstm_generator.build_model()
            
            # Train model
            print(f"\n⚡ INICIANDO ENTRENAMIENTO INTENSIVO...")
            print(f"   Tiempo estimado: ~{epochs * 2} minutos")
            
            trainer = ModelTrainer(model, device)
            
            start_time = datetime.now()
            history = trainer.train(
                X, y,
                batch_size=self.model_config.batch_size,
                epochs=epochs,
                validation_split=self.model_config.validation_split
            )
            end_time = datetime.now()
            
            # Save model
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            model_path = f"models/robo_poet_model_{timestamp}.h5"
            
            ModelManager.save_model(model, model_path)
            
            # Save metadata
            metadata = {
                'vocab_size': processor.vocab_size,
                'sequence_length': self.model_config.sequence_length,
                'lstm_units': self.model_config.lstm_units,
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['loss']),
                'training_time': str(end_time - start_time),
                'char_to_idx': processor.char_to_idx,
                'idx_to_char': processor.idx_to_char
            }
            
            metadata_path = model_path.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
            print(f"   ⏱️  Duración: {end_time - start_time}")
            print(f"   💾 Modelo guardado: {model_path}")
            print(f"   📊 Loss final: {metadata['final_loss']:.4f}")
            print(f"   ✅ Listo para FASE 2")
            
        except Exception as e:
            print(f"\n❌ Error en entrenamiento: {e}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def phase2_text_generation(self):
        """Execute Phase 2: Text Generation."""
        print("\n🎨 FASE 2: GENERACIÓN DE TEXTO")
        print("=" * 50)
        
        # List available models
        models = self.list_available_models()
        if not models:
            input("\n📖 Presiona Enter para continuar...")
            return
        
        # Select model
        while True:
            try:
                choice = input(f"\n🤔 Selecciona modelo (1-{len(models)}) o 'c' para cancelar: ").strip()
                if choice.lower() == 'c':
                    return
                
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(models):
                    selected_model = models[model_idx]
                    break
                else:
                    print(f"❌ Número inválido. Usa 1-{len(models)}")
            except ValueError:
                print("❌ Ingresa un número válido")
        
        try:
            # Load model
            print(f"\n📁 Cargando modelo: {Path(selected_model).name}")
            model = ModelManager.load_model(selected_model)
            
            # Load metadata
            metadata_path = selected_model.replace('.h5', '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            char_to_idx = metadata['char_to_idx']
            idx_to_char = metadata['idx_to_char']
            
            print(f"✅ Modelo cargado exitosamente")
            
            # Generation menu
            self.generation_menu(model, char_to_idx, idx_to_char)
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            input("\n📖 Presiona Enter para continuar...")
    
    def generation_menu(self, model, char_to_idx: dict, idx_to_char: dict):
        """Submenu for text generation options."""
        generator = TextGenerator(model, char_to_idx, idx_to_char)
        
        while True:
            print("\n🎨 OPCIONES DE GENERACIÓN")
            print("-" * 30)
            print("1. 📝 Generación simple")
            print("2. 🎮 Modo interactivo")
            print("3. 📊 Generación en lote")
            print("4. 🔙 Volver al menú principal")
            
            choice = input("\n🎯 Selecciona opción (1-4): ").strip()
            
            if choice == '1':
                self.simple_generation(generator)
            elif choice == '2':
                self.interactive_generation(generator)
            elif choice == '3':
                self.batch_generation(generator)
            elif choice == '4':
                break
            else:
                print("❌ Opción inválida")
    
    def simple_generation(self, generator):
        """Single text generation."""
        seed = input("\n🌱 Ingresa seed text: ").strip()
        if not seed:
            print("❌ Seed vacío")
            return
        
        length = self.get_number_input("📏 Longitud (default 200): ", 200)
        temperature = self.get_float_input("🌡️  Temperature (default 0.8): ", 0.8)
        
        print(f"\n🎨 Generando...")
        result = generator.generate(seed, length, temperature)
        
        print(f"\n📝 RESULTADO:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
        input("\n📖 Presiona Enter para continuar...")
    
    def interactive_generation(self, generator):
        """Interactive generation mode."""
        print("\n🎮 MODO INTERACTIVO")
        print("   'exit' para salir, 'temp X' para cambiar temperature")
        
        temperature = 0.8
        length = 200
        
        while True:
            try:
                user_input = input(f"\n🌱 Seed (T={temperature}, L={length}): ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'salir']:
                    break
                
                if user_input.startswith('temp '):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"🌡️  Temperature: {temperature}")
                        continue
                    except:
                        print("❌ Uso: temp 0.8")
                        continue
                
                if user_input.startswith('len '):
                    try:
                        length = int(user_input.split()[1])
                        print(f"📏 Length: {length}")
                        continue
                    except:
                        print("❌ Uso: len 200")
                        continue
                
                if not user_input:
                    print("⚠️  Ingresa un seed")
                    continue
                
                result = generator.generate(user_input, length, temperature)
                print(f"\n📝 {result}\n")
                
            except KeyboardInterrupt:
                break
    
    def batch_generation(self, generator):
        """Batch text generation."""
        seeds_input = input("\n🌱 Seeds separados por comas: ").strip()
        if not seeds_input:
            print("❌ No se ingresaron seeds")
            return
        
        seeds = [s.strip() for s in seeds_input.split(',')]
        length = self.get_number_input("📏 Longitud por seed (default 150): ", 150)
        temperature = self.get_float_input("🌡️  Temperature (default 0.8): ", 0.8)
        
        print(f"\n📊 GENERACIÓN EN LOTE")
        print(f"   🌱 {len(seeds)} seeds")
        print("=" * 50)
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n{i}. Seed: '{seed}'")
            print("-" * 30)
            result = generator.generate(seed, length, temperature)
            print(result)
        
        input("\n📖 Presiona Enter para continuar...")
    
    def get_text_file_input(self) -> str:
        """Get text file input from user."""
        default_file = "The+48+Laws+Of+Power_texto.txt"
        
        print(f"\n📁 ARCHIVO DE ENTRENAMIENTO")
        file_input = input(f"   Archivo (Enter para '{default_file}'): ").strip()
        
        text_file = file_input if file_input else default_file
        
        if not Path(text_file).exists():
            print(f"❌ Archivo no encontrado: {text_file}")
            raise FileNotFoundError(f"No se encontró el archivo: {text_file}")
        
        return text_file
    
    def get_epochs_input(self) -> int:
        """Get epochs input from user."""
        while True:
            try:
                epochs_input = input("🎯 Épocas (Enter para 50): ").strip()
                if not epochs_input:
                    return 50
                
                epochs = int(epochs_input)
                if epochs < 1:
                    print("❌ Épocas debe ser >= 1")
                    continue
                
                if epochs > 100:
                    confirm = input(f"⚠️  {epochs} épocas es mucho tiempo. ¿Continuar? (s/N): ")
                    if confirm.lower() not in ['s', 'si', 'y', 'yes']:
                        continue
                
                return epochs
                
            except ValueError:
                print("❌ Ingresa un número válido")
    
    def get_number_input(self, prompt: str, default: int) -> int:
        """Get number input with default."""
        try:
            value = input(prompt).strip()
            return int(value) if value else default
        except ValueError:
            return default
    
    def get_float_input(self, prompt: str, default: float) -> float:
        """Get float input with default."""
        try:
            value = input(prompt).strip()
            return float(value) if value else default
        except ValueError:
            return default
    
    def run_interface(self):
        """Run the main academic interface."""
        while True:
            try:
                self.show_header()
                choice = self.show_main_menu()
                
                if choice == '1':
                    self.phase1_intensive_training()
                elif choice == '2':
                    self.phase2_text_generation()
                elif choice == '3':
                    self.list_available_models()
                    input("\n📖 Presiona Enter para continuar...")
                elif choice == '4':
                    self.monitor_training_progress()
                elif choice == '5':
                    self.show_system_status()
                elif choice == '6':
                    print("\n👋 ¡Hasta luego!")
                    break
                
                # Clear screen for next iteration
                os.system('clear' if os.name == 'posix' else 'cls')
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error inesperado: {e}")
                input("📖 Presiona Enter para continuar...")
    

def main():
    """Main execution function."""
    # Initialize and run academic interface
    interface = AcademicInterface()
    interface.run_interface()

if __name__ == "__main__":
    main()