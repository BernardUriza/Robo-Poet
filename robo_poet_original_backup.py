#!/usr/bin/env python3
"""
Robo-Poet: Academic Neural Text Generation Framework

Interfaz académica unificada para entrenamiento y generación de texto con LSTM.
Sistema de dos fases: entrenamiento intensivo y generación pura.

Author: Student ML Researcher
Version: 2.0.1 - WSL2 GPU Detection Solution
Hardware: Optimized for NVIDIA RTX 2000 Ada
Platform: WSL2 + Kali Linux + Windows 11

TECHNICAL NOTE: WSL2 GPU Detection Solution
===========================================
This framework implements a robust GPU detection system specifically designed
for WSL2 environments where standard TensorFlow GPU detection often fails.

Problem: tf.config.list_physical_devices('GPU') returns empty list in WSL2
Solution: Direct GPU operation test bypasses detection issues
Result: Full GPU acceleration in WSL2 without manual configuration

Key Features:
- Multi-strategy GPU detection (standard + direct access)
- Automatic CUDA library environment configuration  
- Graceful degradation to CPU mode when needed
- Zero-configuration setup for end users
"""

# CRITICAL: Configure GPU environment FIRST, before any imports
import os
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if conda_prefix:
    os.environ['CUDA_HOME'] = conda_prefix
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # FORCE correct library paths, don't append potentially wrong existing ones
    lib_paths = [f'{conda_prefix}/lib', f'{conda_prefix}/lib64']
    # Only append system paths, not potentially wrong conda paths
    system_paths = ['/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu']
    lib_paths.extend(system_paths)
    clean_ld = ':'.join(lib_paths)
    os.environ['LD_LIBRARY_PATH'] = clean_ld
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# Environment already configured at the top of the file
print(f"🔧 GPU environment FORZADO para: {conda_prefix}" if conda_prefix else "⚠️ CONDA_PREFIX no encontrado")
if conda_prefix:
    print(f"🔧 CUDA_HOME forzado a: {os.environ.get('CUDA_HOME', 'NO_SET')}")
    print(f"🔧 LD_LIBRARY_PATH limpiado y reconfigurado")
    print(f"🔧 Nuevas rutas: {os.environ.get('LD_LIBRARY_PATH', '')[:120]}...")

# CRITICAL: WSL2 GPU Detection - Robust multi-strategy approach
def detect_gpu_for_wsl2():
    """
    Advanced GPU detection specifically designed for WSL2 environments.
    
    WSL2 has known issues with standard TensorFlow GPU detection where:
    1. tf.config.list_physical_devices('GPU') returns empty list
    2. But direct GPU operations work perfectly
    3. This affects NVIDIA drivers in WSL2 environments
    
    Returns:
        tuple: (gpu_available: bool, tf_module: module)
    """
    import os
    import time
    
    print("🔧 Iniciando detección de GPU optimizada para WSL2...")
    
    # Ensure optimal logging for diagnosis
    original_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    
    try:
        import tensorflow as tf
        
        # Strategy 1: Standard TensorFlow detection
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            print(f"✅ GPU detectada vía método estándar: {len(tf_gpus)} GPU(s)")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_log_level
            return True, tf
        
        # Strategy 2: Direct GPU operation test (WSL2 fix)
        print("🔍 Método estándar falló, probando acceso directo GPU (fix WSL2)...")
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                result = tf.reduce_sum(test_tensor)
            
            print("🎯 ¡GPU funciona perfectamente via acceso directo!")
            print("💡 Aplicando workaround WSL2 para usar GPU")
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_log_level
            return True, tf
            
        except Exception as gpu_error:
            print(f"❌ GPU no accesible: {str(gpu_error)[:100]}...")
            if "Cannot dlopen" in str(gpu_error):
                print("💡 Error de librerías CUDA detectado")
                print("🔧 Instala: conda install -c conda-forge cudnn libcublas libcufft libcurand libcusolver libcusparse")
    
    except Exception as e:
        print(f"❌ Error importando TensorFlow: {e}")
    
    # Fallback: CPU mode
    try:
        print("🔍 Fallback: Importando TensorFlow en modo CPU...")
        import tensorflow as tf
        print("✅ TensorFlow disponible en modo CPU")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_log_level
        return False, tf
    except Exception as e:
        print(f"❌ Fallo crítico: {e}")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_log_level
        return False, None

# Execute GPU detection
gpu_available, tf = detect_gpu_for_wsl2()

# Only import our modules if GPU validation passes OR if we're using direct training
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--text', help='Text file for training')
parser.add_argument('--epochs', type=int, help='Number of training epochs')
args, unknown = parser.parse_known_args()

# Add src to path for imports
sys.path.append('src')

# Now import modules (TensorFlow already initialized properly)
print("🔧 Importando módulos del sistema...")
from config import get_config, GPUConfigurator
if gpu_available:
    from data_processor import TextProcessor, TextGenerator
    from model import LSTMTextGenerator, ModelTrainer, ModelManager
    print("✅ Todos los módulos GPU importados correctamente")
else:
    print("⚠️ Módulos GPU limitados debido a falta de GPU")

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
        print("5. 🧹 Limpiar Todos los Modelos")
        print("6. ⚙️  Configuración del Sistema")
        print("7. 🚪 Salir")
        print("-" * 50)
        
        while True:
            try:
                choice = input("🎯 Selecciona una opción (1-7): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6', '7']:
                    return choice
                else:
                    print("❌ Opción inválida. Ingresa un número del 1 al 7.")
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
        
        if not gpu_available:
            print("\n🚨 AVISO ACADÉMICO:")
            print("   GPU NVIDIA es obligatoria para entrenamiento.")
            print("   Comando directo funciona: python robo_poet.py --text archivo.txt --epochs N")
        
        print(f"📦 Batch size: {self.model_config.batch_size}")
        print(f"🧠 LSTM units: {self.model_config.lstm_units}")
        print(f"📏 Sequence length: {self.model_config.sequence_length}")
        print(f"💧 Dropout rate: {self.model_config.dropout_rate}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def clean_all_models(self):
        """Clean all trained models with confirmation."""
        print("\n🧹 LIMPIAR TODOS LOS MODELOS")
        print("=" * 50)
        
        models_dir = Path("models")
        if not models_dir.exists():
            print("📁 El directorio de modelos no existe")
            input("\n📖 Presiona Enter para continuar...")
            return
        
        # Count existing models
        model_files = list(models_dir.glob("*.keras"))
        metadata_files = list(models_dir.glob("*_metadata.json"))
        checkpoint_files = list(models_dir.glob("checkpoint_*.keras"))
        
        total_files = len(model_files) + len(metadata_files) + len(checkpoint_files)
        
        if total_files == 0:
            print("✅ No hay modelos para limpiar")
            input("\n📖 Presiona Enter para continuar...")
            return
        
        print(f"📊 ARCHIVOS ENCONTRADOS:")
        print(f"   🎯 Modelos finales: {len(model_files)}")
        print(f"   📋 Archivos metadata: {len(metadata_files)}")
        print(f"   🔄 Checkpoints: {len(checkpoint_files)}")
        print(f"   📦 Total: {total_files} archivos")
        
        # Calculate total size
        total_size = 0
        for file_list in [model_files, metadata_files, checkpoint_files]:
            for file_path in file_list:
                if file_path.exists():
                    total_size += file_path.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        print(f"   💾 Espacio a liberar: {size_mb:.1f} MB")
        
        print(f"\n⚠️  CONFIRMACIÓN DE LIMPIEZA")
        print(f"Esta acción eliminará PERMANENTEMENTE todos los modelos entrenados.")
        print(f"No podrás usar FASE 2 (Generación) hasta entrenar nuevos modelos.")
        
        confirm = input(f"\n🗑️  ¿Confirmar limpieza de {total_files} archivos? (escribe 'ELIMINAR' para confirmar): ").strip()
        
        if confirm != 'ELIMINAR':
            print("❌ Limpieza cancelada")
            input("\n📖 Presiona Enter para continuar...")
            return
        
        # Perform cleanup
        print(f"\n🗑️  ELIMINANDO ARCHIVOS...")
        deleted_count = 0
        
        try:
            # Delete model files
            for model_file in model_files:
                if model_file.exists():
                    model_file.unlink()
                    deleted_count += 1
                    print(f"   ✅ {model_file.name}")
            
            # Delete metadata files
            for metadata_file in metadata_files:
                if metadata_file.exists():
                    metadata_file.unlink()
                    deleted_count += 1
                    print(f"   ✅ {metadata_file.name}")
            
            # Delete checkpoint files
            for checkpoint_file in checkpoint_files:
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    deleted_count += 1
                    print(f"   ✅ {checkpoint_file.name}")
            
            print(f"\n🎉 LIMPIEZA COMPLETADA")
            print(f"   🗑️  Archivos eliminados: {deleted_count}")
            print(f"   💾 Espacio liberado: {size_mb:.1f} MB")
            print(f"   📁 Directorio: models/")
            
        except Exception as e:
            print(f"\n❌ Error durante limpieza: {e}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def list_available_models(self) -> List[str]:
        """List all available trained models."""
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
            return []
        
        model_files = list(models_dir.glob("*.keras"))
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
    
    def list_available_models_enhanced(self) -> List[str]:
        """Enhanced model listing with detailed information - ONLY complete models with metadata."""
        models_dir = Path("models")
        if not models_dir.exists():
            models_dir.mkdir(exist_ok=True)
            return []
        
        # Filter ONLY models with metadata (complete models)
        all_model_files = list(models_dir.glob("*.keras"))
        model_files = []
        
        for model_file in all_model_files:
            metadata_file = model_file.with_suffix('.json').name.replace('.json', '_metadata.json')
            metadata_path = models_dir / metadata_file
            if metadata_path.exists():
                model_files.append(model_file)
        
        print("\n📊 MODELOS COMPLETOS DISPONIBLES")
        print("=" * 60)
        
        if not model_files:
            print("❌ No se encontraron modelos completos con metadata")
            print("   Ejecuta FASE 1 primero para entrenar un modelo completo")
            return []
        
        model_info = []
        for i, model_file in enumerate(model_files, 1):
            metadata_file = model_file.with_suffix('.json').name.replace('.json', '_metadata.json')
            metadata_path = models_dir / metadata_file
            
            print(f"\n🏷️  MODELO COMPLETO #{i}")
            print(f"   📁 Archivo: {model_file.name}")
            print(f"   📊 Tamaño: {model_file.stat().st_size / (1024*1024):.1f} MB")
            print(f"   📅 Creado: {datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
            
            # We know metadata exists because we filtered for it
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"   🧠 Arquitectura: LSTM {metadata.get('lstm_units', 'N/A')} units")
                print(f"   📝 Vocabulario: {metadata.get('vocab_size', 'N/A')} caracteres")
                print(f"   🎯 Épocas: {metadata.get('epochs_trained', 'N/A')}")
                print(f"   📉 Loss final: {metadata.get('final_loss', 0):.4f}")
                print(f"   ⏱️  Tiempo entrenamiento: {metadata.get('training_time', 'N/A')}")
                
                # Performance indicator
                loss = metadata.get('final_loss', 999)
                if loss < 1.0:
                    print("   🌟 Estado: ⭐⭐⭐ Excelente")
                elif loss < 2.0:
                    print("   🌟 Estado: ⭐⭐ Bueno")
                else:
                    print("   🌟 Estado: ⭐ Básico")
                    
            except Exception as e:
                # This shouldn't happen since we filtered for metadata existence
                print(f"   ⚠️  Error leyendo metadata: {str(e)[:50]}...")
                continue  # Skip this model
            
            model_info.append(str(model_file))
            print("-" * 40)
        
        return model_info
    
    def select_model_enhanced(self, models):
        """Enhanced model selection with preview."""
        while True:
            try:
                print(f"\n🎯 SELECCIÓN DE MODELO")
                choice = input(f"🤔 Selecciona modelo (1-{len(models)}) o 'c' para cancelar: ").strip()
                if choice.lower() == 'c':
                    return None, None
                
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(models):
                    selected_model = models[model_idx]
                    
                    # Load metadata for preview
                    metadata_path = selected_model.replace('.keras', '_metadata.json')
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        print(f"\n✅ Modelo seleccionado: {Path(selected_model).name}")
                        print(f"   🎯 Precisión esperada: {100 - metadata.get('final_loss', 2.0) * 30:.1f}%")
                        print(f"   🧠 Complejidad: {metadata.get('lstm_units', 128)} unidades LSTM")
                        
                        return selected_model, metadata
                    except:
                        print(f"⚠️  Usando modelo sin metadata completa")
                        return selected_model, {}
                else:
                    print(f"❌ Número inválido. Usa 1-{len(models)}")
            except ValueError:
                print("❌ Ingresa un número válido")
    
    def monitor_training_progress(self):
        """Monitor current training progress."""
        print("\n📈 MONITOR DE PROGRESO")
        print("=" * 50)
        
        logs_dir = Path("logs")
        models_dir = Path("models")
        
        # Check for active training
        checkpoints = list(models_dir.glob("checkpoint_*.keras")) if models_dir.exists() else []
        
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
        print("\n🔥 FASE 1: ENTRENAMIENTO INTENSIVO - GPU OBLIGATORIA")
        print("=" * 50)
        
        # Check if GPU validation passed during startup
        if not gpu_available:
            print("\n❌ ENTRENAMIENTO NO DISPONIBLE")
            print("🚨 GPU no fue detectada durante las estrategias de inicialización")
            print("\n🔍 DIAGNÓSTICO RÁPIDO:")
            print("   Ejecuta estos comandos para verificar:")
            print("   1. nvidia-smi                    # ¿GPU visible?")
            print("   2. echo $CONDA_PREFIX            # ¿Entorno correcto?")
            print("   3. python -c 'import tensorflow as tf; print(tf.config.list_physical_devices())'")
            print("\n🔧 ALTERNATIVAS:")
            print("   • Comando directo: python robo_poet.py --text 'archivo.txt' --epochs N")
            print("   • Reiniciar terminal y reactivar: conda activate robo-poet-gpu")
            print("   • Verificar drivers: nvidia-smi")
            input("\n📖 Presiona Enter para volver al menú...")
            return
        
        # Get training parameters
        text_file = self.get_text_file_input()
        epochs = self.get_epochs_input()
        
        # Confirm intensive training
        print(f"\n⚠️  CONFIRMACIÓN DE ENTRENAMIENTO INTENSIVO")
        print(f"📁 Archivo: {text_file}")
        print(f"🎯 Épocas: {epochs}")
        print(f"⏱️  Tiempo estimado: ~{epochs * 2} minutos")
        
        confirm = input("\n🚀 ¿Confirmar entrenamiento? (s/N): ").strip().lower()
        if confirm not in ['s', 'si', 'sí', 'yes', 'y']:
            print("❌ Entrenamiento cancelado")
            input("\n📖 Presiona Enter para volver al menú...")
            return
        
        print(f"\n🚀 INICIANDO ENTRENAMIENTO DESDE INTERFAZ ACADÉMICA...")
        print("=" * 60)
        
        # Use subprocess to call the direct training method that works
        import subprocess
        import sys
        import os
        
        try:
            print(f"🔧 Iniciando proceso limpio para entrenamiento GPU...")
            print(f"🔧 Archivo: {text_file}, Épocas: {epochs}")
            
            # Use a completely fresh Python process to avoid TensorFlow library conflicts
            script_path = "robo_poet.py"  # Since we're in the same directory
            cmd = [
                sys.executable,  # Use the same Python interpreter 
                script_path,     # Execute this same script
                "--text", text_file,
                "--epochs", str(epochs)
            ]
            
            print(f"🔧 Ejecutando: {' '.join(cmd[:3])} ...")
            
            # Run in a fresh process with proper environment
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print("\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
                print("💾 Modelo guardado automáticamente")
                print("🎨 Ya puedes usar FASE 2 para generar texto")
            else:
                print(f"\n❌ Error en entrenamiento (código: {result.returncode})")
                
        except Exception as e:
            print(f"\n❌ Error ejecutando entrenamiento: {e}")
        
        input("\n📖 Presiona Enter para volver al menú...")
    
    def phase2_text_generation(self):
        """Execute Phase 2: Enhanced Text Generation."""
        print("\n🎨 FASE 2: GENERACIÓN DE TEXTO AVANZADA")
        print("=" * 60)
        
        # List available models with enhanced display
        models = self.list_available_models_enhanced()
        if not models:
            input("\n📖 Presiona Enter para continuar...")
            return
        
        # Enhanced model selection with preview
        selected_model, metadata = self.select_model_enhanced(models)
        if not selected_model:
            return
        
        try:
            # Load model with progress indication
            print(f"\n🔄 Cargando modelo: {Path(selected_model).name}")
            print("   └── Importando arquitectura...")
            model = ModelManager.load_model(selected_model)
            print("   └── Configurando generador...")
            
            char_to_idx = metadata['char_to_idx']
            # Convert string keys back to integers for idx_to_char
            idx_to_char = {int(k): v for k, v in metadata['idx_to_char'].items()}
            
            print("   └── ✅ Modelo listo para generación")
            
            # Enhanced generation interface
            self.enhanced_generation_interface(model, char_to_idx, idx_to_char, metadata)
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            import traceback
            print(f"   Detalles: {traceback.format_exc()[:200]}...")
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
    
    def enhanced_generation_interface(self, model, char_to_idx: dict, idx_to_char: dict, metadata: dict):
        """Enhanced generation interface with advanced features."""
        generator = TextGenerator(model, char_to_idx, idx_to_char)
        
        # Display model statistics
        self.show_model_stats(metadata)
        
        while True:
            print("\n🎨 ESTUDIO DE GENERACIÓN AVANZADO")
            print("=" * 50)
            print("🎯 MODOS DE GENERACIÓN:")
            print("1. 🚀 Generación Rápida (presets optimizados)")
            print("2. 🔬 Laboratorio Creativo (control total)")
            print("3. 🎮 Sesión Interactiva (exploración en vivo)")
            print("4. 📊 Generación en Lote (múltiples experimentos)")
            print("5. 🎨 Plantillas Temáticas (estilos predefinidos)")
            print("\n📋 HERRAMIENTAS:")
            print("6. 📈 Estadísticas del Modelo")
            print("7. 💾 Ver Generaciones Guardadas")
            print("8. 🔙 Volver al Menú Principal")
            print("-" * 50)
            
            choice = input("\n🎯 Selecciona opción (1-8): ").strip()
            
            if choice == '1':
                self.quick_generation(generator)
            elif choice == '2':
                self.creative_lab(generator)
            elif choice == '3':
                self.interactive_session(generator)
            elif choice == '4':
                self.batch_experiments(generator)
            elif choice == '5':
                self.thematic_templates(generator)
            elif choice == '6':
                self.detailed_model_stats(metadata)
            elif choice == '7':
                self.view_saved_generations()
            elif choice == '8':
                break
            else:
                print("❌ Opción inválida")
    
    def show_model_stats(self, metadata: dict):
        """Display basic model statistics."""
        print(f"\n📊 ESTADÍSTICAS DEL MODELO")
        print("-" * 30)
        print(f"🧠 Unidades LSTM: {metadata.get('lstm_units', 'N/A')}")
        print(f"📝 Vocabulario: {metadata.get('vocab_size', 'N/A')} caracteres")
        print(f"🎯 Épocas entrenadas: {metadata.get('epochs_trained', 'N/A')}")
        print(f"📉 Loss final: {metadata.get('final_loss', 0):.4f}")
        
        # Performance rating
        loss = metadata.get('final_loss', 999)
        if loss < 1.0:
            rating = "🌟🌟🌟 Excelente"
        elif loss < 2.0:
            rating = "🌟🌟 Bueno"
        else:
            rating = "🌟 Básico"
        print(f"⭐ Calidad: {rating}")
    
    def quick_generation(self, generator):
        """Quick generation with optimized presets."""
        print("\n🚀 GENERACIÓN RÁPIDA")
        print("=" * 40)
        
        # Preset options
        presets = {
            '1': {'name': '📖 Narrativa (Conservador)', 'temp': 0.6, 'length': 300},
            '2': {'name': '✨ Creativo (Balanceado)', 'temp': 0.8, 'length': 250},
            '3': {'name': '🎲 Experimental (Aleatorio)', 'temp': 1.2, 'length': 200},
            '4': {'name': '📝 Académico (Formal)', 'temp': 0.5, 'length': 350},
            '5': {'name': '🎨 Artístico (Expresivo)', 'temp': 1.0, 'length': 280}
        }
        
        print("\n🎛️  PRESETS DISPONIBLES:")
        for key, preset in presets.items():
            print(f"{key}. {preset['name']} (T={preset['temp']}, L={preset['length']})")
        
        preset_choice = input("\n🎯 Selecciona preset (1-5) o 'c' para personalizar: ").strip()
        
        if preset_choice == 'c':
            temperature = self.get_float_input("🌡️  Temperature (0.1-2.0): ", 0.8)
            length = self.get_number_input("📏 Longitud (50-500): ", 200)
        elif preset_choice in presets:
            preset = presets[preset_choice]
            temperature = preset['temp']
            length = preset['length']
            print(f"\n✅ Preset seleccionado: {preset['name']}")
        else:
            print("❌ Preset inválido, usando valores por defecto")
            temperature, length = 0.8, 200
        
        # Suggested seeds
        seed_suggestions = [
            "The power of", "In the beginning", "Once upon a time", 
            "The secret to", "It was a dark", "The greatest",
            "Behind the scenes", "The art of", "In this world"
        ]
        
        print(f"\n🌱 SUGERENCIAS DE SEED:")
        for i, suggestion in enumerate(seed_suggestions[:6], 1):
            print(f"{i}. \"{suggestion}\"")
        
        seed_input = input("\n🌱 Ingresa seed (o número 1-6 para sugerencia): ").strip()
        
        if seed_input.isdigit() and 1 <= int(seed_input) <= 6:
            seed = seed_suggestions[int(seed_input) - 1]
            print(f"✅ Seed seleccionado: \"{seed}\"")
        else:
            seed = seed_input if seed_input else "The power of"
        
        # Generate with progress
        print(f"\n🎨 Generando texto...")
        print(f"   🎛️  Configuración: T={temperature}, L={length}")
        print(f"   🌱 Seed: \"{seed}\"")
        print("   ⏳ Procesando...")
        
        start_time = time.time()
        result = generator.generate(seed, length, temperature)
        generation_time = time.time() - start_time
        
        # Enhanced result display
        self.display_generation_result(result, seed, temperature, length, generation_time)
        
        # Save option
        self.offer_save_generation(result, seed, temperature, length)
    
    def display_generation_result(self, result: str, seed: str, temperature: float, length: int, gen_time: float):
        """Display generation result with metadata."""
        print(f"\n🎨 RESULTADO GENERADO")
        print("=" * 60)
        print(f"📊 Metadata: {len(result)} chars | {gen_time:.2f}s | T={temperature} | Seed: \"{seed}\"")
        print("-" * 60)
        print(result)
        print("-" * 60)
        print(f"⚡ Velocidad: {len(result)/gen_time:.0f} chars/segundo")
        
        # Text analysis
        words = len(result.split())
        sentences = result.count('.') + result.count('!') + result.count('?')
        print(f"📈 Análisis: {words} palabras, ~{sentences} oraciones")
    
    def offer_save_generation(self, result: str, seed: str, temperature: float, length: int):
        """Offer to save the generation to file."""
        save_choice = input("\n💾 ¿Guardar esta generación? (s/N): ").strip().lower()
        if save_choice in ['s', 'si', 'sí', 'y', 'yes']:
            self.save_generation_to_file(result, seed, temperature, length)
    
    def save_generation_to_file(self, result: str, seed: str, temperature: float, length: int):
        """Save generation to timestamped file."""
        try:
            # Create generations directory
            gen_dir = Path("generations")
            gen_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generation_{timestamp}.txt"
            filepath = gen_dir / filename
            
            # Save with metadata
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Robo-Poet Generation\n")
                f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Seed: \"{seed}\"\n")
                f.write(f"# Temperature: {temperature}\n")
                f.write(f"# Length: {length}\n")
                f.write(f"# Generated chars: {len(result)}\n")
                f.write("# " + "="*50 + "\n\n")
                f.write(result)
            
            print(f"✅ Guardado en: {filepath}")
            
        except Exception as e:
            print(f"❌ Error guardando archivo: {e}")
    
    def creative_lab(self, generator):
        """Advanced creative laboratory with full control."""
        print("\n🔬 LABORATORIO CREATIVO")
        print("=" * 40)
        print("🎛️  Control total sobre la generación")
        
        # Advanced parameters
        seed = input("\n🌱 Seed text: ").strip() or "The art of"
        temperature = self.get_float_input("🌡️  Temperature (0.1-2.0): ", 0.8)
        length = self.get_number_input("📏 Longitud (10-1000): ", 200)
        
        # Additional creative options
        print("\n🎨 OPCIONES CREATIVAS:")
        print("1. 🌊 Flujo normal")
        print("2. 🎯 Generación dirigida (múltiples intentos)")
        print("3. 🔄 Variaciones del mismo seed")
        
        mode = input("\n🎯 Selecciona modo (1-3): ").strip()
        
        if mode == '2':
            attempts = self.get_number_input("🔄 Número de intentos (2-5): ", 3)
            print(f"\n🎲 Generando {attempts} variaciones...")
            
            for i in range(attempts):
                print(f"\n--- INTENTO {i+1} ---")
                result = generator.generate(seed, length, temperature)
                print(result[:150] + "..." if len(result) > 150 else result)
            
            choice = input(f"\n🤔 ¿Generar alguna completa? (1-{attempts} o 'n'): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= attempts:
                result = generator.generate(seed, length, temperature)
                self.display_generation_result(result, seed, temperature, length, 0)
                self.offer_save_generation(result, seed, temperature, length)
                
        elif mode == '3':
            variations = ["", " and", " but", " so", " yet"]
            print(f"\n🔄 Generando variaciones de \"{seed}\"...")
            
            for i, variation in enumerate(variations[:3], 1):
                varied_seed = seed + variation
                print(f"\n--- VARIACIÓN {i}: \"{varied_seed}\" ---")
                result = generator.generate(varied_seed, length//2, temperature)
                print(result)
                
        else:
            # Normal flow
            start_time = time.time()
            result = generator.generate(seed, length, temperature)
            gen_time = time.time() - start_time
            
            self.display_generation_result(result, seed, temperature, length, gen_time)
            self.offer_save_generation(result, seed, temperature, length)
        
        input("\n📖 Presiona Enter para continuar...")
    
    def interactive_session(self, generator):
        """Enhanced interactive generation session."""
        print("\n🎮 SESIÓN INTERACTIVA AVANZADA")
        print("=" * 50)
        print("🎛️  COMANDOS DISPONIBLES:")
        print("   • temp X.X      - Cambiar temperatura (0.1-2.0)")
        print("   • len XXX       - Cambiar longitud (50-500)")
        print("   • save         - Guardar última generación")
        print("   • stats        - Ver estadísticas de sesión")
        print("   • clear        - Limpiar historial")
        print("   • help         - Mostrar ayuda")
        print("   • exit         - Salir de la sesión")
        print("-" * 50)
        
        temperature = 0.8
        length = 200
        session_history = []
        generation_count = 0
        total_chars = 0
        
        while True:
            try:
                # Enhanced prompt with statistics
                prompt = f"\n🎨 [{generation_count} gens] [T={temperature}] [L={length}] Seed: "
                user_input = input(prompt).strip()
                
                # Command processing
                if user_input.lower() in ['exit', 'quit', 'salir']:
                    if session_history:
                        print(f"\n📊 RESUMEN DE SESIÓN:")
                        print(f"   🎯 Generaciones: {generation_count}")
                        print(f"   📝 Caracteres totales: {total_chars:,}")
                        print(f"   📈 Promedio por generación: {total_chars//max(1,generation_count):,} chars")
                    break
                
                elif user_input.startswith('temp '):
                    try:
                        new_temp = float(user_input.split()[1])
                        if 0.1 <= new_temp <= 2.0:
                            temperature = new_temp
                            print(f"✅ Temperature: {temperature}")
                        else:
                            print("❌ Temperature debe estar entre 0.1 y 2.0")
                    except:
                        print("❌ Uso: temp 0.8")
                    continue
                
                elif user_input.startswith('len '):
                    try:
                        new_len = int(user_input.split()[1])
                        if 50 <= new_len <= 500:
                            length = new_len
                            print(f"✅ Longitud: {length}")
                        else:
                            print("❌ Longitud debe estar entre 50 y 500")
                    except:
                        print("❌ Uso: len 200")
                    continue
                
                elif user_input.lower() == 'save':
                    if session_history:
                        last_gen = session_history[-1]
                        self.save_generation_to_file(last_gen['result'], last_gen['seed'], 
                                                    last_gen['temperature'], last_gen['length'])
                    else:
                        print("❌ No hay generaciones para guardar")
                    continue
                
                elif user_input.lower() == 'stats':
                    self.show_session_stats(session_history, generation_count, total_chars)
                    continue
                
                elif user_input.lower() == 'clear':
                    session_history.clear()
                    generation_count = 0
                    total_chars = 0
                    print("✅ Historial limpiado")
                    continue
                
                elif user_input.lower() == 'help':
                    self.show_interactive_help()
                    continue
                
                elif not user_input:
                    print("⚠️  Ingresa un seed o comando")
                    continue
                
                # Generate text
                start_time = time.time()
                result = generator.generate(user_input, length, temperature)
                gen_time = time.time() - start_time
                
                # Update statistics
                generation_count += 1
                total_chars += len(result)
                
                # Store in history
                session_history.append({
                    'seed': user_input,
                    'result': result,
                    'temperature': temperature,
                    'length': length,
                    'time': gen_time,
                    'timestamp': time.strftime('%H:%M:%S')
                })
                
                # Display result with metadata
                print(f"\n📝 RESULTADO #{generation_count} ({len(result)} chars, {gen_time:.1f}s):")
                print("-" * 50)
                print(result)
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Sesión interrumpida")
                break
    
    def show_session_stats(self, history, count, total_chars):
        """Show interactive session statistics."""
        if not history:
            print("📊 No hay estadísticas disponibles")
            return
            
        print(f"\n📊 ESTADÍSTICAS DE SESIÓN")
        print("-" * 30)
        print(f"🎯 Total generaciones: {count}")
        print(f"📝 Caracteres totales: {total_chars:,}")
        print(f"📈 Promedio por gen: {total_chars//max(1,count):,} chars")
        
        if len(history) >= 3:
            recent = history[-3:]
            avg_temp = sum(g['temperature'] for g in recent) / len(recent)
            avg_time = sum(g['time'] for g in recent) / len(recent)
            print(f"🌡️  Temperature promedio (últimas 3): {avg_temp:.2f}")
            print(f"⏱️  Tiempo promedio (últimas 3): {avg_time:.2f}s")
    
    def show_interactive_help(self):
        """Show interactive mode help."""
        print(f"\n🆘 AYUDA - MODO INTERACTIVO")
        print("-" * 40)
        print("📝 GENERACIÓN:")
        print("   Escribe cualquier texto como seed para generar")
        print("\n⚙️  CONFIGURACIÓN:")
        print("   temp 0.5   - Más conservador, repetitivo")
        print("   temp 1.0   - Balanceado, creativo")
        print("   temp 1.5   - Muy experimental, impredecible")
        print("   len 100    - Texto corto")
        print("   len 300    - Texto largo")
        print("\n💾 GESTIÓN:")
        print("   save       - Guarda la última generación")
        print("   stats      - Ver estadísticas de la sesión")
        print("   clear      - Limpia historial de la sesión")
    
    def detailed_model_stats(self, metadata):
        """Show detailed model statistics."""
        print(f"\n📈 ESTADÍSTICAS DETALLADAS DEL MODELO")
        print("=" * 60)
        
        # Basic stats
        print(f"🏗️  ARQUITECTURA:")
        print(f"   🧠 Unidades LSTM: {metadata.get('lstm_units', 'N/A')}")
        print(f"   📏 Longitud de secuencia: {metadata.get('sequence_length', 'N/A')}")
        print(f"   📝 Tamaño de vocabulario: {metadata.get('vocab_size', 'N/A')}")
        
        # Training stats
        print(f"\n📚 ENTRENAMIENTO:")
        print(f"   🎯 Épocas completadas: {metadata.get('epochs_trained', 'N/A')}")
        print(f"   📉 Loss final: {metadata.get('final_loss', 'N/A'):.4f}")
        print(f"   📈 Loss validación: {metadata.get('final_val_loss', 'N/A'):.4f}")
        print(f"   ⏱️  Tiempo total: {metadata.get('training_time', 'N/A')}")
        
        # Performance analysis
        loss = metadata.get('final_loss', 999)
        val_loss = metadata.get('final_val_loss', 999)
        
        print(f"\n📊 ANÁLISIS DE RENDIMIENTO:")
        if loss < 1.0:
            print("   🌟 Calidad: Excelente (⭐⭐⭐)")
            print("   💡 Recomendación: Ideal para generación de calidad")
        elif loss < 2.0:
            print("   🌟 Calidad: Buena (⭐⭐)")
            print("   💡 Recomendación: Adecuado para la mayoría de casos")
        else:
            print("   🌟 Calidad: Básica (⭐)")
            print("   💡 Recomendación: Considerar reentrenamiento")
        
        if abs(loss - val_loss) > 0.5:
            print("   ⚠️  Posible overfitting detectado")
        else:
            print("   ✅ Generalización saludable")
        
        # Usage recommendations
        print(f"\n💡 RECOMENDACIONES DE USO:")
        if loss < 1.5:
            print("   🌡️  Temperature recomendada: 0.6-0.9")
            print("   📏 Longitud óptima: 200-400 caracteres")
        else:
            print("   🌡️  Temperature recomendada: 0.7-1.2")
            print("   📏 Longitud óptima: 100-250 caracteres")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def view_saved_generations(self):
        """View and manage saved generations."""
        gen_dir = Path("generations")
        if not gen_dir.exists():
            print("📁 No hay generaciones guardadas")
            input("\n📖 Presiona Enter para continuar...")
            return
        
        gen_files = list(gen_dir.glob("generation_*.txt"))
        if not gen_files:
            print("📁 No hay generaciones guardadas")
            input("\n📖 Presiona Enter para continuar...")
            return
        
        print(f"\n💾 GENERACIONES GUARDADAS ({len(gen_files)} archivos)")
        print("=" * 50)
        
        # Sort by modification time (newest first)
        gen_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for i, gen_file in enumerate(gen_files[:10], 1):  # Show latest 10
            mod_time = datetime.fromtimestamp(gen_file.stat().st_mtime)
            file_size = gen_file.stat().st_size
            
            print(f"{i}. 📄 {gen_file.name}")
            print(f"   📅 {mod_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   📊 {file_size} bytes")
            
            # Try to read metadata from file
            try:
                with open(gen_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]  # Read first 10 lines for metadata
                    for line in lines:
                        if line.startswith('# Seed:'):
                            print(f"   🌱 {line.strip()[2:]}")
                        elif line.startswith('# Temperature:'):
                            print(f"   🌡️  {line.strip()[2:]}")
                            break
            except:
                pass
            print()
        
        if len(gen_files) > 10:
            print(f"... y {len(gen_files) - 10} archivos más")
        
        view_choice = input(f"\n👀 ¿Ver contenido de algún archivo? (1-{min(10, len(gen_files))} o 'n'): ").strip()
        
        if view_choice.isdigit() and 1 <= int(view_choice) <= min(10, len(gen_files)):
            selected_file = gen_files[int(view_choice) - 1]
            try:
                with open(selected_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"\n📄 CONTENIDO DE: {selected_file.name}")
                print("=" * 60)
                print(content)
                print("=" * 60)
                
            except Exception as e:
                print(f"❌ Error leyendo archivo: {e}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def batch_experiments(self, generator):
        """Advanced batch generation with experiments."""
        print("\n📊 LABORATORIO DE EXPERIMENTOS EN LOTE")
        print("=" * 60)
        
        # Experiment type selection
        print("🧪 TIPOS DE EXPERIMENTO:")
        print("1. 🌱 Múltiples Seeds (mismo parámetro)")
        print("2. 🌡️  Barrido de Temperature (mismo seed)")
        print("3. 📏 Variación de Longitud (mismo seed)")
        print("4. 🎨 Matriz Completa (seeds × temperatures)")
        
        exp_type = input("\n🔬 Selecciona experimento (1-4): ").strip()
        
        if exp_type == '1':
            self.multi_seed_experiment(generator)
        elif exp_type == '2':
            self.temperature_sweep_experiment(generator)
        elif exp_type == '3':
            self.length_variation_experiment(generator)
        elif exp_type == '4':
            self.full_matrix_experiment(generator)
        else:
            print("❌ Tipo de experimento inválido")
            input("\n📖 Presiona Enter para continuar...")
    
    def multi_seed_experiment(self, generator):
        """Multiple seeds with same parameters."""
        print("\n🌱 EXPERIMENTO: MÚLTIPLES SEEDS")
        print("=" * 40)
        
        seeds_input = input("🌱 Seeds separados por comas: ").strip()
        if not seeds_input:
            print("❌ No se ingresaron seeds")
            return
        
        seeds = [s.strip() for s in seeds_input.split(',')]
        temperature = self.get_float_input("🌡️  Temperature fija: ", 0.8)
        length = self.get_number_input("📏 Longitud fija: ", 200)
        
        print(f"\n📊 EJECUTANDO EXPERIMENTO:")
        print(f"   🌱 {len(seeds)} seeds")
        print(f"   🌡️  Temperature: {temperature}")
        print(f"   📏 Longitud: {length}")
        print("=" * 60)
        
        results = []
        for i, seed in enumerate(seeds, 1):
            print(f"\n🧪 EXPERIMENTO {i}/{len(seeds)}: \"{seed}\"")
            print("-" * 40)
            
            start_time = time.time()
            result = generator.generate(seed, length, temperature)
            gen_time = time.time() - start_time
            
            results.append({
                'seed': seed,
                'result': result,
                'time': gen_time,
                'chars': len(result),
                'words': len(result.split())
            })
            
            print(result)
            print(f"⚡ {len(result)} chars en {gen_time:.1f}s")
        
        # Summary statistics
        self.show_experiment_summary(results, "seeds")
        self.offer_save_experiment(results, f"multi_seed_T{temperature}_L{length}")
    
    def temperature_sweep_experiment(self, generator):
        """Temperature sweep with same seed."""
        print("\n🌡️  EXPERIMENTO: BARRIDO DE TEMPERATURE")
        print("=" * 40)
        
        seed = input("🌱 Seed fijo: ").strip() or "The power of"
        length = self.get_number_input("📏 Longitud fija: ", 150)
        
        # Temperature range
        start_temp = self.get_float_input("🌡️  Temperature inicial (default 0.5): ", 0.5)
        end_temp = self.get_float_input("🌡️  Temperature final (default 1.5): ", 1.5)
        steps = self.get_number_input("📈 Número de pasos (default 5): ", 5)
        
        temps = [start_temp + (end_temp - start_temp) * i / (steps - 1) for i in range(steps)]
        
        print(f"\n📊 EJECUTANDO BARRIDO:")
        print(f"   🌱 Seed: \"{seed}\"")
        print(f"   📏 Longitud: {length}")
        print(f"   🌡️  Temperatures: {[f'{t:.2f}' for t in temps]}")
        print("=" * 60)
        
        results = []
        for i, temp in enumerate(temps, 1):
            print(f"\n🌡️  TEMPERATURE {i}/{len(temps)}: {temp:.2f}")
            print("-" * 40)
            
            start_time = time.time()
            result = generator.generate(seed, length, temp)
            gen_time = time.time() - start_time
            
            results.append({
                'temperature': temp,
                'result': result,
                'time': gen_time,
                'chars': len(result),
                'words': len(result.split())
            })
            
            print(result)
            print(f"⚡ {len(result)} chars en {gen_time:.1f}s")
        
        self.show_experiment_summary(results, "temperatures")
        self.offer_save_experiment(results, f"temp_sweep_{seed.replace(' ', '_')}_L{length}")
    
    def length_variation_experiment(self, generator):
        """Length variation with same seed and temperature."""
        print("\n📏 EXPERIMENTO: VARIACIÓN DE LONGITUD")
        print("=" * 40)
        
        seed = input("🌱 Seed fijo: ").strip() or "The art of"
        temperature = self.get_float_input("🌡️  Temperature fija: ", 0.8)
        
        # Length range
        lengths = [100, 150, 200, 300, 400]
        custom = input(f"📏 Usar longitudes por defecto {lengths}? (s/N): ").strip().lower()
        
        if custom not in ['s', 'si', 'sí', 'y', 'yes']:
            lengths_input = input("📏 Longitudes separadas por comas: ").strip()
            if lengths_input:
                try:
                    lengths = [int(x.strip()) for x in lengths_input.split(',')]
                except:
                    print("❌ Error en formato, usando valores por defecto")
        
        print(f"\n📊 EJECUTANDO VARIACIÓN:")
        print(f"   🌱 Seed: \"{seed}\"")
        print(f"   🌡️  Temperature: {temperature}")
        print(f"   📏 Longitudes: {lengths}")
        print("=" * 60)
        
        results = []
        for i, length in enumerate(lengths, 1):
            print(f"\n📏 LONGITUD {i}/{len(lengths)}: {length}")
            print("-" * 40)
            
            start_time = time.time()
            result = generator.generate(seed, length, temperature)
            gen_time = time.time() - start_time
            
            results.append({
                'length': length,
                'result': result,
                'time': gen_time,
                'chars': len(result),
                'words': len(result.split())
            })
            
            print(result)
            print(f"⚡ {len(result)} chars en {gen_time:.1f}s")
        
        self.show_experiment_summary(results, "lengths")
        self.offer_save_experiment(results, f"length_var_{seed.replace(' ', '_')}_T{temperature}")
    
    def full_matrix_experiment(self, generator):
        """Full matrix experiment: seeds × temperatures."""
        print("\n🎨 EXPERIMENTO: MATRIZ COMPLETA")
        print("=" * 40)
        print("⚠️  Advertencia: Esto puede generar muchos resultados")
        
        seeds_input = input("🌱 Seeds separados por comas (max 3): ").strip()
        if not seeds_input:
            print("❌ No se ingresaron seeds")
            return
        
        seeds = [s.strip() for s in seeds_input.split(',')][:3]  # Limit to 3
        temps = [0.6, 0.8, 1.0, 1.2]
        length = self.get_number_input("📏 Longitud fija: ", 150)
        
        total_experiments = len(seeds) * len(temps)
        
        confirm = input(f"\n⚠️  Esto generará {total_experiments} textos. ¿Continuar? (s/N): ").strip().lower()
        if confirm not in ['s', 'si', 'sí', 'y', 'yes']:
            return
        
        print(f"\n📊 EJECUTANDO MATRIZ COMPLETA:")
        print(f"   🌱 Seeds: {len(seeds)}")
        print(f"   🌡️  Temperatures: {temps}")
        print(f"   📏 Longitud: {length}")
        print(f"   🧪 Total experimentos: {total_experiments}")
        print("=" * 60)
        
        all_results = []
        experiment_num = 0
        
        for seed in seeds:
            for temp in temps:
                experiment_num += 1
                print(f"\n🧪 EXPERIMENTO {experiment_num}/{total_experiments}")
                print(f"   🌱 Seed: \"{seed}\" | 🌡️  T: {temp}")
                print("-" * 40)
                
                start_time = time.time()
                result = generator.generate(seed, length, temp)
                gen_time = time.time() - start_time
                
                all_results.append({
                    'seed': seed,
                    'temperature': temp,
                    'result': result,
                    'time': gen_time,
                    'chars': len(result),
                    'words': len(result.split())
                })
                
                # Show shortened result for matrix view
                preview = result[:100] + "..." if len(result) > 100 else result
                print(preview)
                print(f"⚡ {len(result)} chars en {gen_time:.1f}s")
        
        self.show_matrix_summary(all_results, seeds, temps)
        self.offer_save_experiment(all_results, f"matrix_L{length}")
    
    def show_experiment_summary(self, results, variable_type):
        """Show summary statistics for experiment."""
        print(f"\n📈 RESUMEN DEL EXPERIMENTO")
        print("=" * 40)
        
        total_chars = sum(r['chars'] for r in results)
        total_words = sum(r['words'] for r in results)
        total_time = sum(r['time'] for r in results)
        
        print(f"📊 Estadísticas generales:")
        print(f"   🧪 Experimentos: {len(results)}")
        print(f"   📝 Total caracteres: {total_chars:,}")
        print(f"   📚 Total palabras: {total_words:,}")
        print(f"   ⏱️  Tiempo total: {total_time:.1f}s")
        print(f"   📈 Promedio chars/experimento: {total_chars//len(results):,}")
        print(f"   ⚡ Velocidad promedio: {total_chars/total_time:.0f} chars/s")
        
        if variable_type == "temperatures":
            print(f"\n🌡️  Análisis por temperature:")
            for r in results:
                diversity = len(set(r['result'].split())) / len(r['result'].split()) if r['result'].split() else 0
                print(f"   T={r['temperature']:.2f}: {r['chars']} chars, diversidad={diversity:.2f}")
    
    def show_matrix_summary(self, results, seeds, temps):
        """Show matrix experiment summary."""
        print(f"\n📊 RESUMEN MATRIZ COMPLETA")
        print("=" * 50)
        
        print("📈 Resultados por seed:")
        for seed in seeds:
            seed_results = [r for r in results if r['seed'] == seed]
            avg_chars = sum(r['chars'] for r in seed_results) / len(seed_results)
            print(f"   \"{seed}\": {avg_chars:.0f} chars promedio")
        
        print("\n🌡️  Resultados por temperature:")
        for temp in temps:
            temp_results = [r for r in results if r['temperature'] == temp]
            avg_chars = sum(r['chars'] for r in temp_results) / len(temp_results)
            print(f"   T={temp}: {avg_chars:.0f} chars promedio")
    
    def offer_save_experiment(self, results, experiment_name):
        """Offer to save experiment results."""
        save_choice = input(f"\n💾 ¿Guardar resultados del experimento? (s/N): ").strip().lower()
        if save_choice in ['s', 'si', 'sí', 'y', 'yes']:
            self.save_experiment_to_file(results, experiment_name)
    
    def save_experiment_to_file(self, results, experiment_name):
        """Save experiment results to file."""
        try:
            # Create experiments directory
            exp_dir = Path("experiments")
            exp_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_{experiment_name}_{timestamp}.txt"
            filepath = exp_dir / filename
            
            # Save experiment
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Robo-Poet Experiment: {experiment_name}\n")
                f.write(f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total results: {len(results)}\n")
                f.write("# " + "="*60 + "\n\n")
                
                for i, result in enumerate(results, 1):
                    f.write(f"## RESULT {i}\n")
                    for key, value in result.items():
                        if key != 'result':
                            f.write(f"# {key}: {value}\n")
                    f.write("# " + "-"*40 + "\n")
                    f.write(result['result'])
                    f.write("\n\n" + "="*60 + "\n\n")
            
            print(f"✅ Experimento guardado en: {filepath}")
            
        except Exception as e:
            print(f"❌ Error guardando experimento: {e}")
    
    def thematic_templates(self, generator):
        """Thematic text generation templates."""
        print("\n🎨 PLANTILLAS TEMÁTICAS")
        print("=" * 40)
        
        templates = {
            '1': {
                'name': '📚 Narrativa Clásica',
                'seeds': ['Once upon a time', 'In a distant land', 'Long ago'],
                'temp': 0.7,
                'length': 300
            },
            '2': {
                'name': '🔬 Estilo Académico',
                'seeds': ['The research shows', 'According to studies', 'It is well established'],
                'temp': 0.5,
                'length': 250
            },
            '3': {
                'name': '🎭 Drama Filosófico',
                'seeds': ['The meaning of', 'One must consider', 'In the depths of'],
                'temp': 0.9,
                'length': 280
            },
            '4': {
                'name': '⚡ Acción Épica',
                'seeds': ['The battle began', 'Against all odds', 'In that moment'],
                'temp': 1.0,
                'length': 200
            },
            '5': {
                'name': '💭 Reflexión Personal',
                'seeds': ['I have learned', 'Looking back', 'The truth is'],
                'temp': 0.8,
                'length': 220
            }
        }
        
        print("🎭 PLANTILLAS DISPONIBLES:")
        for key, template in templates.items():
            print(f"{key}. {template['name']} (T={template['temp']}, L={template['length']})")
        
        template_choice = input("\n🎯 Selecciona plantilla (1-5): ").strip()
        
        if template_choice not in templates:
            print("❌ Plantilla inválida")
            return
        
        template = templates[template_choice]
        print(f"\n✅ Plantilla seleccionada: {template['name']}")
        print(f"🌱 Seeds disponibles: {template['seeds']}")
        
        # Generate with all seeds from template
        print(f"\n🎨 GENERANDO CON PLANTILLA...")
        print("=" * 50)
        
        for i, seed in enumerate(template['seeds'], 1):
            print(f"\n🎭 VARIANTE {i}: \"{seed}\"")
            print("-" * 40)
            
            start_time = time.time()
            result = generator.generate(seed, template['length'], template['temp'])
            gen_time = time.time() - start_time
            
            print(result)
            print(f"⚡ {len(result)} chars en {gen_time:.1f}s")
            
            # Offer to save each variant
            save_choice = input(f"\n💾 ¿Guardar esta variante? (s/N): ").strip().lower()
            if save_choice in ['s', 'si', 'sí', 'y', 'yes']:
                variant_name = f"{template['name'].replace(' ', '_')}_variante_{i}"
                self.save_generation_to_file(result, seed, template['temp'], template['length'])
        
        input("\n📖 Presiona Enter para continuar...")
    
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
                    self.clean_all_models()
                elif choice == '6':
                    self.show_system_status()
                elif choice == '7':
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
    

def run_direct_training(text_file: str, epochs: int):
    """Run direct training without interactive interface."""
    print(f"🔥 ENTRENAMIENTO DIRECTO - MODO ACADÉMICO GPU")
    print(f"📁 Archivo: {text_file}")
    print(f"🎯 Épocas: {epochs}")
    
    try:
        # Setup environment - GPU is MANDATORY
        print("\n🚀 Validando GPU obligatoria...")
        gpu_available = GPUConfigurator.setup_gpu()
        if not gpu_available:
            print("\n❌ ENTRENAMIENTO ABORTADO: GPU es obligatoria")
            print("🚨 REQUERIMIENTO ACADÉMICO: GPU NVIDIA es obligatoria")
            return False
            
        device = GPUConfigurator.get_device_strategy()
        print(f"🎓 Device académico: {device}")
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Get configuration
        model_config, system_config = get_config()
        
        # Prepare data
        print("\n📚 Preparando datos...")
        processor = TextProcessor(
            sequence_length=model_config.sequence_length,
            step_size=model_config.step_size
        )
        
        X, y = processor.prepare_data(
            text_file,
            max_length=model_config.max_text_length
        )
        
        print(f"✅ Datos preparados: {len(X):,} secuencias")
        
        # Build model
        print("\n🧠 Construyendo modelo...")
        lstm_generator = LSTMTextGenerator(
            vocab_size=processor.vocab_size,
            sequence_length=model_config.sequence_length,
            lstm_units=model_config.lstm_units,
            dropout_rate=model_config.dropout_rate
        )
        
        model = lstm_generator.build_model()
        
        # Train model
        print(f"\n⚡ INICIANDO ENTRENAMIENTO...")
        print(f"   Tiempo estimado: ~{epochs * 2} minutos")
        
        trainer = ModelTrainer(model, device)
        
        start_time = datetime.now()
        history = trainer.train(
            X, y,
            batch_size=model_config.batch_size,
            epochs=epochs,
            validation_split=model_config.validation_split
        )
        end_time = datetime.now()
        
        # Save model
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        model_path = f"models/robo_poet_model_{timestamp}.keras"
        
        ModelManager.save_model(model, model_path)
        
        # Save metadata
        metadata = {
            'vocab_size': processor.vocab_size,
            'sequence_length': model_config.sequence_length,
            'lstm_units': model_config.lstm_units,
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'epochs_trained': len(history.history['loss']),
            'training_time': str(end_time - start_time),
            'char_to_idx': processor.char_to_idx,
            'idx_to_char': processor.idx_to_char
        }
        
        metadata_path = model_path.replace('.keras', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"   ⏱️  Duración: {end_time - start_time}")
        print(f"   💾 Modelo guardado: {model_path}")
        print(f"   📊 Loss final: {metadata['final_loss']:.4f}")
        print(f"   ✅ Listo para generación")
        
    except Exception as e:
        print(f"\n❌ Error en entrenamiento: {e}")
        return False
    
    return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Robo-Poet: Academic Neural Text Generation')
    parser.add_argument('--text', type=str, help='Text file for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Check if CLI arguments provided for direct training
    if args.text and args.epochs:
        # Direct training mode
        if not Path(args.text).exists():
            print(f"❌ Archivo no encontrado: {args.text}")
            return 1
        
        success = run_direct_training(args.text, args.epochs)
        return 0 if success else 1
    else:
        # Interactive interface mode
        interface = AcademicInterface()
        interface.run_interface()
        return 0

if __name__ == "__main__":
    sys.exit(main())