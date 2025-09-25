#!/usr/bin/env python3
"""
Phase 1 Training Interface for Robo-Poet Framework

Handles intensive training workflow with academic guidance and GPU optimization.

Author: ML Academic Framework
Version: 2.1
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.file_manager import FileManager
from utils.input_validator import InputValidator
from utils.display_utils import DisplayUtils


class Phase1TrainingInterface:
    """Handles Phase 1: Intensive Training workflow."""
    
    def __init__(self, config):
        """Initialize training interface with configuration."""
        self.config = config
        self.file_manager = FileManager()
        self.validator = InputValidator()
        self.display = DisplayUtils()
    
    def _get_model_name(self) -> Optional[str]:
        """Get and validate model name from user (mandatory)."""
        print("\n[TARGET] NOMBRE DEL MODELO (OBLIGATORIO)")
        print("=" * 50)
        print(" Cada modelo debe tener un nombre único para identificarlo")
        print("[IDEA] Ejemplos: 'shakespeare_v1', 'poeta_español', 'novela_fantasia'")
        print("WARNING:  El nombre no puede dejarse vacío")
        
        while True:
            model_name = input("\n  Ingresa el nombre del modelo: ").strip()
            
            if not model_name:
                self.display.show_error("[X] El nombre del modelo es OBLIGATORIO")
                print("   Por favor, ingresa un nombre válido para continuar")
                continue
            
            # Validar caracteres permitidos (alfanuméricos, guiones y underscores)
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
                self.display.show_error(
                    "[X] El nombre solo puede contener letras, números, guiones y underscores"
                )
                continue
            
            # Verificar si ya existe un modelo con ese nombre
            models_dir = Path('models')
            if models_dir.exists():
                existing_model = models_dir / f"{model_name}.keras"
                if existing_model.exists():
                    self.display.show_warning(
                        f"WARNING:  Ya existe un modelo llamado '{model_name}'"
                    )
                    overwrite = self.validator.get_confirmation(
                        "¿Deseas sobrescribir el modelo existente?",
                        default_yes=False
                    )
                    if not overwrite:
                        continue
            
            print(f"[OK] Nombre del modelo: '{model_name}'")
            return model_name
    
    def run_intensive_training(self) -> bool:
        """Execute Phase 1 intensive training workflow."""
        self.display.clear_screen()
        print("[FIRE]" * 20 + " FASE 1: ENTRENAMIENTO INTENSIVO " + "[FIRE]" * 20)
        print("=" * 80)
        print("[GRAD] PROCESO ACADÉMICO DE ENTRENAMIENTO NEURONAL")
        print("[FAST] Optimizado para GPU NVIDIA RTX 2000 Ada")
        print("⏰ Duración estimada: 1-3 horas (dependiendo de épocas)")
        print("=" * 80)
        
        # Step 1: Get model name FIRST (mandatory)
        model_name = self._get_model_name()
        if not model_name:
            return False
        
        # Step 2: Select training text
        text_file = self._get_training_text()
        if not text_file:
            return False
        
        # Step 3: Configure epochs
        epochs = self._get_epochs_configuration()
        if epochs == 0:
            return False
        
        # Step 4: Show training preview and confirmation
        if not self._confirm_training(text_file, epochs, model_name):
            return False
        
        # Step 5: Execute training
        return self._execute_training(text_file, epochs, model_name)
    
    def _get_training_text(self) -> Optional[str]:
        """Process entire corpus automatically for PyTorch training."""
        print("\n[LAUNCH] PROCESAMIENTO AUTOMÁTICO DE MULTI-CORPUS PYTORCH")
        print("=" * 60)
        print("[BOOKS] Procesando TODOS los archivos del corpus automáticamente...")
        print("[CYCLE] No se requiere selección manual - sistema académico optimizado")
        print()
        
        try:
            # Import and use the multi-corpus processor using direct import
            import sys
            import importlib.util
            from pathlib import Path
            
            # Direct import of the module
            processor_path = Path(__file__).parent.parent / "data" / "pytorch_multicorpus_processor.py"
            spec = importlib.util.spec_from_file_location("pytorch_multicorpus_processor", processor_path)
            processor_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(processor_module)
            process_corpus_automatically = processor_module.process_corpus_automatically
            
            # Process entire corpus folder automatically
            print(" Procesando carpeta corpus completa...")
            metadata = process_corpus_automatically(
                corpus_dir="corpus",
                output_dir="data/processed"
            )
            
            print(f"\n[OK] PROCESAMIENTO MULTI-CORPUS COMPLETADO")
            print(f"   [CHART] Archivos procesados: {metadata['files_processed']}")
            print(f"    Archivos: {', '.join(metadata['file_names'])}")
            print(f"    Total caracteres: {metadata['total_characters']:,}")
            print(f"    Vocabulario: {metadata['vocabulary_size']} tokens")
            print(f"   [CHART] Splits: Train/Val/Test creados automáticamente")
            
            # Return the unified corpus path for training
            unified_corpus_path = "data/processed/unified_corpus.txt"
            print(f"\n[TARGET] Corpus unificado listo para entrenamiento PyTorch:")
            print(f"    {unified_corpus_path}")
            
            return unified_corpus_path
            
        except Exception as e:
            self.display.show_error(f"Error en procesamiento automático: {e}")
            print("\nWARNING: Fallback: Intentando procesamiento manual...")
            return self._get_training_text_fallback()
    
    def _get_training_text_fallback(self) -> Optional[str]:
        """Fallback method for manual corpus selection if automatic processing fails."""
        print("\n[BOOKS] SELECCIÓN MANUAL DE CORPUS (FALLBACK)")
        print("=" * 50)
        
        # Show available text files
        text_files = self.file_manager.get_text_files()
        
        if text_files:
            print(" Archivos de texto disponibles:")
            for i, text_file in enumerate(text_files, 1):
                file_size = Path(text_file).stat().st_size / 1024
                print(f"   {i}. {text_file} ({file_size:.1f} KB)")
            print(f"   {len(text_files) + 1}. Especificar ruta manual")
        else:
            print("WARNING: No se encontraron archivos de texto automáticamente")
            print("[IDEA] Puedes especificar la ruta manualmente")
        
        # Get user choice
        while True:
            if text_files:
                choice = self.validator.get_menu_choice(
                    "\n[TARGET] Selecciona archivo de entrenamiento", 
                    len(text_files) + 1
                )
                
                if choice <= len(text_files):
                    selected_file = text_files[choice - 1]
                    valid, message = self.file_manager.validate_text_file(selected_file)
                    if valid:
                        print(f"[OK] {message}")
                        return selected_file
                    else:
                        self.display.show_error(message)
                        continue
            else:
                choice = len(text_files) + 1
            
            # Manual file path
            manual_path = self.validator.get_file_path_input(
                "\n[DOC] Ingresa la ruta del archivo de texto"
            )
            if manual_path:
                valid, message = self.file_manager.validate_text_file(manual_path)
                if valid:
                    print(f"[OK] {message}")
                    return manual_path
                else:
                    self.display.show_error(message)
            else:
                print("[X] Entrenamiento cancelado")
                return None
    
    def _get_epochs_configuration(self) -> int:
        """Get and validate epochs configuration."""
        print("\n[TARGET] CONFIGURACIÓN DE ÉPOCAS DE ENTRENAMIENTO")
        print("=" * 50)
        
        self.display.show_academic_tip(
            "Las épocas determinan cuántas veces el modelo verá todo el dataset.\n"
            "   • 5-10 épocas: Prueba rápida (15-30 min)\n"
            "   • 20-30 épocas: Entrenamiento estándar (1-2 horas)\n"
            "   • 50+ épocas: Entrenamiento intensivo (3+ horas)"
        )
        
        while True:
            epochs = self.validator.get_number_input(
                " Número de épocas", 
                default=20, 
                min_val=1, 
                max_val=100
            )
            
            if self.validator.validate_epochs_input(epochs):
                return epochs
            
            # If validation failed but user wants to continue anyway
            if epochs > 100:
                return 0  # Cancel training
    
    def _confirm_training(self, text_file: str, epochs: int, model_name: str) -> bool:
        """Show training configuration and get final confirmation."""
        print("\nWARNING: CONFIRMACIÓN FINAL DE ENTRENAMIENTO")
        print("=" * 50)
        
        # Show configuration summary
        file_size = Path(text_file).stat().st_size / 1024
        estimated_time_min = epochs * 3
        estimated_time_max = epochs * 8
        
        print(f"  Modelo: {model_name}")
        print(f"[BOOKS] Archivo: {text_file}")
        print(f"[CHART] Tamaño: {file_size:.1f} KB")
        print(f"[TARGET] Épocas: {epochs}")
        print(f"⏰ Tiempo estimado: {estimated_time_min}-{estimated_time_max} minutos")
        print(f"[BRAIN] LSTM units: {self.config.model.lstm_units}")
        print(f"[PACKAGE] Batch size: {self.config.model.batch_size}")
        
        self.display.show_warning(
            "El entrenamiento usará GPU intensivamente y no debe interrumpirse.\n"
            "   Asegúrate de tener tiempo suficiente y energía estable."
        )
        
        return self.validator.get_confirmation(
            "\n[LAUNCH] ¿Iniciar entrenamiento intensivo?", 
            default_yes=True
        )
    
    def _execute_training(self, text_file: str, epochs: int, model_name: str) -> bool:
        """Execute the actual training process."""
        self.display.show_training_header(text_file, epochs)
        
        print(f"  Modelo: {model_name}")
        print("[FIX] Configurando entorno GPU...")
        print("[BOOKS] Preparando datos...")
        print("[BRAIN] Construyendo modelo LSTM...")
        print("[FAST] Iniciando entrenamiento...")
        print()
        
        try:
            # Use PyTorch training system instead of old TensorFlow
            print("[LAUNCH] Iniciando entrenamiento PyTorch GPT (Academic Performance Mode)")
            print(f"   [GRAD] GPU mandatory for academic standards")
            print(f"   [FIRE] Using: NVIDIA RTX 2000 Ada")
            print()
            
            # Prepare PyTorch training command
            pytorch_main = "robo-poet-pytorch/main.py"
            cmd = [
                sys.executable, 
                pytorch_main,
                'train',
                '--epochs', str(epochs),
                '--batch_size', '32',
                '--context_length', '128'
            ]
            
            # Configure environment for subprocess
            env = os.environ.copy()
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            if conda_prefix:
                env['CUDA_HOME'] = conda_prefix
                env['TF_CPP_MIN_LOG_LEVEL'] = '2'
                lib_paths = [f'{conda_prefix}/lib', f'{conda_prefix}/lib64']
                existing_ld = env.get('LD_LIBRARY_PATH', '')
                if existing_ld:
                    lib_paths.append(existing_ld)
                clean_ld = ':'.join(lib_paths)
                env['LD_LIBRARY_PATH'] = clean_ld
                env['CUDA_VISIBLE_DEVICES'] = '0'
                env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            
            print("[LAUNCH] Ejecutando entrenamiento intensivo...")
            print("[IDEA] Monitoreo disponible: nvidia-smi (en otra terminal)")
            print("[CHART] Logs: TensorBoard se iniciará automáticamente")
            print("-" * 80)
            
            # Execute training
            start_time = time.time()
            result = subprocess.run(cmd, env=env, capture_output=False, text=True)
            end_time = time.time()
            
            training_duration = end_time - start_time
            
            if result.returncode == 0:
                self.display.show_success(
                    f"Entrenamiento completado exitosamente en {training_duration/60:.1f} minutos"
                )
                print(f" Modelo '{model_name}' guardado en directorio models/")
                print(f" Ubicación: models/{model_name}.keras")
                print("[ART] Ahora puedes usar FASE 2: Generación de Texto")
                self.display.pause_for_user()
                return True
            else:
                self.display.show_error(
                    f"Error durante entrenamiento (código: {result.returncode})"
                )
                print("[IDEA] Revisa los logs anteriores para detalles del error")
                self.display.pause_for_user()
                return False
                
        except subprocess.TimeoutExpired:
            self.display.show_error("Entrenamiento excedió tiempo límite")
            return False
        except KeyboardInterrupt:
            self.display.show_warning("Entrenamiento interrumpido por usuario")
            return False
        except Exception as e:
            self.display.show_error(f"Error ejecutando entrenamiento: {e}")
            return False
    
    def show_training_tips(self):
        """Display academic tips for training optimization."""
        print("\n[IDEA] CONSEJOS ACADÉMICOS PARA ENTRENAMIENTO ÓPTIMO")
        print("=" * 60)
        print("[TARGET] PREPARACIÓN DE DATOS:")
        print("   • Usa archivos de texto >50KB para mejores resultados")
        print("   • Texto en español o inglés funciona mejor")
        print("   • Evita archivos con muchos caracteres especiales")
        print()
        print("[FAST] OPTIMIZACIÓN GPU:")
        print("   • Asegúrate que nvidia-smi muestre tu RTX 2000 Ada")
        print("   • Cierra aplicaciones que usen GPU (juegos, videos)")
        print("   • Monitorea temperatura con 'watch nvidia-smi'")
        print()
        print("[BRAIN] CONFIGURACIÓN DE ÉPOCAS:")
        print("   • 10-20 épocas: Suficiente para textos simples")
        print("   • 30-50 épocas: Recomendado para calidad profesional")
        print("   • 50+ épocas: Solo para investigación avanzada")
        print()
        print("[CHART] MONITOREO:")
        print("   • Loss <1.5: Excelente calidad")
        print("   • Loss 1.5-2.0: Calidad aceptable")
        print("   • Loss >2.0: Necesita más entrenamiento")
        print("=" * 60)
        
        self.display.pause_for_user()