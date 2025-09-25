#!/usr/bin/env python3
"""
Phase 2 Generation Interface for Robo-Poet Framework

Advanced text generation interface with 8 specialized modes.

Author: ML Academic Framework
Version: 2.1
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.file_manager import FileManager
from utils.input_validator import InputValidator
from utils.display_utils import DisplayUtils
from interface.generation_modes import GenerationModes
from src.data_processor import TextGenerator
try:
    from model_pytorch import RoboPoetModel as ModelManager
except ImportError:
    ModelManager = None


class Phase2GenerationInterface:
    """Handles Phase 2: Advanced text generation workflow."""
    
    def __init__(self, config):
        """Initialize generation interface."""
        self.config = config
        self.file_manager = FileManager()
        self.validator = InputValidator()
        self.display = DisplayUtils()
        self.generation_modes = GenerationModes()
    
    def run_generation_studio(self) -> bool:
        """Execute Phase 2 generation studio workflow."""
        self.display.clear_screen()
        print("[ART]" * 15 + " FASE 2: ESTUDIO DE GENERACIÓN AVANZADO " + "[ART]" * 15)
        print("=" * 80)
        print("[GRAD] GENERACIÓN DE TEXTO CON MODELOS PRE-ENTRENADOS")
        print("[BRAIN] 8 modos especializados para diferentes necesidades académicas")
        print("=" * 80)
        
        # Step 1: List and select model
        models = self.file_manager.list_available_models_enhanced()
        if not models:
            self.display.show_error("No hay modelos entrenados disponibles")
            print("[IDEA] Ejecuta primero FASE 1: Entrenamiento Intensivo")
            self.display.pause_for_user()
            return False
        
        selected_model = self._select_model(models)
        if not selected_model:
            return False
        
        # Step 2: Load model and metadata
        model, char_to_idx, idx_to_char, metadata = self._load_model_and_metadata(selected_model)
        if not model:
            return False
        
        # Step 3: Show model stats and enter generation loop
        self._show_model_summary(selected_model, metadata)
        return self._generation_loop(model, char_to_idx, idx_to_char, metadata, selected_model['name'])
    
    def _select_model(self, models: list) -> Optional[Dict]:
        """Display models and let user select one."""
        print("\n[CHART] MODELOS DISPONIBLES PARA GENERACIÓN")
        print("=" * 60)
        
        for i, model_info in enumerate(models, 1):
            print(f"{i}. ", end="")
            self.display.format_model_info(model_info)
        
        choice = self.validator.get_menu_choice("[TARGET] Selecciona modelo", len(models))
        return models[choice - 1]
    
    def _load_model_and_metadata(self, selected_model: Dict) -> tuple:
        """Load selected model and its metadata."""
        try:
            print(f"\n[BOOKS] Cargando modelo: {selected_model['name']}")
            
            # Load model
            model_manager = ModelManager()
            model = model_manager.load_model(selected_model['path'])
            
            if not model:
                self.display.show_error("No se pudo cargar el modelo")
                return None, None, None, None
            
            # Load metadata and tokenizer info
            metadata = selected_model.get('metadata', {})
            if metadata:
                char_to_idx = metadata.get('char_to_idx', {})
                raw_idx_to_char = metadata.get('idx_to_char', {})
                # Ensure idx_to_char has integer keys
                idx_to_char = {int(k): v for k, v in raw_idx_to_char.items()}
            else:
                self.display.show_warning("Metadata no disponible, creando tokenizer básico")
                # Fallback tokenizer creation
                char_to_idx, idx_to_char = self._create_fallback_tokenizer()
            
            print("[OK] Modelo y metadata cargados exitosamente")
            return model, char_to_idx, idx_to_char, metadata
            
        except Exception as e:
            self.display.show_error(f"Error cargando modelo: {e}")
            return None, None, None, None
    
    def _create_fallback_tokenizer(self) -> tuple:
        """Create basic tokenizer from training text if metadata not available."""
        text_path = Path('The+48+Laws+Of+Power_texto.txt')
        if text_path.exists():
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chars = sorted(set(text))
            char_to_idx = {c: i for i, c in enumerate(chars)}
            idx_to_char = {i: c for i, c in enumerate(chars)}
            return char_to_idx, idx_to_char
        else:
            self.display.show_error("No se puede crear tokenizer sin archivo de texto")
            return {}, {}
    
    def _show_model_summary(self, selected_model: Dict, metadata: Dict) -> None:
        """Display detailed model summary for academic purposes."""
        print(f"\n[AI] MODELO SELECCIONADO: {selected_model['name']}")
        print("=" * 60)
        print(f" Creado: {selected_model['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[SAVE] Tamaño: {selected_model['size_mb']:.1f} MB")
        print(f"[STAR] Calidad: {selected_model['quality_rating']}")
        
        if metadata:
            self.display.show_model_stats_summary(metadata)
        
        print("=" * 60)
        self.display.pause_for_user("Presiona Enter para continuar al estudio de generación...")
    
    def _generation_loop(self, model, char_to_idx: Dict, idx_to_char: Dict, 
                        metadata: Dict, model_name: str) -> bool:
        """Main generation loop with 8 specialized modes."""
        # Create text generator
        generator = TextGenerator(model, char_to_idx, idx_to_char)
        
        # Set model name in generation modes for proper display
        self.generation_modes.set_model_name(model_name)
        
        while True:
            self.display.clear_screen()
            self._show_generation_menu(model_name)
            
            choice = self.validator.get_menu_choice("[TARGET] Selecciona modo de generación", 8)
            
            try:
                if choice == 1:
                    self.generation_modes.quick_generation(generator)
                elif choice == 2:
                    self.generation_modes.creative_lab(generator)
                elif choice == 3:
                    self.generation_modes.interactive_session(generator)
                elif choice == 4:
                    self.generation_modes.batch_experiments(generator)
                elif choice == 5:
                    self.generation_modes.thematic_templates(generator)
                elif choice == 6:
                    self._show_detailed_analysis(metadata)
                elif choice == 7:
                    self._manage_generations()
                elif choice == 8:
                    print("[TARGET] Regresando al menú principal...")
                    return True
                
            except KeyboardInterrupt:
                print("\nWARNING: Operación interrumpida")
                if self.validator.get_confirmation("¿Regresar al menú principal?", True):
                    return True
            except Exception as e:
                self.display.show_error(f"Error en generación: {e}")
                self.display.pause_for_user()
    
    def _show_generation_menu(self, model_name: str) -> None:
        """Display the generation studio menu."""
        print("[ART] ESTUDIO DE GENERACIÓN AVANZADO")
        print("=" * 50)
        print(f"[AI] Modelo activo: {model_name}")
        print()
        print("[TARGET] MODOS DE GENERACIÓN:")
        print("1. [LAUNCH] Generación Rápida (presets optimizados)")
        print("2. [SCIENCE] Laboratorio Creativo (control total)")
        print("3. [GAME] Sesión Interactiva (comandos avanzados)")
        print("4.  Experimentos en Lote (análisis sistemático)")
        print("5. [ART] Plantillas Temáticas (estilos literarios)")
        print()
        print("[CHART] ANÁLISIS Y GESTIÓN:")
        print("6. [GROWTH] Análisis Detallado del Modelo")
        print("7.  Gestionar Generaciones Guardadas")
        print("8.  Regresar al Menú Principal")
        print("=" * 50)
    
    def _show_detailed_analysis(self, metadata: Dict) -> None:
        """Show comprehensive model analysis."""
        print("\n[GROWTH] ANÁLISIS DETALLADO DEL MODELO")
        print("=" * 60)
        
        if not metadata:
            print("WARNING: Metadata no disponible para análisis detallado")
            self.display.pause_for_user()
            return
        
        # Training metrics
        print("[TARGET] MÉTRICAS DE ENTRENAMIENTO:")
        print(f"   [GROWTH] Épocas completadas: {metadata.get('final_epoch', 'N/A')}")
        print(f"    Loss final: {metadata.get('final_loss', 'N/A'):.4f}" 
              if isinstance(metadata.get('final_loss'), (int, float)) else "    Loss final: N/A")
        print(f"   [CHART] Accuracy final: {metadata.get('final_accuracy', 'N/A'):.4f}" 
              if isinstance(metadata.get('final_accuracy'), (int, float)) else "   [CHART] Accuracy final: N/A")
        print(f"   [TIME] Duración entrenamiento: {metadata.get('training_duration', 'N/A')}")
        
        # Model architecture
        print(f"\n[BRAIN] ARQUITECTURA:")
        print(f"   [TARGET] Vocabulario: {metadata.get('vocab_size', 'N/A')} caracteres únicos")
        print(f"    Longitud de secuencia: {metadata.get('sequence_length', 'N/A')} tokens")
        print(f"   [BRAIN] LSTM units: {metadata.get('lstm_units', 'N/A')}")
        print(f"    Dropout rate: {metadata.get('dropout_rate', 'N/A')}")
        print(f"   [PACKAGE] Batch size: {metadata.get('batch_size', 'N/A')}")
        
        # Performance recommendations
        final_loss = metadata.get('final_loss')
        if isinstance(final_loss, (int, float)):
            print(f"\n[TARGET] RECOMENDACIONES DE USO:")
            if final_loss < 1.0:
                print("   [STAR] Excelente para generación creativa y narrativa")
                print("   [DOC] Usa temperature 0.7-1.0 para mejores resultados")
                print("    Longitudes 200-500 caracteres son óptimas")
            elif final_loss < 1.5:
                print("   [STAR] Bueno para textos coherentes y experimentación")
                print("   [DOC] Usa temperature 0.6-0.9 para estabilidad")
                print("    Longitudes 150-300 caracteres recomendadas")
            elif final_loss < 2.0:
                print("   [CHART] Aceptable para pruebas y aprendizaje")
                print("   [DOC] Usa temperature 0.5-0.7 para mayor coherencia")
                print("    Longitudes cortas (100-200) son más estables")
            else:
                print("   WARNING: Modelo requiere más entrenamiento")
                print("   [DOC] Usa temperature baja (0.4-0.6) y longitudes cortas")
                print("   [CYCLE] Considera entrenar más épocas")
        
        print("=" * 60)
        self.display.pause_for_user()
    
    def _manage_generations(self) -> None:
        """Manage saved generations."""
        print("\n GESTIONAR GENERACIONES GUARDADAS")
        print("=" * 50)
        
        generations = self.file_manager.get_available_generations()
        
        if not generations:
            print(" No hay generaciones guardadas")
            self.display.pause_for_user()
            return
        
        print(f"[CHART] Total de generaciones guardadas: {len(generations)}")
        print()
        
        for i, gen in enumerate(generations[:10], 1):  # Show last 10
            print(f"{i}. {gen['timestamp']} ({gen['size_kb']:.1f} KB)")
            if gen['metadata']:
                meta = gen['metadata']
                print(f"    Seed: \"{meta.get('seed', 'N/A')}\"")
                print(f"    T={meta.get('temperature', 'N/A')} | L={meta.get('length', 'N/A')}")
            print()
        
        if len(generations) > 10:
            print(f"   ... y {len(generations) - 10} más")
        
        print("\n[IDEA] Las generaciones se guardan en el directorio 'generations/'")
        self.display.pause_for_user()