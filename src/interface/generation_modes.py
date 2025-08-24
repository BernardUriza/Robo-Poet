#!/usr/bin/env python3
"""
Advanced Generation Modes for Robo-Poet Framework

Implements the 8 specialized generation modes for Phase 2 interface.

Author: ML Academic Framework
Version: 2.1
"""

import time
import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.input_validator import InputValidator
from utils.display_utils import DisplayUtils
from utils.file_manager import FileManager


class GenerationModes:
    """Implements advanced generation modes for Phase 2."""
    
    def __init__(self):
        """Initialize generation modes with utilities."""
        self.validator = InputValidator()
        self.display = DisplayUtils()
        self.file_manager = FileManager()
        
        # Model name for display
        self.model_name = "unknown"
        
        # Session statistics
        self.session_stats = {
            'generations_count': 0,
            'total_chars': 0,
            'total_time': 0,
            'start_time': datetime.now()
        }
        self.last_generation = None
    
    def set_model_name(self, model_name: str):
        """Set the model name for generation display."""
        self.model_name = model_name
    
    def quick_generation(self, generator) -> None:
        """Quick generation with optimized presets."""
        print("\n🚀 GENERACIÓN RÁPIDA - PRESETS OPTIMIZADOS")
        print("=" * 50)
        
        presets = {
            '1': {'name': '📖 Narrativa', 'temp': 0.7, 'length': 300, 'seed': 'Once upon a time'},
            '2': {'name': '🎨 Creativo', 'temp': 1.0, 'length': 250, 'seed': 'The secret to'},
            '3': {'name': '🔬 Experimental', 'temp': 1.3, 'length': 200, 'seed': 'In this world'},
            '4': {'name': '📚 Académico', 'temp': 0.5, 'length': 350, 'seed': 'The purpose of'},
            '5': {'name': '🎭 Artístico', 'temp': 1.1, 'length': 220, 'seed': 'Behind the scenes'}
        }
        
        print("🎯 PRESETS DISPONIBLES:")
        for key, preset in presets.items():
            print(f"{key}. {preset['name']} (T={preset['temp']}, L={preset['length']})")
        
        choice = self.validator.get_menu_choice("\n🎯 Selecciona preset", 5)
        preset = presets[str(choice)]
        
        print(f"\n✅ Preset seleccionado: {preset['name']}")
        print(f"🎛️ Configuración: T={preset['temp']}, L={preset['length']}")
        print(f"🌱 Seed sugerido: \"{preset['seed']}\"")
        
        # Allow seed customization
        custom_seed = input(f"\n🌱 Usar seed personalizado (Enter para '{preset['seed']}'): ").strip()
        final_seed = custom_seed if custom_seed else preset['seed']
        
        # Generate
        self._generate_and_display(generator, final_seed, preset['temp'], preset['length'])
    
    def creative_lab(self, generator) -> None:
        """Advanced creative laboratory with full control."""
        print("\n🔬 LABORATORIO CREATIVO")
        print("=" * 40)
        print("🎛️ Control total sobre la generación")
        
        # Get parameters
        seed = self.validator.get_text_input("🌱 Seed text", "The art of", 1, 200)
        temperature = self.validator.get_float_input("🌡️ Temperature", 0.8, 0.1, 2.0)
        length = self.validator.get_number_input("📏 Longitud", 200, 10, 1000)
        
        # Creative options
        print("\n🎨 OPCIONES CREATIVAS:")
        print("1. 🌊 Flujo normal")
        print("2. 🎯 Generación dirigida (múltiples intentos)")
        print("3. 🔄 Variaciones del mismo seed")
        
        mode = self.validator.get_menu_choice("🎯 Selecciona modo", 3)
        
        if mode == 2:
            self._directed_generation(generator, seed, temperature, length)
        elif mode == 3:
            self._seed_variations(generator, seed, temperature, length)
        else:
            self._generate_and_display(generator, seed, temperature, length)
    
    def interactive_session(self, generator) -> None:
        """Enhanced interactive generation session."""
        print("\n🎮 SESIÓN INTERACTIVA AVANZADA")
        print("=" * 50)
        print("🎛️ COMANDOS DISPONIBLES:")
        print("   • temp X.X     - Cambiar temperatura (0.1-2.0)")
        print("   • len XXX      - Cambiar longitud (50-500)")
        print("   • save         - Guardar última generación")
        print("   • stats        - Ver estadísticas de sesión")
        print("   • clear        - Limpiar historial")
        print("   • help         - Mostrar ayuda")
        print("   • exit         - Salir de sesión interactiva")
        print("=" * 50)
        
        # Default settings
        current_temp = 0.8
        current_length = 200
        
        while True:
            print(f"\n🎛️ Config actual: T={current_temp}, L={current_length}")
            user_input = input("🎯 Seed o comando > ").strip()
            
            if not user_input:
                continue
            
            # Parse commands
            if user_input.startswith('temp '):
                try:
                    current_temp = float(user_input.split()[1])
                    if self.validator.validate_temperature_input(current_temp):
                        print(f"✅ Temperature actualizada: {current_temp}")
                    else:
                        current_temp = 0.8
                except:
                    print("❌ Formato inválido. Uso: temp 0.8")
                continue
            
            elif user_input.startswith('len '):
                try:
                    current_length = int(user_input.split()[1])
                    if self.validator.validate_length_input(current_length):
                        print(f"✅ Longitud actualizada: {current_length}")
                    else:
                        current_length = 200
                except:
                    print("❌ Formato inválido. Uso: len 200")
                continue
            
            elif user_input == 'save':
                if self.last_generation:
                    self._save_last_generation()
                else:
                    print("❌ No hay generación para guardar")
                continue
            
            elif user_input == 'stats':
                self._show_session_stats()
                continue
            
            elif user_input == 'clear':
                self.session_stats = {
                    'generations_count': 0,
                    'total_chars': 0,
                    'total_time': 0,
                    'start_time': datetime.now()
                }
                print("✅ Estadísticas de sesión reiniciadas")
                continue
            
            elif user_input == 'help':
                self._show_interactive_help()
                continue
            
            elif user_input == 'exit':
                print("🎯 Saliendo de sesión interactiva...")
                self._show_session_stats()
                break
            
            else:
                # Treat as seed for generation
                self._generate_and_display(generator, user_input, current_temp, current_length)
    
    def batch_experiments(self, generator) -> None:
        """Batch experiments with systematic parameter variation."""
        print("\n⚗️ EXPERIMENTOS EN LOTE")
        print("=" * 40)
        print("📊 Generación sistemática para análisis comparativo")
        
        experiment_types = {
            '1': 'Multi-seed con temperatura fija',
            '2': 'Barrido de temperatura con seed fijo',
            '3': 'Variación de longitud con parámetros fijos',
            '4': 'Matriz completa (múltiples parámetros)'
        }
        
        print("\n🔬 TIPOS DE EXPERIMENTO:")
        for key, desc in experiment_types.items():
            print(f"{key}. {desc}")
        
        exp_type = self.validator.get_menu_choice("🎯 Selecciona experimento", 4)
        
        if exp_type == 1:
            self._multi_seed_experiment(generator)
        elif exp_type == 2:
            self._temperature_sweep_experiment(generator)
        elif exp_type == 3:
            self._length_variation_experiment(generator)
        elif exp_type == 4:
            self._matrix_experiment(generator)
    
    def thematic_templates(self, generator) -> None:
        """Themed generation templates for different literary styles."""
        print("\n🎨 PLANTILLAS TEMÁTICAS")
        print("=" * 40)
        print("📚 Estilos literarios predefinidos")
        
        templates = {
            '1': {
                'name': '🏰 Fantasía Épica',
                'seeds': ['In a kingdom far away', 'The ancient prophecy', 'Beyond the mountains'],
                'temp': 0.9, 'length': 350
            },
            '2': {
                'name': '🔬 Ciencia Ficción', 
                'seeds': ['In the year 2157', 'The spaceship landed', 'The artificial intelligence'],
                'temp': 0.8, 'length': 300
            },
            '3': {
                'name': '🕵️ Misterio/Suspenso',
                'seeds': ['The detective found', 'On a dark night', 'The secret document'],
                'temp': 0.7, 'length': 280
            },
            '4': {
                'name': '💫 Filosofía/Reflexión',
                'seeds': ['The meaning of life', 'In human nature', 'The essence of'],
                'temp': 0.6, 'length': 400
            },
            '5': {
                'name': '🎭 Drama/Romance',
                'seeds': ['Their eyes met', 'In that moment', 'The heart remembers'],
                'temp': 1.0, 'length': 250
            }
        }
        
        print("\n📖 PLANTILLAS DISPONIBLES:")
        for key, template in templates.items():
            print(f"{key}. {template['name']} (T={template['temp']}, L={template['length']})")
        
        choice = self.validator.get_menu_choice("🎯 Selecciona plantilla", 5)
        template = templates[str(choice)]
        
        print(f"\n✅ Plantilla: {template['name']}")
        print(f"🌱 Seeds disponibles: {', '.join(template['seeds'])}")
        
        # Select seed
        print("\n🌱 SEEDS TEMÁTICOS:")
        for i, seed in enumerate(template['seeds'], 1):
            print(f"{i}. \"{seed}\"")
        
        seed_choice = self.validator.get_menu_choice("🎯 Selecciona seed", len(template['seeds']))
        selected_seed = template['seeds'][seed_choice - 1]
        
        # Generate with template
        self._generate_and_display(generator, selected_seed, template['temp'], template['length'])
    
    def _generate_and_display(self, generator, seed: str, temperature: float, length: int) -> None:
        """Generate text and display with full academic formatting."""
        print(f"\n🎨 Generando texto...")
        print(f"   🎛️ Configuración: T={temperature}, L={length}")
        print(f"   🌱 Seed: \"{seed}\"")
        print("   ⏳ Procesando...")
        
        start_time = time.time()
        result = generator.generate(seed, length, temperature)
        generation_time = time.time() - start_time
        
        # Update session stats
        self.session_stats['generations_count'] += 1
        self.session_stats['total_chars'] += len(result)
        self.session_stats['total_time'] += generation_time
        
        # Store last generation
        self.last_generation = {
            'result': result,
            'seed': seed,
            'temperature': temperature,
            'length': length,
            'generation_time': generation_time,
            'timestamp': datetime.now(),
            'model_name': self.model_name
        }
        
        # Display result with model name
        self.display.format_generation_result(result, seed, temperature, length, generation_time, self.model_name)
        
        # Offer save
        if self.validator.get_confirmation("💾 ¿Guardar esta generación?", False):
            self._save_last_generation()
    
    def _directed_generation(self, generator, seed: str, temperature: float, length: int) -> None:
        """Generate multiple attempts for comparison."""
        attempts = self.validator.get_number_input("🔄 Número de intentos", 3, 2, 5)
        print(f"\n🎲 Generando {attempts} variaciones...")
        
        results = []
        for i in range(attempts):
            print(f"\n--- INTENTO {i+1} ---")
            result = generator.generate(seed, length//3, temperature)
            results.append(result)
            print(result[:150] + "..." if len(result) > 150 else result)
        
        choice = self.validator.get_menu_choice(f"🤔 ¿Generar alguna completa? (1-{attempts})", attempts)
        if choice <= len(results):
            self._generate_and_display(generator, seed, temperature, length)
    
    def _seed_variations(self, generator, base_seed: str, temperature: float, length: int) -> None:
        """Generate variations of the same seed."""
        variations = ["", " and", " but", " so", " yet", " when"]
        print(f"\n🔄 Generando variaciones de \"{base_seed}\"...")
        
        for i, variation in enumerate(variations[:3], 1):
            varied_seed = base_seed + variation
            print(f"\n--- VARIACIÓN {i}: \"{varied_seed}\" ---")
            result = generator.generate(varied_seed, length//2, temperature)
            print(result)
    
    def _multi_seed_experiment(self, generator) -> None:
        """Multi-seed experiment with fixed parameters."""
        base_seeds = [
            "The power of", "In the beginning", "The secret to",
            "Behind the scenes", "The art of", "In this world"
        ]
        
        temperature = self.validator.get_float_input("🌡️ Temperature fija", 0.8, 0.1, 2.0)
        length = self.validator.get_number_input("📏 Longitud fija", 150, 50, 300)
        
        results = []
        for i, seed in enumerate(base_seeds, 1):
            print(f"\n🧪 Experimento {i}/{len(base_seeds)}: \"{seed}\"")
            start_time = time.time()
            result = generator.generate(seed, length, temperature)
            gen_time = time.time() - start_time
            
            results.append({
                'seed': seed,
                'temperature': temperature,
                'length': length,
                'result': result,
                'generation_time': gen_time
            })
            
            print(f"   ⏱️ {gen_time:.2f}s | {len(result)} chars")
            print(f"   📝 {result[:100]}...")
        
        # Save experiment
        filepath = self.file_manager.save_experiment_results("multi_seed", results)
        print(f"\n✅ Experimento guardado en: {filepath}")
    
    def _temperature_sweep_experiment(self, generator) -> None:
        """Temperature sweep experiment."""
        seed = self.validator.get_text_input("🌱 Seed fijo", "The essence of", 1, 100)
        length = self.validator.get_number_input("📏 Longitud fija", 150, 50, 300)
        
        temperatures = [0.5, 0.7, 0.9, 1.1, 1.3]
        results = []
        
        for i, temp in enumerate(temperatures, 1):
            print(f"\n🌡️ Temperatura {i}/{len(temperatures)}: {temp}")
            start_time = time.time()
            result = generator.generate(seed, length, temp)
            gen_time = time.time() - start_time
            
            results.append({
                'seed': seed,
                'temperature': temp,
                'length': length,
                'result': result,
                'generation_time': gen_time
            })
            
            print(f"   ⏱️ {gen_time:.2f}s | Creatividad: {'Alta' if temp > 1.0 else 'Media' if temp > 0.8 else 'Baja'}")
            print(f"   📝 {result[:100]}...")
        
        # Save experiment
        filepath = self.file_manager.save_experiment_results("temperature_sweep", results)
        print(f"\n✅ Experimento guardado en: {filepath}")
    
    def _length_variation_experiment(self, generator) -> None:
        """Length variation experiment."""
        seed = self.validator.get_text_input("🌱 Seed fijo", "In this story", 1, 100)
        temperature = self.validator.get_float_input("🌡️ Temperature fija", 0.8, 0.1, 2.0)
        
        lengths = [100, 200, 300, 400, 500]
        results = []
        
        for i, length in enumerate(lengths, 1):
            print(f"\n📏 Longitud {i}/{len(lengths)}: {length}")
            start_time = time.time()
            result = generator.generate(seed, length, temperature)
            gen_time = time.time() - start_time
            
            results.append({
                'seed': seed,
                'temperature': temperature,
                'length': length,
                'result': result,
                'generation_time': gen_time
            })
            
            words = len(result.split())
            print(f"   ⏱️ {gen_time:.2f}s | {words} palabras | {len(result)/gen_time:.0f} chars/s")
        
        # Save experiment
        filepath = self.file_manager.save_experiment_results("length_variation", results)
        print(f"\n✅ Experimento guardado en: {filepath}")
    
    def _matrix_experiment(self, generator) -> None:
        """Full matrix experiment with multiple parameters."""
        print("⚠️ Experimento matriz generará muchas combinaciones")
        if not self.validator.get_confirmation("¿Continuar?", False):
            return
        
        seeds = ["The power", "The secret"]
        temperatures = [0.7, 1.0, 1.3]
        lengths = [150, 250]
        
        total_combinations = len(seeds) * len(temperatures) * len(lengths)
        print(f"🔬 Generando {total_combinations} combinaciones...")
        
        results = []
        current = 0
        
        for seed in seeds:
            for temp in temperatures:
                for length in lengths:
                    current += 1
                    print(f"\n🧪 {current}/{total_combinations}: \"{seed}\" T={temp} L={length}")
                    
                    start_time = time.time()
                    result = generator.generate(seed, length, temp)
                    gen_time = time.time() - start_time
                    
                    results.append({
                        'seed': seed,
                        'temperature': temp,
                        'length': length,
                        'result': result,
                        'generation_time': gen_time
                    })
        
        # Save experiment
        filepath = self.file_manager.save_experiment_results("matrix_full", results)
        print(f"\n✅ Experimento matriz guardado en: {filepath}")
    
    def _save_last_generation(self) -> None:
        """Save the last generation to file."""
        if not self.last_generation:
            print("❌ No hay generación para guardar")
            return
        
        lg = self.last_generation
        filepath = self.file_manager.save_generation_to_file(
            lg['result'], lg['seed'], lg['temperature'], lg['length']
        )
        print(f"✅ Generación guardada en: {filepath}")
    
    def _show_session_stats(self) -> None:
        """Show current session statistics."""
        duration = datetime.now() - self.session_stats['start_time']
        avg_time = (self.session_stats['total_time'] / self.session_stats['generations_count'] 
                   if self.session_stats['generations_count'] > 0 else 0)
        avg_chars = (self.session_stats['total_chars'] / self.session_stats['generations_count']
                    if self.session_stats['generations_count'] > 0 else 0)
        
        print("\n📊 ESTADÍSTICAS DE SESIÓN:")
        print("-" * 40)
        print(f"   ⏱️ Duración de sesión: {duration}")
        print(f"   🎯 Generaciones: {self.session_stats['generations_count']}")
        print(f"   📝 Total caracteres: {self.session_stats['total_chars']:,}")
        print(f"   ⚡ Tiempo total generación: {self.session_stats['total_time']:.1f}s")
        print(f"   📊 Promedio por generación: {avg_time:.2f}s, {avg_chars:.0f} chars")
        print("-" * 40)
    
    def _show_interactive_help(self) -> None:
        """Show detailed help for interactive session."""
        print("\n📖 AYUDA DETALLADA - SESIÓN INTERACTIVA")
        print("=" * 50)
        print("🎛️ COMANDOS DE CONFIGURACIÓN:")
        print("   temp 0.8     - Cambiar temperatura (0.1-2.0)")
        print("   len 200      - Cambiar longitud (50-1000)")
        print()
        print("💾 COMANDOS DE GESTIÓN:")
        print("   save         - Guardar última generación")
        print("   stats        - Ver estadísticas de sesión")
        print("   clear        - Reiniciar estadísticas")
        print()
        print("🎯 GENERACIÓN:")
        print("   [cualquier texto] - Usar como seed para generar")
        print()
        print("📚 EJEMPLOS:")
        print("   The power of  - Genera con seed 'The power of'")
        print("   temp 1.2      - Cambia temperatura a 1.2")
        print("   len 300       - Cambia longitud a 300")
        print("   save          - Guarda último resultado")
        print("=" * 50)