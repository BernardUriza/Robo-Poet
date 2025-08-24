#!/usr/bin/env python3
"""
Display Utilities for Robo-Poet Framework

Provides consistent formatting and display functions for academic interface.

Author: ML Academic Framework
Version: 2.1
"""

import time
from typing import Dict, List, Any
from datetime import datetime


class DisplayUtils:
    """Utilities for consistent academic display formatting."""
    
    @staticmethod
    def format_generation_result(result: str, seed: str, temperature: float, 
                               length: int, gen_time: float, model_name: str = "unknown") -> None:
        """Display generation result with academic formatting."""
        print("\n" + "=" * 70)
        print("🎨 RESULTADO DE GENERACIÓN")
        print("=" * 70)
        print(f"🤖 Modelo: {model_name}")
        print(f"🌱 Seed: '{seed}'")
        print(f"🌡️ Temperature: {temperature}")
        print(f"📏 Longitud solicitada: {length}")
        print(f"📊 Longitud real: {len(result)}")
        print(f"⏱️ Tiempo de generación: {gen_time:.2f} segundos")
        print("=" * 70)
        print("\n📝 TEXTO GENERADO:")
        print("-" * 70)
        print(result)
        print("-" * 70)
    
    @staticmethod
    def format_model_info(model_info: Dict) -> None:
        """Display detailed model information."""
        print(f"📊 {model_info['name']}")
        print(f"   📅 Creado: {model_info['created'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   💾 Tamaño: {model_info['size_mb']:.1f} MB")
        print(f"   ⭐ Calidad: {model_info['quality_rating']}")
        
        if model_info['metadata']:
            metadata = model_info['metadata']
            print(f"   📈 Épocas: {metadata.get('final_epoch', 'N/A')}")
            print(f"   📉 Loss final: {metadata.get('final_loss', 'N/A'):.4f}" 
                  if isinstance(metadata.get('final_loss'), (int, float)) else "   📉 Loss final: N/A")
            print(f"   ⏱️ Tiempo entrenamiento: {metadata.get('training_duration', 'N/A')}")
        print()
    
    @staticmethod
    def show_progress_bar(current: int, total: int, prefix: str = "Progreso") -> None:
        """Display simple progress bar for academic feedback."""
        percent = (current / total) * 100
        filled = int(percent // 2)
        bar = "█" * filled + "░" * (50 - filled)
        print(f"\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
    
    @staticmethod
    def format_experiment_results(experiment_type: str, results: List[Dict]) -> None:
        """Display batch experiment results in academic format."""
        print(f"\n🔬 RESULTADOS DEL EXPERIMENTO: {experiment_type.upper()}")
        print("=" * 70)
        print(f"📊 Total de generaciones: {len(results)}")
        print("=" * 70)
        
        for i, result in enumerate(results, 1):
            print(f"\n🧪 Experimento {i}:")
            print(f"   🌱 Seed: '{result.get('seed', 'N/A')}'")
            print(f"   🌡️ Temperature: {result.get('temperature', 'N/A')}")
            print(f"   📏 Longitud: {result.get('length', 'N/A')}")
            print(f"   ⏱️ Tiempo: {result.get('generation_time', 'N/A'):.2f}s" 
                  if isinstance(result.get('generation_time'), (int, float)) else "   ⏱️ Tiempo: N/A")
            print(f"   📝 Muestra: {result.get('result', '')[:100]}..." 
                  if result.get('result') else "   📝 Sin resultado")
        
        print("\n" + "=" * 70)
    
    @staticmethod
    def show_training_header(text_file: str, epochs: int) -> None:
        """Display training start header."""
        print("\n" + "🔥" * 20 + " FASE 1: ENTRENAMIENTO INTENSIVO " + "🔥" * 20)
        print("=" * 80)
        print(f"📚 Archivo de texto: {text_file}")
        print(f"🎯 Épocas configuradas: {epochs}")
        print(f"⏰ Tiempo estimado: {epochs * 3:.0f}-{epochs * 8:.0f} minutos")
        print(f"🧠 Iniciando entrenamiento de red neuronal LSTM...")
        print("=" * 80)
    
    @staticmethod
    def show_generation_header(model_name: str, generation_mode: str) -> None:
        """Display generation start header."""
        print("\n" + "🎨" * 15 + " FASE 2: GENERACIÓN DE TEXTO " + "🎨" * 15)
        print("=" * 70)
        print(f"🤖 Modelo seleccionado: {model_name}")
        print(f"🎯 Modo de generación: {generation_mode}")
        print("=" * 70)
    
    @staticmethod
    def show_academic_tip(tip_text: str) -> None:
        """Display academic tip or educational note."""
        print(f"\n💡 CONSEJO ACADÉMICO:")
        print(f"   {tip_text}")
        print()
    
    @staticmethod
    def show_warning(warning_text: str) -> None:
        """Display warning message with academic formatting."""
        print(f"\n⚠️ ADVERTENCIA:")
        print(f"   {warning_text}")
        print()
    
    @staticmethod
    def show_error(error_text: str) -> None:
        """Display error message with academic formatting."""
        print(f"\n❌ ERROR:")
        print(f"   {error_text}")
        print()
    
    @staticmethod
    def show_success(success_text: str) -> None:
        """Display success message with academic formatting."""
        print(f"\n✅ ÉXITO:")
        print(f"   {success_text}")
        print()
    
    @staticmethod
    def format_cleanup_results(results: Dict[str, int]) -> None:
        """Display model cleanup results."""
        print(f"\n🗑️ RESULTADOS DE LIMPIEZA:")
        print("=" * 50)
        print(f"   🎯 Modelos eliminados: {results['models']}")
        print(f"   📋 Metadata eliminada: {results['metadata']}")
        print(f"   🔄 Checkpoints eliminados: {results['checkpoints']}")
        print(f"   💾 Espacio liberado: {results['total_mb']:.1f} MB")
        print("=" * 50)
    
    @staticmethod
    def pause_for_user(message: str = "Presiona Enter para continuar...") -> None:
        """Academic pause for user to read information."""
        input(f"\n📖 {message}")
    
    @staticmethod
    def clear_screen() -> None:
        """Clear screen for clean academic interface (secure implementation)."""
        import subprocess
        import sys
        
        try:
            if sys.platform.startswith('win'):
                subprocess.run(['cls'], shell=False, check=False, capture_output=True, timeout=5)
            else:
                subprocess.run(['clear'], shell=False, check=False, capture_output=True, timeout=5)
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # Fallback to ANSI escape sequences if system commands fail
            print('\\033[2J\\033[H', end='')
    
    @staticmethod
    def format_timestamp() -> str:
        """Get formatted timestamp for academic logging."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def show_model_stats_summary(metadata: Dict) -> None:
        """Display concise model statistics."""
        print("📊 ESTADÍSTICAS DEL MODELO:")
        print("-" * 40)
        
        if metadata:
            print(f"   📈 Épocas completadas: {metadata.get('final_epoch', 'N/A')}")
            print(f"   📉 Loss final: {metadata.get('final_loss', 'N/A'):.4f}" 
                  if isinstance(metadata.get('final_loss'), (int, float)) else "   📉 Loss final: N/A")
            print(f"   🎯 Vocabulario: {metadata.get('vocab_size', 'N/A')} caracteres")
            print(f"   📏 Secuencias: {metadata.get('sequence_length', 'N/A')} tokens")
            print(f"   ⚙️ LSTM units: {metadata.get('lstm_units', 'N/A')}")
            
            # Calculate quality assessment
            final_loss = metadata.get('final_loss')
            if isinstance(final_loss, (int, float)):
                if final_loss < 1.0:
                    quality = "🌟 Excelente para generación creativa"
                elif final_loss < 1.5:
                    quality = "⭐ Bueno para textos coherentes" 
                elif final_loss < 2.0:
                    quality = "📊 Aceptable para experimentación"
                else:
                    quality = "⚠️ Requiere más entrenamiento"
                print(f"   🎯 Calidad: {quality}")
        else:
            print("   ⚠️ Metadata no disponible")
        
        print("-" * 40)