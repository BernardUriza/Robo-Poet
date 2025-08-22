#!/usr/bin/env python3
"""
Academic Menu System for Robo-Poet Framework

Provides clean, educational interface for students and researchers.
Handles main navigation and system information display.

Author: ML Academic Framework  
Version: 2.1
"""

from typing import Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from gpu_detection import get_gpu_info


class AcademicMenuSystem:
    """Main menu system for academic interface."""
    
    def __init__(self):
        """Initialize menu system with configuration."""
        self.config = get_config()
        self.model_config = self.config.model
    
    def show_header(self):
        """Display academic framework header."""
        print("=" * 70)
        print("🎓 ROBO-POET: ACADEMIC NEURAL TEXT GENERATION FRAMEWORK")
        print("=" * 70)
        print("📚 Version: 2.1 - Enhanced Phase 2 Generation Studio")
        print("🏛️ Academic Interface: Two-Phase Learning System")
        print("⚡ Hardware: Optimized for NVIDIA RTX 2000 Ada + WSL2")
        print("=" * 70)
    
    def show_main_menu(self) -> str:
        """Display main academic menu and get user choice."""
        print("\n🎯 MENÚ ACADÉMICO PRINCIPAL")
        print("=" * 50)
        print("🎓 FLUJO DE TRABAJO ACADÉMICO:")
        print("1. 🔥 FASE 1: Entrenamiento Intensivo (1+ hora)")
        print("2. 🎨 FASE 2: Generación de Texto (Estudio Avanzado)")
        print()
        print("📊 GESTIÓN Y MONITOREO:")
        print("3. 📊 Ver Modelos Disponibles")
        print("4. 📈 Monitorear Progreso de Entrenamiento")
        print("5. 🧹 Limpiar Todos los Modelos")
        print()
        print("⚙️ SISTEMA:")
        print("6. ⚙️ Configuración y Estado del Sistema")
        print("7. 🚪 Salir del Sistema")
        print("=" * 50)
        
        choice = input("🎯 Selecciona una opción (1-7): ").strip()
        return choice
    
    def show_system_status(self):
        """Display comprehensive system status for academic purposes."""
        print("\n⚙️ CONFIGURACIÓN Y ESTADO DEL SISTEMA")
        print("=" * 60)
        
        # GPU Information
        gpu_info = get_gpu_info()
        gpu_available = gpu_info['gpu_available']
        
        print("🖥️ HARDWARE:")
        print(f"   🎯 GPU: {gpu_info['gpu_name']}")
        print(f"   🔧 Driver: {gpu_info['driver_version']}")
        print(f"   💾 VRAM: {gpu_info['memory_total']}")
        print(f"   🎯 GPU disponible: {'✅ Sí' if gpu_available else '❌ No'}")
        
        if not gpu_available:
            print("\n🚨 AVISO ACADÉMICO:")
            print("   GPU NVIDIA es obligatoria para entrenamiento.")
            print("   Comando directo funciona: python robo_poet.py --text archivo.txt --epochs N")
        
        print(f"\n🧠 CONFIGURACIÓN DEL MODELO:")
        print(f"   📦 Batch size: {self.model_config.batch_size}")
        print(f"   🧠 LSTM units: {self.model_config.lstm_units}")
        print(f"   📏 Sequence length: {self.model_config.sequence_length}")
        print(f"   💧 Dropout rate: {self.model_config.dropout_rate}")
        
        print(f"\n🔧 SOFTWARE:")
        print(f"   🐍 TensorFlow: {gpu_info['tf_version']}")
        print(f"   🎯 CUDA: {gpu_info['cuda_version']}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def show_exit_message(self):
        """Display academic exit message."""
        print("\n" + "=" * 60)
        print("🎓 ¡GRACIAS POR USAR ROBO-POET ACADEMIC FRAMEWORK!")
        print("=" * 60)
        print("📚 Sistema de aprendizaje de IA desarrollado para estudiantes")
        print("🧠 Implementación educacional de redes LSTM para generación de texto")
        print("⚡ Optimizado para GPU NVIDIA en entornos WSL2 + Linux")
        print()
        print("💡 RECURSOS ACADÉMICOS:")
        print("   📖 Documentación: readme.md")
        print("   🎯 Metodología: CLAUDE.md")
        print("   🔧 Código fuente: src/")
        print("=" * 60)
        print("🎯 ¡Continúa explorando el mundo del Machine Learning! 🚀")
        print("=" * 60)