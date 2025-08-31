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
# GPU info fallback for compatibility
try:
    from gpu_detection import get_gpu_info
except ImportError:
    def get_gpu_info():
        """Fallback GPU info function."""
        return {"gpu_available": False, "gpu_name": "N/A", "memory": "N/A"}


class AcademicMenuSystem:
    """Main menu system for academic interface."""
    
    def __init__(self):
        """Initialize menu system with configuration."""
        self.config = get_config()
        self.model_config = self.config.model
    
    def show_header(self):
        """Display academic framework header."""
        print("=" * 75)
        print("🎓 ROBO-POET: ACADEMIC NEURAL TEXT GENERATION FRAMEWORK")
        print("=" * 75)
        print("📚 Version: 2.1 - Enhanced with Deep Analysis Suite")
        print("🏛️ Features: Training • Generation • Analysis • Diagnosis")
        print("🔬 New: Gradient Analysis & Loss Landscape Detection")
        print("⚡ Hardware: Optimized for NVIDIA RTX 2000 Ada + WSL2")
        print("=" * 75)
    
    def show_main_menu(self) -> str:
        """Display main academic menu and get user choice."""
        print("\n🎯 MENÚ ACADÉMICO PRINCIPAL")
        print("=" * 50)
        print("🎓 FLUJO DE TRABAJO ACADÉMICO:")
        print("1. 🚨 TELARES DETECTOR: Entrenamiento Anti-Pirámides (5-10 min)")
        print("2. 🎨 FASE 2: Generación de Texto (Estudio Avanzado)")
        print("3. 📊 Ver Modelos Disponibles")
        print()
        print("🔬 ANÁLISIS Y DIAGNÓSTICO (NUEVO):")
        print("4. 🏥 HOSPITAL: Cirugía de Gates LSTM")
        print("5. 🔬 ANÁLISIS: Gradient Flow & Loss Landscape")
        print()
        print("📊 GESTIÓN Y MONITOREO:")
        print("6. 📈 Monitorear Progreso de Entrenamiento")
        print("7. 🧹 Limpiar Todos los Modelos")
        print()
        print("🧪 TESTING Y VALIDACIÓN:")
        print("8. 🧪 Suite de Tests Módulo 2 (Demo + Validación)")
        print()
        print("📁 ARCHIVOS Y VISUALIZACIÓN:")
        print("A. 📝 Ver Logs y Archivos Generados")
        print("B. 📈 Explorar Visualizaciones y Gráficos")
        print()
        print("🔬 HERRAMIENTAS AVANZADAS:")
        print("C. 🎯 Attention Mechanism Demo & Validation")
        print("D. 🏗️ Dataset Preprocessing Pipeline")
        print()
        print("⚙️ SISTEMA:")
        print("9. ⚙️ Configuración y Estado del Sistema") 
        print("0. 🚪 Salir del Sistema")
        print("=" * 50)
        
        choice = input("🎯 Selecciona una opción (0-9, A-D): ").strip().upper()
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
    
    def run_main_loop(self):
        """Execute the main menu loop."""
        while True:
            self.show_header()
            choice = self.show_main_menu()
            
            if choice == "1":
                # TELARES DETECTOR - Integrated Phase 1 Replacement
                self._handle_telares_detector_menu()
                
            elif choice == "2":
                # Phase 2 - Text Generation with Auto-Amplification
                self._handle_phase2_generation_with_amplification()
                
            elif choice == "3":
                # Show available models
                self._show_available_models()
                
            elif choice == "4":
                # LSTM Hospital (original diagnostic functionality)
                print("\n🏥 LSTM HOSPITAL - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "5":
                # Analysis (original functionality)
                print("\n🔬 ANÁLISIS - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "6":
                # Monitor training
                print("\n📈 MONITOREO - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "7":
                # Clean models
                print("\n🧹 LIMPIEZA - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "8":
                # Testing suite
                print("\n🧪 TESTING - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "A":
                # View logs
                print("\n📝 LOGS - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "B":
                # Visualizations
                print("\n📈 VISUALIZACIONES - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "C":
                # Attention demo
                print("\n🎯 ATTENTION DEMO - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "D":
                # Dataset preprocessing
                print("\n🏗️ PREPROCESSING - Funcionalidad en desarrollo")
                input("Presiona Enter para continuar...")
                
            elif choice == "9":
                # System configuration
                self.show_system_status()
                
            elif choice == "0":
                # Exit
                self.show_exit_message()
                break
                
            else:
                print("\n❌ Opción inválida. Selecciona 0-9 o A-D.")
                input("Presiona Enter para continuar...")
    
    def _show_available_models(self):
        """Show available trained models."""
        print("\n📊 MODELOS DISPONIBLES")
        print("=" * 40)
        
        from pathlib import Path
        models_dir = Path("models")
        
        if not models_dir.exists():
            print("⚠️ Directorio 'models' no encontrado")
            print("💡 Entrena un modelo primero con TELARES DETECTOR")
        else:
            model_files = list(models_dir.glob("*.keras")) + list(models_dir.glob("*.joblib"))
            
            if not model_files:
                print("⚠️ No hay modelos entrenados")
                print("💡 Usa TELARES DETECTOR para entrenar tu primer modelo")
            else:
                print(f"🎯 Modelos encontrados: {len(model_files)}")
                for i, model_file in enumerate(model_files, 1):
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    print(f"   {i}. {model_file.name} ({size_mb:.1f} MB)")
        
        print()
        input("📖 Presiona Enter para continuar...")
    
    def _handle_telares_detector_menu(self):
        """Handle Telares Detector integrated menu system"""
        from src.infrastructure.container import get_service_locator
        
        try:
            # Get integrated services
            service_locator = get_service_locator()
            telares_training_service = service_locator.get_telares_training_service()
            telares_detection_service = service_locator.get_telares_detection_service()
            
            while True:
                self._show_telares_menu()
                choice = input("\n🎯 Selecciona una opción: ").strip()
                
                if choice == "1":
                    # Standard training
                    self._run_telares_standard_training(telares_training_service)
                elif choice == "2":
                    # Hybrid training with poetic corpus
                    self._run_telares_hybrid_training(telares_training_service)
                elif choice == "3":
                    # Analyze sample messages
                    self._run_telares_analysis_demo(telares_detection_service)
                elif choice == "4":
                    # System status
                    self._show_telares_system_status(telares_detection_service, telares_training_service)
                elif choice == "0":
                    break
                else:
                    print("❌ Opción inválida. Selecciona 0-4.")
                    input("Presiona Enter para continuar...")
                    
        except Exception as e:
            print(f"❌ Error en sistema Telares: {e}")
            input("Presiona Enter para continuar...")
    
    def _show_telares_menu(self):
        """Show Telares Detector menu options"""
        print("\n🚨 TELARES DETECTOR - DETECCIÓN DE ESQUEMAS PIRAMIDALES")
        print("=" * 65)
        print("🛡️  SISTEMA ANTI-MANIPULACIÓN INTEGRADO EN ROBO-POET")
        print("⚡ Compatible WSL2 + Scikit-Learn (sin PyTorch)")
        print("📊 Dataset: 147 mensajes reales de grupos de WhatsApp/Telegram")
        print()
        print("🎯 OPCIONES DE ENTRENAMIENTO:")
        print("1. 🔥 ENTRENAR DETECTOR ESTÁNDAR (Solo dataset Telares)")
        print("2. 🧪 ENTRENAR MODELO HÍBRIDO (Telares + Corpus Poético)")
        print("3. 🔬 ANÁLISIS DEMO (Probar detector con mensajes)")
        print("4. 📊 ESTADO DEL SISTEMA")
        print("0. 🔙 Volver al menú principal")
        print("=" * 65)
    
    def _run_telares_standard_training(self, training_service):
        """Run standard telares training"""
        print("\n🔥 ENTRENAMIENTO ESTÁNDAR TELARES DETECTOR")
        print("=" * 50)
        print("📊 Usando dataset de 147 mensajes reales de esquemas piramidales")
        print("⚡ Tiempo estimado: 2-5 minutos")
        
        confirm = input("\n¿Iniciar entrenamiento estándar? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes', 'sí', 's']:
            try:
                print("\n🚀 Iniciando entrenamiento...")
                metrics = training_service.train_standard_model()
                
                print("✅ ENTRENAMIENTO COMPLETADO")
                print(f"⏱️  Tiempo: {metrics.get('training_time', 0):.1f} segundos")
                print(f"📊 Mensajes entrenados: {metrics.get('dataset_size', 0)}")
                print(f"💾 Modelo guardado en: {metrics.get('model_path', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Error en entrenamiento: {e}")
        
        input("\nPresiona Enter para continuar...")
    
    def _run_telares_hybrid_training(self, training_service):
        """Run hybrid training with poetic corpus"""
        print("\n🧪 ENTRENAMIENTO HÍBRIDO (TELARES + CORPUS POÉTICO)")
        print("=" * 60)
        print("🔬 Metodología científica: controles negativos")
        print("📚 Corpus poético = ejemplos sin manipulación")
        print("🎯 Mejor precisión esperada en detección real")
        
        confirm = input("\n¿Iniciar entrenamiento híbrido científico? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes', 'sí', 's']:
            try:
                print("\n🚀 Iniciando entrenamiento híbrido...")
                metrics = training_service.train_hybrid_model()
                
                print("✅ ENTRENAMIENTO HÍBRIDO COMPLETADO")
                print(f"⏱️  Tiempo: {metrics.get('training_time', 0):.1f} segundos")
                print(f"📨 Mensajes telares: {metrics.get('telares_messages', 0)}")
                print(f"📚 Fragmentos poéticos: {metrics.get('poetic_fragments', 0)}")
                print(f"📊 Dataset total: {metrics.get('total_dataset_size', 0)}")
                print(f"💾 Modelo híbrido guardado")
                
            except Exception as e:
                print(f"❌ Error en entrenamiento híbrido: {e}")
        
        input("\nPresiona Enter para continuar...")
    
    def _run_telares_analysis_demo(self, detection_service):
        """Run analysis demo with sample messages"""
        print("\n🔬 DEMO DE ANÁLISIS - DETECTOR EN ACCIÓN")
        print("=" * 50)
        
        sample_messages = [
            "Únete a este negocio increíble donde ganarás millones sin esfuerzo. Solo necesitas invitar a 3 personas y ellas invitarán a otras 3. ¡Es muy fácil!",
            "Hermosa tarde para leer poesía en el parque.",
            "URGENTE! Últimos cupos disponibles. Esta oportunidad única cambiará tu vida para siempre. Dios te está dando esta bendición.",
            "¿Alguien sabe dónde comprar libros usados en la ciudad?",
            "Yo logré ganar $50,000 el primer mes. Gracias a este sistema mi vida cambió completamente. Ahora trabajo desde casa y viajo por el mundo."
        ]
        
        print("📝 Analizando 5 mensajes de ejemplo...")
        print()
        
        try:
            for i, message in enumerate(sample_messages, 1):
                result = detection_service.analyze_message(message)
                
                print(f"📨 MENSAJE {i}:")
                print(f"   '{message[:60]}...' " if len(message) > 60 else f"   '{message}'")
                print(f"   🎯 Riesgo: {result['risk_level']}")
                print(f"   📊 Puntuación: {result['total_score']:.2f}/7.0")
                print(f"   🚨 Tácticas detectadas: {', '.join(result['detected_tactics']) if result['detected_tactics'] else 'Ninguna'}")
                print()
                
        except Exception as e:
            print(f"❌ Error en análisis demo: {e}")
        
        input("📖 Presiona Enter para continuar...")
    
    def _show_telares_system_status(self, detection_service, training_service):
        """Show Telares system status"""
        print("\n📊 ESTADO DEL SISTEMA TELARES DETECTOR")
        print("=" * 50)
        
        try:
            det_status = detection_service.get_system_status()
            train_status = training_service.get_training_status()
            
            print("🔍 DETECTOR:")
            print(f"   🎯 Estado: {'✅ Listo' if det_status.get('ready_for_detection') else '❌ No listo'}")
            print(f"   🧠 Modelo: {'✅ Cargado' if det_status.get('detector_loaded') else '❌ No cargado'}")
            print(f"   📋 Versión: {det_status.get('model_version', 'N/A')}")
            print(f"   🏷️  Tácticas: {len(det_status.get('supported_tactics', []))}")
            
            print("\n🎓 ENTRENAMIENTO:")
            print(f"   📊 Disponible: {'✅ Sí' if train_status.get('data_loader_ready') else '❌ No'}")
            print(f"   📁 Datasets: {len(train_status.get('available_datasets', []))}")
            
            datasets = train_status.get('available_datasets', [])
            for dataset in datasets:
                print(f"      • {dataset}")
            
            last_training = train_status.get('last_training_metrics', {})
            if last_training:
                print(f"\n🏆 ÚLTIMO ENTRENAMIENTO:")
                print(f"   ⏱️  Duración: {last_training.get('training_time', 0):.1f}s")
                print(f"   📊 Dataset: {last_training.get('dataset_size', 0)} mensajes")
                
        except Exception as e:
            print(f"❌ Error obteniendo estado: {e}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def _handle_phase2_generation_with_amplification(self):
        """Handle Phase 2 text generation with automatic Telares amplification"""
        from src.infrastructure.container import get_service_locator
        
        try:
            service_locator = get_service_locator()
            generation_service = service_locator.get_generation_service()
            telares_training_service = service_locator.get_telares_training_service()
            
            print("\n🎨 FASE 2: GENERACIÓN DE TEXTO + AMPLIFICACIÓN AUTOMÁTICA")
            print("=" * 70)
            print("🧠 Generación de texto con modelo entrenado")
            print("🤖 Auto-amplificación de dataset Telares con muestras sintéticas")
            print("🔄 Re-entrenamiento automático del detector")
            print()
            
            # Check for available models
            available_models = self._get_available_generation_models()
            if not available_models:
                print("❌ No hay modelos de generación disponibles")
                print("💡 Ejecuta FASE 1 primero para entrenar un modelo")
                input("Presiona Enter para continuar...")
                return
            
            # Show generation options
            print("🎯 OPCIONES DE GENERACIÓN:")
            print("1. 🎨 Generación normal (sin amplificación)")
            print("2. 🔥 Generación + Auto-amplificación Telares")
            print("3. 🧪 Solo amplificación (sin generación visible)")
            print("0. 🔙 Volver")
            
            choice = input("\n🎯 Selecciona una opción: ").strip()
            
            if choice == "1":
                self._run_normal_generation(generation_service, available_models)
            elif choice == "2":
                self._run_generation_with_amplification(generation_service, telares_training_service, available_models)
            elif choice == "3":
                self._run_silent_amplification(generation_service, telares_training_service, available_models)
            elif choice == "0":
                return
            else:
                print("❌ Opción inválida")
                input("Presiona Enter para continuar...")
                
        except Exception as e:
            print(f"❌ Error en Phase 2: {e}")
            input("Presiona Enter para continuar...")
    
    def _get_available_generation_models(self) -> List[str]:
        """Get list of available trained generation models"""
        from pathlib import Path
        
        models = []
        
        # Check PyTorch models
        pytorch_models = Path("robo-poet-pytorch/checkpoints")
        if pytorch_models.exists():
            pth_files = list(pytorch_models.glob("*.pth"))
            for model_file in pth_files:
                if "best" in model_file.name:
                    models.append(f"PyTorch: {model_file.name}")
        
        # Check legacy models
        models_dir = Path("models")
        if models_dir.exists():
            keras_files = list(models_dir.glob("*.keras"))
            for model_file in keras_files:
                models.append(f"Legacy: {model_file.name}")
        
        return models
    
    def _run_normal_generation(self, generation_service, available_models):
        """Run normal text generation without amplification"""
        print("\n🎨 GENERACIÓN NORMAL")
        print("=" * 30)
        print("🔄 Generando texto sin amplificación automática...")
        
        # For now, show placeholder
        print("💡 Funcionalidad de generación normal disponible")
        print("🎯 Use la interfaz existente de robo_poet.py --generate")
        
        input("Presiona Enter para continuar...")
    
    def _run_generation_with_amplification(self, generation_service, telares_training_service, available_models):
        """Run text generation with automatic Telares amplification"""
        print("\n🔥 GENERACIÓN + AUTO-AMPLIFICACIÓN")
        print("=" * 45)
        print("🎨 Genera texto visible para el usuario")
        print("🤖 Crea muestras sintéticas etiquetadas automáticamente")
        print("🔄 Re-entrena detector Telares con dataset ampliado")
        print()
        
        confirm = input("¿Ejecutar generación con amplificación automática? [Y/n]: ").strip().lower()
        if confirm not in ['', 'y', 'yes', 'sí', 's']:
            return
        
        try:
            # Mock model ID for demonstration
            model_id = "best_trained_model"
            
            print("🚀 Iniciando proceso de amplificación...")
            print("⏱️  Tiempo estimado: 2-3 minutos")
            print()
            
            # Step 1: Generate synthetic samples
            print("🤖 PASO 1: Generando muestras sintéticas...")
            amplification_result = generation_service.auto_amplify_telares_dataset(
                model_id=model_id,
                target_samples=30
            )
            
            if amplification_result["success"]:
                print(f"✅ Generadas {amplification_result['total_generated']} muestras")
                print(f"   🚨 Manipulativas: {amplification_result['manipulative_samples']}")
                print(f"   🧪 Controles: {amplification_result['control_samples']}")
            else:
                raise Exception(amplification_result.get("error", "Generation failed"))
            
            # Step 2: Auto-retrain Telares detector
            print("\n🔄 PASO 2: Re-entrenando detector Telares...")
            retrain_result = telares_training_service.auto_retrain_with_amplification(amplification_result)
            
            if retrain_result["auto_retrain"]:
                stats = retrain_result["improvement_stats"]
                print("✅ AUTO-REENTRENAMIENTO COMPLETADO")
                print(f"📊 Crecimiento dataset: {stats['dataset_growth']:.1f}x")
                print(f"🆕 Nuevas muestras: {stats['new_samples']}")
                print(f"📈 Tamaño total: {stats['total_size']} mensajes")
                print("\n🎉 SISTEMA MEJORADO AUTOMÁTICAMENTE")
                print("🛡️ Telares detector ahora es más preciso")
            else:
                print(f"⚠️ Re-entrenamiento falló: {retrain_result.get('reason', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error en amplificación: {e}")
        
        input("\n📖 Presiona Enter para continuar...")
    
    def _run_silent_amplification(self, generation_service, telares_training_service, available_models):
        """Run silent amplification without showing generated text"""
        print("\n🧪 AMPLIFICACIÓN SILENCIOSA")
        print("=" * 35)
        print("🤖 Genera y procesa muestras sin mostrarlas")
        print("🎯 Enfoque: Solo mejora del detector Telares")
        print("⚡ Proceso optimizado y rápido")
        print()
        
        samples = input("¿Cuántas muestras generar? [30]: ").strip()
        target_samples = int(samples) if samples.isdigit() else 30
        
        confirm = input(f"¿Generar {target_samples} muestras para amplificación? [Y/n]: ").strip().lower()
        if confirm not in ['', 'y', 'yes', 'sí', 's']:
            return
        
        try:
            model_id = "best_trained_model"
            
            print(f"🚀 Generando {target_samples} muestras silenciosamente...")
            
            # Generate and integrate directly
            amplification_result = generation_service.auto_amplify_telares_dataset(
                model_id=model_id,
                target_samples=target_samples
            )
            
            if amplification_result["success"]:
                retrain_result = telares_training_service.auto_retrain_with_amplification(amplification_result)
                
                if retrain_result["auto_retrain"]:
                    print("✅ AMPLIFICACIÓN SILENCIOSA COMPLETADA")
                    print(f"📊 Dataset mejorado: +{retrain_result['improvement_stats']['new_samples']} muestras")
                    print("🛡️ Detector Telares optimizado automáticamente")
                else:
                    print("⚠️ Amplificación parcialmente exitosa")
            else:
                print(f"❌ Amplificación falló: {amplification_result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Error en amplificación silenciosa: {e}")
        
        input("\n📖 Presiona Enter para continuar...")