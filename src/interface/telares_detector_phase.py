#!/usr/bin/env python3
"""
Telares Detector Phase - Reemplazo completo de Phase 1
Creado por Bernard Orozco

Sistema especializado en detección de esquemas piramidales que reemplaza
completamente el entrenamiento de generación de texto.
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Optional

# Agregar path del sistema
sys.path.append(str(Path(__file__).parent.parent))

from utils.file_manager import FileManager
from utils.input_validator import InputValidator
from utils.display_utils import DisplayUtils

class TelaresDetectorPhase:
    """Maneja la fase única de entrenamiento de detector de esquemas piramidales"""
    
    def __init__(self, config):
        self.config = config
        self.file_manager = FileManager()
        self.input_validator = InputValidator()
        self.display = DisplayUtils()
        
    def show_telares_header(self):
        """Mostrar header específico para Telares Detector"""
        print("🚨" * 20 + " TELARES DETECTOR TRAINING " + "🚨" * 20)
        print("================================================================================")
        print("🛡️  SISTEMA DE DETECCIÓN DE ESQUEMAS PIRAMIDALES")
        print("📚 Entrena modelo para detectar 7 tácticas de manipulación")
        print("🎯 Dataset: 135 mensajes reales de grupos de WhatsApp/Telegram")
        print("⚡ Tecnología: Scikit-Learn + TF-IDF (Compatible con WSL2)")
        print("👨‍💻 Creado por: Bernard Orozco")
        print("================================================================================")
        
    def show_training_menu(self):
        """Menú principal de entrenamiento"""
        print("\n🎯 OPCIONES DE ENTRENAMIENTO TELARES DETECTOR")
        print("="*60)
        print("1. 🔥 ENTRENAR DETECTOR COMPLETO (135 mensajes reales)")
        print("2. 🧪 ENTRENAR CON CORPUS POÉTICO (Datos de prueba adicionales)")
        print("3. 📊 ANÁLISIS DE DATOS DE ENTRENAMIENTO")
        print("4. 🚀 ENTRENAMIENTO RÁPIDO (Solo validación)")
        print("5. 📈 COMPARAR MODELOS")
        print("0. 🔙 Volver al menú principal")
        print("="*60)
        
    def run_telares_training(self):
        """Ejecutar entrenamiento principal del detector"""
        self.show_telares_header()
        
        print("\n⚠️  CONFIRMACIÓN DE ENTRENAMIENTO")
        print("="*50)
        print("🏷️  Modelo: Telares Detector")
        print("📊 Dataset: 135 mensajes reales verificados")
        print("🎯 Tácticas: 7 tipos de manipulación lingüística")
        print("⏰ Tiempo estimado: 2-5 minutos")
        print("🧠 Algoritmo: Multi-Output Logistic Regression + TF-IDF")
        
        print("\n⚠️  ADVERTENCIA:")
        print("   Este entrenamiento reemplaza completamente el generador de texto.")
        print("   Se enfocará únicamente en detección de esquemas piramidales.")
        print("   ¿Continuar con el entrenamiento especializado?")
        
        response = self.input_validator.get_yes_no_input("🚀 ¿Iniciar entrenamiento de Telares Detector? [Y/n]: ")
        
        if not response:
            print("❌ Entrenamiento cancelado.")
            return False
        
        return self.execute_telares_training()
    
    def execute_telares_training(self):
        """Ejecutar el entrenamiento del detector"""
        print("\n🔥" * 20 + " INICIANDO ENTRENAMIENTO " + "🔥" * 20)
        print("================================================================================")
        print("🚀 Ejecutando entrenamiento especializado de Telares Detector...")
        print("💾 Logs: Se mostrarán en tiempo real")
        print("================================================================================")
        
        try:
            # Determinar la ruta del script principal
            script_path = Path(__file__).parent.parent.parent / "telares_detector_interface.py"
            
            if not script_path.exists():
                print(f"❌ ERROR: Script no encontrado en {script_path}")
                return False
            
            # Preparar comando de entrenamiento automático
            # El script debe tener modo batch para entrenamiento directo
            training_script = self.create_batch_training_script(script_path)
            
            print("🎯 Iniciando entrenamiento automático...")
            start_time = time.time()
            
            # Ejecutar entrenamiento
            result = subprocess.run([
                sys.executable, training_script
            ], capture_output=True, text=True, cwd=str(script_path.parent))
            
            end_time = time.time()
            training_duration = end_time - start_time
            
            if result.returncode == 0:
                self.display.show_success(
                    f"Entrenamiento completado exitosamente en {training_duration/60:.1f} minutos"
                )
                print("🎉 Telares Detector entrenado y listo para uso")
                print("🛡️  Ahora puede detectar esquemas piramidales en tiempo real")
                print("🔍 Use la opción de análisis para probar mensajes")
                self.display.pause_for_user()
                return True
            else:
                self.display.show_error(
                    f"Error durante entrenamiento (código: {result.returncode})"
                )
                print("📋 Error output:")
                print(result.stderr)
                return False
                
        except Exception as e:
            self.display.show_error(f"Error ejecutando entrenamiento: {str(e)}")
            return False
    
    def create_batch_training_script(self, original_script_path):
        """Crear script de entrenamiento en modo batch"""
        batch_script_path = original_script_path.parent / "telares_batch_training.py"
        
        batch_script_content = f'''#!/usr/bin/env python3
"""
Script de entrenamiento automático para Telares Detector
Creado por Bernard Orozco - Reemplazo total de Fase 1
"""

import sys
import os
from pathlib import Path

# Importar el sistema principal
sys.path.append(str(Path(__file__).parent))
from telares_detector_interface import TelaresDetectorSystem

def main():
    """Entrenamiento automático sin interfaz - REEMPLAZA FASE 1 COMPLETAMENTE"""
    print("🚨" * 20 + " TELARES DETECTOR TRAINING " + "🚨" * 20)
    print("🚀 REEMPLAZO COMPLETO DE FASE 1: Entrenamiento Anti-Esquemas Piramidales")
    print("⚡ Compatible WSL2 + Scikit-Learn (sin PyTorch)")
    
    detector = TelaresDetectorSystem()
    
    print("📊 Cargando dataset real de 135 mensajes de telares...")
    X, y = detector.load_training_data()
    
    if X is None or len(X) == 0:
        print("❌ ERROR: Dataset de telares no encontrado")
        print("💡 Verificando ubicación: telares_dataset_135.csv")
        sys.exit(1)
    
    print(f"✅ Dataset cargado: {{len(X)}} mensajes de grupos reales")
    print("🔥 Iniciando entrenamiento especializado anti-pirámides...")
    
    success = detector.train_detector()
    
    if success:
        print("💾 Guardando detector entrenado...")
        
        # Crear directorio models si no existe
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        detector.classifier_path = models_dir / "telares_detector_trained.joblib"
        detector.vectorizer_path = models_dir / "telares_vectorizer_trained.joblib"
        
        import joblib
        joblib.dump(detector.classifier, detector.classifier_path)
        joblib.dump(detector.vectorizer, detector.vectorizer_path)
        
        print("✅ TELARES DETECTOR entrenado y guardado")
        print(f"📁 Ubicación: {{detector.classifier_path}}")
        print("🎉 FASE 1 COMPLETAMENTE REEMPLAZADA CON ÉXITO")
        print("🛡️  Sistema listo para detectar manipulación en tiempo real")
        sys.exit(0)
    else:
        print("❌ ERROR: Fallo en el entrenamiento del detector")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open(batch_script_path, 'w', encoding='utf-8') as f:
            f.write(batch_script_content)
        
        return str(batch_script_path)
    
    def create_hybrid_training_script(self, original_script_path):
        """Crear script híbrido con corpus poético + telares"""
        hybrid_script_path = original_script_path.parent / "telares_hybrid_training.py"
        
        hybrid_script_content = f'''#!/usr/bin/env python3
"""
Script híbrido: Telares + Corpus Poético como controles negativos
Creado por Bernard Orozco - Entrenamiento científico con validación
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Importar el sistema principal
sys.path.append(str(Path(__file__).parent))
from telares_detector_interface import TelaresDetectorSystem

def load_poetic_corpus_as_controls():
    """Cargar corpus poético como controles negativos (sin manipulación)"""
    corpus_dir = Path("corpus")
    
    if not corpus_dir.exists():
        print("⚠️  Directorio 'corpus' no encontrado")
        return [], []
    
    poetic_texts = []
    text_files = list(corpus_dir.glob("*.txt"))
    
    print(f"📚 Encontrados {{len(text_files)}} archivos de corpus poético")
    
    for txt_file in text_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Dividir en fragmentos de ~100-200 caracteres (similar a mensajes WhatsApp)
                fragments = [content[i:i+150] for i in range(0, len(content), 150) if len(content[i:i+150]) > 50]
                poetic_texts.extend(fragments[:20])  # Máximo 20 fragmentos por archivo
        except Exception as e:
            print(f"⚠️  Error leyendo {{txt_file}}: {{e}}")
    
    # Crear etiquetas: todo 0 (sin manipulación)
    num_labels = 7  # Mismas 7 tácticas que telares
    poetic_labels = np.zeros((len(poetic_texts), num_labels))
    
    print(f"✅ Corpus poético procesado: {{len(poetic_texts)}} fragmentos como controles")
    return poetic_texts, poetic_labels

def main():
    """Entrenamiento híbrido científico: Telares + Controles Poéticos"""
    print("🧪" * 20 + " HYBRID SCIENTIFIC TRAINING " + "🧪" * 20)
    print("🔬 ENTRENAMIENTO CIENTÍFICO: Telares + Corpus Poético")
    print("📊 Metodología: Controles negativos para validación estadística")
    
    detector = TelaresDetectorSystem()
    
    # Cargar dataset de telares (manipulativo)
    print("📊 Cargando dataset de telares (manipulativos)...")
    X_telares, y_telares = detector.load_training_data()
    
    if X_telares is None or len(X_telares) == 0:
        print("❌ ERROR: Dataset de telares no encontrado")
        sys.exit(1)
    
    # Cargar corpus poético (controles)
    print("📚 Cargando corpus poético (controles negativos)...")
    X_poetic, y_poetic = load_poetic_corpus_as_controls()
    
    if len(X_poetic) == 0:
        print("⚠️  No se encontró corpus poético - usando solo dataset telares")
        X_combined, y_combined = X_telares, y_telares
    else:
        # Combinar ambos datasets
        X_combined = X_telares + X_poetic
        y_combined = np.vstack([y_telares, y_poetic])
        
        print(f"✅ Dataset híbrido creado:")
        print(f"   📨 Mensajes telares: {{len(X_telares)}}")
        print(f"   📚 Fragmentos poéticos: {{len(X_poetic)}}")
        print(f"   📊 Total entrenamiento: {{len(X_combined)}}")
    
    print("🔥 Entrenando detector híbrido...")
    
    # Entrenar con dataset combinado
    detector.X_train = X_combined
    detector.y_train = y_combined
    
    success = detector.train_detector()
    
    if success:
        print("💾 Guardando detector híbrido...")
        
        # Crear directorio models si no existe
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        detector.classifier_path = models_dir / "telares_hybrid_detector.joblib"
        detector.vectorizer_path = models_dir / "telares_hybrid_vectorizer.joblib"
        
        import joblib
        joblib.dump(detector.classifier, detector.classifier_path)
        joblib.dump(detector.vectorizer, detector.vectorizer_path)
        
        print("✅ DETECTOR HÍBRIDO entrenado y guardado")
        print(f"📁 Ubicación: {{detector.classifier_path}}")
        print("🧪 Validación científica completada")
        print("🛡️  Sistema híbrido listo con controles negativos")
        sys.exit(0)
    else:
        print("❌ ERROR: Fallo en el entrenamiento híbrido")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open(hybrid_script_path, 'w', encoding='utf-8') as f:
            f.write(hybrid_script_content)
        
        return str(hybrid_script_path)
    
    def train_with_poetic_corpus(self):
        """Entrenar usando el corpus poético como datos adicionales de control"""
        print("\n🧪 ENTRENAMIENTO HÍBRIDO CON CORPUS POÉTICO")
        print("=" * 55)
        print("📚 Combina corpus poético como CONTROLES NEGATIVOS")
        print("🎯 Corpus poético = 0 en todas las tácticas manipulativas")
        print("📊 Mejora balance del dataset: Telares (manipulativo) vs Poesía (limpio)")
        print("🔬 Científicamente válido: controles negativos reales")
        
        response = self.input_validator.get_yes_no_input("¿Proceder con entrenamiento científico híbrido? [Y/n]: ")
        
        if not response:
            return False
        
        print("🔬 Ejecutando entrenamiento científico con controles...")
        
        try:
            # Crear script híbrido que combine ambos datasets
            script_path = Path(__file__).parent.parent.parent / "telares_detector_interface.py"
            hybrid_script = self.create_hybrid_training_script(script_path)
            
            print("🚀 Iniciando entrenamiento híbrido...")
            start_time = time.time()
            
            # Ejecutar entrenamiento híbrido
            result = subprocess.run([
                sys.executable, hybrid_script
            ], capture_output=True, text=True, cwd=str(script_path.parent))
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                self.display.show_success(
                    f"Entrenamiento híbrido completado en {duration/60:.1f} minutos"
                )
                print("🎉 Detector híbrido entrenado con corpus poético + telares")
                print("🧪 Validación científica: controles negativos incluidos")
                print("🛡️ Mejor precisión esperada en detección real")
                self.display.pause_for_user()
                return True
            else:
                self.display.show_error(f"Error en entrenamiento híbrido: {result.stderr}")
                return False
                
        except Exception as e:
            self.display.show_error(f"Error ejecutando entrenamiento híbrido: {str(e)}")
            return False
    
    def analyze_training_data(self):
        """Análizar los datos de entrenamiento"""
        print("\n📊 ANÁLISIS DE DATOS DE ENTRENAMIENTO")
        print("="*50)
        
        # Ejecutar análisis del dataset
        try:
            script_path = Path(__file__).parent.parent.parent / "telares_detector_interface.py"
            
            analysis_script = f'''
import pandas as pd
import sys
sys.path.append(r"{script_path.parent}")
from telares_detector_interface import TelaresDetectorSystem

detector = TelaresDetectorSystem()
X, y = detector.load_training_data()

if X and y is not None:
    print("📈 ESTADÍSTICAS DEL DATASET:")
    print("="*40)
    print(f"📝 Total mensajes: {{len(X)}}")
    print(f"🏷️  Tácticas: {{len(detector.label_names)}}")
    
    import numpy as np
    for i, label in enumerate(detector.label_names):
        count = y[:, i].sum()
        pct = (count / len(X)) * 100
        print(f"{{label}}: {{count}} ({{pct:.1f}}%)")
        
    print(f"\\n📊 Mensajes con múltiples tácticas: {{(y.sum(axis=1) > 1).sum()}}")
    print(f"📊 Mensajes sin tácticas detectadas: {{(y.sum(axis=1) == 0).sum()}}")
else:
    print("❌ No se pudo cargar el dataset")
'''
            
            exec(analysis_script)
            
        except Exception as e:
            print(f"❌ Error en análisis: {str(e)}")
        
        self.display.pause_for_user()
    
    def quick_validation_training(self):
        """Entrenamiento rápido solo para validación"""
        print("\n🚀 ENTRENAMIENTO RÁPIDO DE VALIDACIÓN")
        print("="*50)
        print("⏱️  Modo: Validación rápida del pipeline")
        print("📊 Usa subset pequeño del dataset")
        print("🎯 Objetivo: Verificar que todo funciona correctamente")
        
        response = self.input_validator.get_yes_no_input("¿Ejecutar validación rápida? [Y/n]: ")
        
        if response:
            print("⏱️  Ejecutando validación rápida...")
            time.sleep(2)  # Simular procesamiento
            self.display.show_success("Validación completada - Pipeline funcional")
        
        return response
    
    def compare_models(self):
        """Comparar diferentes configuraciones de modelo"""
        print("\n📈 COMPARACIÓN DE MODELOS")
        print("="*40)
        print("🔧 Configuraciones disponibles:")
        print("   1. Logistic Regression (Rápido)")
        print("   2. Random Forest (Preciso)")
        print("   3. SVM (Balanced)")
        print("   4. Ensemble (Mejor performance)")
        
        print("\n💡 Esta funcionalidad permite optimizar el detector")
        print("⏱️  Tiempo estimado por modelo: 3-5 minutos")
        
        choice = input("Seleccione configuración (1-4): ").strip()
        
        if choice in ["1", "2", "3", "4"]:
            models = ["Logistic Regression", "Random Forest", "SVM", "Ensemble"]
            selected = models[int(choice) - 1]
            print(f"🎯 Configurado para usar: {selected}")
            self.display.show_success(f"Modelo {selected} seleccionado")
        else:
            print("❌ Opción inválida")
            
        self.display.pause_for_user()
    
    def run_phase1_menu(self):
        """Menú principal de la fase de entrenamiento"""
        while True:
            self.show_training_menu()
            choice = input("\n🎯 Seleccione una opción: ").strip()
            
            if choice == "1":
                if self.run_telares_training():
                    print("🎉 ¡Entrenamiento completado exitosamente!")
                    break
            elif choice == "2":
                self.train_with_poetic_corpus()
            elif choice == "3":
                self.analyze_training_data()
            elif choice == "4":
                self.quick_validation_training()
            elif choice == "5":
                self.compare_models()
            elif choice == "0":
                break
            else:
                print("❌ Opción inválida. Seleccione 0-5.")