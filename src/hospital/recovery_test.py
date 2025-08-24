#!/usr/bin/env python3
"""
🧪 TEST DE RECUPERACIÓN POST-CIRUGÍA
Creado por Bernard Orozco bajo tutela de Aslan

Prueba que el modelo operado puede entrenar correctamente después de la cirugía.
¡Los gates resucitados deben demostrar que pueden aprender!
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure environment
conda_prefix = os.getenv('CONDA_PREFIX', '/usr/local')
if conda_prefix != '/usr/local':
    os.environ['CUDA_HOME'] = conda_prefix
    os.environ['LD_LIBRARY_PATH'] = f'{conda_prefix}/lib:{conda_prefix}/lib64:{os.environ.get("LD_LIBRARY_PATH", "")}'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from model import VariationalDropout, DropConnect
    from data_processor import TextProcessor
    print("✅ Módulos de recuperación cargados")
except Exception as e:
    print(f"❌ Error importando: {e}")
    sys.exit(1)


def test_model_recovery(operated_model_path: str, text_file: str = "alice_wonderland.txt"):
    """
    🧪 Test de recuperación completo del modelo operado
    
    Args:
        operated_model_path: Modelo después de cirugía
        text_file: Texto para entrenamiento de prueba
    """
    print("🧪 INICIANDO TEST DE RECUPERACIÓN POST-CIRUGÍA")
    print("=" * 50)
    
    try:
        # Cargar modelo operado
        print(f"🏥 Cargando paciente operado...")
        model = tf.keras.models.load_model(
            operated_model_path,
            custom_objects={
                'VariationalDropout': VariationalDropout,
                'DropConnect': DropConnect
            }
        )
        print(f"✅ Modelo operado cargado: {Path(operated_model_path).name}")
        
        # Preparar datos de prueba pequeños
        print("📊 Preparando datos de prueba...")
        
        if not Path(text_file).exists():
            print(f"⚠️ Archivo {text_file} no encontrado, creando datos sintéticos...")
            # Datos sintéticos para test básico
            vocab_size = 1000
            sequence_length = 64
            batch_size = 32
            
            X_test = np.random.randint(0, vocab_size, size=(batch_size, sequence_length))
            y_test = np.random.randint(0, vocab_size, size=(batch_size,))
        else:
            # Usar datos multi-corpus pero limitados para test rápido
            try:
                processor = TextProcessor(sequence_length=64, step_size=5)
                X_onehot, y_onehot = processor.prepare_data("corpus", max_length=8_000)  # Corpus limitado para test
                
                # Convert to integer sequences
                X_test = np.argmax(X_onehot, axis=-1)[:100]  # Solo 100 secuencias
                y_test = np.argmax(y_onehot, axis=-1)[:100]
                
                vocab_size = processor.vocab_size
                sequence_length = X_test.shape[1]
                batch_size = min(32, X_test.shape[0])
                print(f"✅ Multi-corpus de test cargado: {X_test.shape}")
                
            except Exception as e:
                print(f"⚠️ Error cargando corpus, usando datos sintéticos: {e}")
                # Datos sintéticos como fallback
                vocab_size = 1000
                sequence_length = 64
                batch_size = 32
                X_test = np.random.randint(0, vocab_size, size=(batch_size, sequence_length))
                y_test = np.random.randint(0, vocab_size, size=(batch_size,))
        
        print(f"   Secuencias de prueba: {X_test.shape[0]}")
        print(f"   Vocabulary size: {vocab_size}")
        print(f"   Sequence length: {sequence_length}")
        
        # Test 1: Evaluar loss inicial
        print("\n🔍 TEST 1: Loss inicial post-cirugía")
        print("-" * 30)
        
        initial_loss = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)
        print(f"📊 Loss inicial: {initial_loss:.4f}")
        print(f"🌊 Perplexity inicial: {np.exp(initial_loss):.1f}")
        
        if initial_loss > 10:
            print("🔴 WARNING: Loss muy alto - modelo aún crítico")
        elif initial_loss > 6:
            print("🟡 CAUTELA: Loss alto pero mejorable")
        else:
            print("✅ BUENO: Loss en rango razonable")
        
        # Test 2: 1 época de entrenamiento
        print("\n🏃 TEST 2: Entrenamiento de recuperación (1 época)")
        print("-" * 40)
        
        # Optimizer ultra-conservador post-cirugía
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=5e-5,  # MUY conservador
            clipnorm=0.5         # Clipping agresivo
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("⚡ Iniciando entrenamiento de prueba...")
        history = model.fit(
            X_test, y_test,
            epochs=1,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.1 if X_test.shape[0] > 20 else 0
        )
        
        # Analizar resultados
        final_loss = history.history['loss'][0]
        final_acc = history.history['accuracy'][0]
        
        print(f"\n📊 RESULTADOS DEL TEST:")
        print(f"   Loss final: {final_loss:.4f}")
        print(f"   Accuracy final: {final_acc:.4f}")
        print(f"   Perplexity final: {np.exp(final_loss):.1f}")
        
        # Calcular mejora
        loss_improvement = initial_loss - final_loss
        print(f"   📈 Mejora en loss: {loss_improvement:.4f}")
        
        # Veredicto de recuperación
        print(f"\n🩺 VEREDICTO DE RECUPERACIÓN:")
        
        if final_loss < 4.0:
            print("🎉 RECUPERACIÓN EXCELENTE - Modelo completamente funcional")
            recovery_status = "excellent"
        elif final_loss < 6.0:
            print("✅ RECUPERACIÓN BUENA - Modelo funcional")
            recovery_status = "good"
        elif loss_improvement > 0.5:
            print("🟡 RECUPERACIÓN PARCIAL - Modelo mejorando")
            recovery_status = "partial"
        elif loss_improvement > 0.1:
            print("🟡 RECUPERACIÓN LENTA - Modelo respondiendo")
            recovery_status = "slow"
        else:
            print("🔴 SIN RECUPERACIÓN - Modelo aún crítico")
            recovery_status = "failed"
        
        # Test 3: Verificar que los gates siguen funcionales
        print(f"\n🔍 TEST 3: Verificación de gates post-entrenamiento")
        print("-" * 45)
        
        gates_still_functional = True
        
        for layer in model.layers:
            if 'lstm' in layer.name.lower():
                weights = layer.get_weights()
                if len(weights) >= 3:
                    bias = weights[2]
                    units = layer.units
                    
                    input_mean = np.mean(bias[:units])
                    forget_mean = np.mean(bias[units:units*2])
                    output_mean = np.mean(bias[units*3:units*4])
                    
                    print(f"   {layer.name}:")
                    print(f"     Input: {input_mean:.3f}")
                    print(f"     Forget: {forget_mean:.3f}")
                    print(f"     Output: {output_mean:.3f}")
                    
                    # Verificar que no se volvieron a saturar
                    if abs(input_mean) < 0.01 and abs(output_mean) < 0.01:
                        print(f"     ⚠️ Gates aún muy cerrados")
                        gates_still_functional = False
                    elif abs(input_mean) > 0.1 or abs(output_mean) > 0.1:
                        print(f"     ✅ Gates activándose normalmente")
                    else:
                        print(f"     🟡 Gates en transición")
        
        # Reporte final
        recovery_report = {
            'model_tested': operated_model_path,
            'initial_loss': float(initial_loss),
            'final_loss': float(final_loss),
            'final_accuracy': float(final_acc),
            'loss_improvement': float(loss_improvement),
            'recovery_status': recovery_status,
            'gates_functional': gates_still_functional,
            'test_successful': recovery_status in ['excellent', 'good', 'partial']
        }
        
        print(f"\n🎯 REPORTE FINAL DE RECUPERACIÓN:")
        print("=" * 35)
        print(f"   Estado de recuperación: {recovery_status.upper()}")
        print(f"   Loss mejorado: {loss_improvement > 0.1}")
        print(f"   Gates funcionales: {gates_still_functional}")
        print(f"   Test exitoso: {recovery_report['test_successful']}")
        
        if recovery_report['test_successful']:
            print("\n🎉 ¡CIRUGÍA Y RECUPERACIÓN EXITOSAS!")
            print("🦁 El paciente ha vuelto a la vida y puede entrenar")
        else:
            print("\n⚠️ RECUPERACIÓN INCOMPLETA")
            print("🏥 Requiere cuidados intensivos adicionales")
        
        return recovery_report
        
    except Exception as e:
        print(f"❌ Error en test de recuperación: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="🧪 Test de Recuperación Post-Cirugía")
    parser.add_argument('--model', required=True, help='Modelo operado a probar')
    parser.add_argument('--text', default='alice_wonderland.txt', help='Archivo de texto para prueba')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Modelo no encontrado: {args.model}")
        sys.exit(1)
    
    print("🧪 INICIANDO TEST DE RECUPERACIÓN...")
    report = test_model_recovery(args.model, args.text)
    
    if report and report['test_successful']:
        print("\n🎉 TEST DE RECUPERACIÓN EXITOSO")
    else:
        print("\n❌ TEST DE RECUPERACIÓN FALLÓ")
        sys.exit(1)