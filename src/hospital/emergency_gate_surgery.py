#!/usr/bin/env python3
"""
🚨 CIRUGÍA DE EMERGENCIA PARA GATES SATURADOS
Creado por Bernard Orozco bajo tutela de Aslan (corrigiendo sus errores)

SITUACIÓN CRÍTICA DETECTADA:
- Loss: 8.57 (CATASTRÓFICO - debería ser <3)
- Input Gate: 0.005 (TOTALMENTE CERRADO - no aprende)
- Output Gate: 0.004 (TOTALMENTE CERRADO - no produce output útil)  
- Forget Gate: 1.006 (funcional pero inútil sin input/output)

SOLUCIÓN: Reset quirúrgico de SOLO los bias de gates
SIN perder embeddings y pesos LSTM principales
"""

import os
import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

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
    
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Import custom classes
    from model import VariationalDropout, DropConnect
    print("✅ Módulos quirúrgicos cargados")
    
except Exception as e:
    print(f"❌ Error importando: {e}")
    sys.exit(1)


class LSTMGateSurgeon:
    """
    🔬 CIRUJANO ESPECIALISTA EN GATES LSTM SATURADOS
    
    Realiza intervención quirúrgica para rescatar modelos con gates muertos:
    - Input gates ~0.005 (cerrados)
    - Output gates ~0.004 (cerrados) 
    - Mantiene pesos principales intactos
    """
    
    def __init__(self, model_path: str):
        """
        Inicializar cirujano con modelo crítico
        
        Args:
            model_path: Ruta al modelo moribundo
        """
        self.model_path = Path(model_path)
        self.surgery_log = []
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Paciente no encontrado: {model_path}")
        
        print("🚨 CIRUJANO DE GATES LSTM PREPARADO")
        print("=" * 50)
        
    def diagnose_patient(self, model):
        """
        🔍 DIAGNÓSTICO PRE-QUIRÚRGICO
        
        Evalúa estado crítico del paciente antes de cirugía
        """
        print("🔍 DIAGNÓSTICO PRE-QUIRÚRGICO:")
        print("-" * 30)
        
        critical_issues = []
        gate_stats = {}
        
        for layer in model.layers:
            if 'lstm' in layer.name.lower():
                print(f"\n🧠 Examinando {layer.name}:")
                
                # Extraer pesos LSTM
                weights = layer.get_weights()
                if len(weights) >= 3:
                    kernel, recurrent, bias = weights[0], weights[1], weights[2]
                    units = layer.units
                    
                    # Analizar bias de cada gate
                    bias_input = bias[:units]                    # Input gate
                    bias_forget = bias[units:units*2]            # Forget gate  
                    bias_cell = bias[units*2:units*3]           # Cell gate
                    bias_output = bias[units*3:units*4]         # Output gate
                    
                    # Estadísticas por gate
                    stats = {
                        'input_gate_mean': float(np.mean(bias_input)),
                        'forget_gate_mean': float(np.mean(bias_forget)),
                        'cell_gate_mean': float(np.mean(bias_cell)),
                        'output_gate_mean': float(np.mean(bias_output))
                    }
                    
                    gate_stats[layer.name] = stats
                    
                    # DIAGNÓSTICO CRÍTICO
                    print(f"   🚪 Input Gate: μ={stats['input_gate_mean']:.3f}")
                    if abs(stats['input_gate_mean']) < 0.1:
                        print("     🔴 CRÍTICO: Input gate CERRADO - no aprende información nueva")
                        critical_issues.append(f"{layer.name}: Input gate saturado")
                    
                    print(f"   🧠 Forget Gate: μ={stats['forget_gate_mean']:.3f}")
                    if stats['forget_gate_mean'] < 0.5 or stats['forget_gate_mean'] > 1.5:
                        print("     🟡 WARNING: Forget gate subóptimo")
                        critical_issues.append(f"{layer.name}: Forget gate subóptimo")
                    else:
                        print("     ✅ Forget gate funcional")
                    
                    print(f"   📤 Output Gate: μ={stats['output_gate_mean']:.3f}")
                    if abs(stats['output_gate_mean']) < 0.1:
                        print("     🔴 CRÍTICO: Output gate CERRADO - no produce salida útil")
                        critical_issues.append(f"{layer.name}: Output gate saturado")
                    
                    print(f"   🔄 Cell Gate: μ={stats['cell_gate_mean']:.3f}")
        
        # Veredicto pre-quirúrgico
        print(f"\n🩺 VEREDICTO PRE-QUIRÚRGICO:")
        print(f"   Problemas críticos detectados: {len(critical_issues)}")
        
        for issue in critical_issues:
            print(f"   🔴 {issue}")
        
        if len(critical_issues) >= 2:
            print("   💀 PACIENTE EN ESTADO CRÍTICO - Cirugía URGENTE requerida")
            surgery_needed = True
        elif len(critical_issues) == 1:
            print("   🟡 PACIENTE GRAVE - Cirugía recomendada")
            surgery_needed = True
        else:
            print("   ✅ PACIENTE ESTABLE - Cirugía no necesaria")
            surgery_needed = False
        
        self.surgery_log.append({
            'timestamp': datetime.now().isoformat(),
            'stage': 'pre_surgery_diagnosis',
            'gate_stats': gate_stats,
            'critical_issues': critical_issues,
            'surgery_needed': surgery_needed
        })
        
        return surgery_needed, critical_issues, gate_stats
    
    def perform_gate_surgery(self, model):
        """
        ⚡ CIRUGÍA PRINCIPAL - RESET DE GATES SATURADOS
        
        Reinicializa SOLO los bias de gates problemáticos
        Mantiene intactos: embeddings, kernels LSTM, recurrent weights
        """
        print("\n⚡ INICIANDO CIRUGÍA DE GATES:")
        print("=" * 40)
        
        surgery_actions = []
        
        for layer in model.layers:
            if 'lstm' in layer.name.lower():
                print(f"\n🔧 Operando {layer.name}...")
                
                # Obtener pesos actuales
                weights = layer.get_weights()
                if len(weights) >= 3:
                    kernel, recurrent, bias = weights[0], weights[1], weights[2]
                    units = layer.units
                    
                    # BACKUP de pesos originales
                    original_bias = bias.copy()
                    
                    # CIRUGÍA: Reinicializar solo bias de gates
                    new_bias = bias.copy()
                    
                    # Input gate bias = 0.0 (estándar Keras)
                    new_bias[:units] = 0.0
                    print(f"   ✂️ Input gate bias: resetear a 0.0")
                    
                    # Forget gate bias = 1.0 (estándar para LSTM - Jozefowicz et al.)
                    new_bias[units:units*2] = 1.0  
                    print(f"   ✂️ Forget gate bias: resetear a 1.0")
                    
                    # Cell gate bias = 0.0 (estándar)
                    new_bias[units*2:units*3] = 0.0
                    print(f"   ✂️ Cell gate bias: resetear a 0.0")
                    
                    # Output gate bias = 0.0 (estándar)
                    new_bias[units*3:units*4] = 0.0
                    print(f"   ✂️ Output gate bias: resetear a 0.0")
                    
                    # APLICAR nuevos pesos (solo bias cambia)
                    layer.set_weights([kernel, recurrent, new_bias])
                    
                    # Log de la operación
                    action = {
                        'layer_name': layer.name,
                        'original_bias_stats': {
                            'input_mean': float(np.mean(original_bias[:units])),
                            'forget_mean': float(np.mean(original_bias[units:units*2])),
                            'cell_mean': float(np.mean(original_bias[units*2:units*3])),
                            'output_mean': float(np.mean(original_bias[units*3:units*4]))
                        },
                        'new_bias_stats': {
                            'input_mean': 0.0,
                            'forget_mean': 1.0, 
                            'cell_mean': 0.0,
                            'output_mean': 0.0
                        },
                        'surgery_successful': True
                    }
                    
                    surgery_actions.append(action)
                    print(f"   ✅ Cirugía en {layer.name} EXITOSA")
        
        self.surgery_log.append({
            'timestamp': datetime.now().isoformat(),
            'stage': 'gate_surgery',
            'actions': surgery_actions
        })
        
        print(f"\n🎉 CIRUGÍA COMPLETADA - {len(surgery_actions)} capas operadas")
        return surgery_actions
    
    def post_surgery_verification(self, model):
        """
        🔍 VERIFICACIÓN POST-QUIRÚRGICA
        
        Confirma que los gates fueron resetados correctamente
        """
        print("\n🔍 VERIFICACIÓN POST-QUIRÚRGICA:")
        print("-" * 35)
        
        verification_passed = True
        
        for layer in model.layers:
            if 'lstm' in layer.name.lower():
                weights = layer.get_weights()
                if len(weights) >= 3:
                    bias = weights[2]
                    units = layer.units
                    
                    # Verificar cada gate
                    input_mean = np.mean(bias[:units])
                    forget_mean = np.mean(bias[units:units*2])
                    cell_mean = np.mean(bias[units*2:units*3])
                    output_mean = np.mean(bias[units*3:units*4])
                    
                    print(f"\n✅ {layer.name} - Post-cirugía:")
                    print(f"   🚪 Input gate: {input_mean:.3f} (target: 0.0)")
                    print(f"   🧠 Forget gate: {forget_mean:.3f} (target: 1.0)")  
                    print(f"   🔄 Cell gate: {cell_mean:.3f} (target: 0.0)")
                    print(f"   📤 Output gate: {output_mean:.3f} (target: 0.0)")
                    
                    # Verificación de rangos
                    if abs(input_mean) > 0.1:
                        print(f"   ❌ Input gate no resetado correctamente")
                        verification_passed = False
                    if abs(forget_mean - 1.0) > 0.1:
                        print(f"   ❌ Forget gate no resetado correctamente")
                        verification_passed = False
                    if abs(output_mean) > 0.1:
                        print(f"   ❌ Output gate no resetado correctamente") 
                        verification_passed = False
        
        if verification_passed:
            print(f"\n✅ VERIFICACIÓN EXITOSA - Todos los gates resetados")
        else:
            print(f"\n❌ VERIFICACIÓN FALLIDA - Algunos gates no resetados")
        
        self.surgery_log.append({
            'timestamp': datetime.now().isoformat(),
            'stage': 'post_surgery_verification',
            'verification_passed': verification_passed
        })
        
        return verification_passed
    
    def test_epoch_recovery(self, model, X_sample, y_sample):
        """
        🧪 TEST DE RECUPERACIÓN - 1 ÉPOCA DE PRUEBA
        
        Verifica que el modelo puede entrenar después de la cirugía
        """
        print("\n🧪 TEST DE RECUPERACIÓN:")
        print("-" * 25)
        
        # Compilar con optimizer conservador
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,  # Muy conservador post-cirugía
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Evaluar loss inicial (post-cirugía)
        initial_loss = model.evaluate(X_sample, y_sample, verbose=0)
        print(f"🔍 Loss inicial post-cirugía: {initial_loss:.4f}")
        
        # 1 época de prueba
        print("🏃 Ejecutando 1 época de prueba...")
        try:
            history = model.fit(
                X_sample, y_sample,
                epochs=1,
                batch_size=32,
                verbose=1,
                validation_split=0.1
            )
            
            final_loss = history.history['loss'][0]
            final_acc = history.history['accuracy'][0]
            
            print(f"📊 Loss final: {final_loss:.4f}")
            print(f"🎯 Accuracy final: {final_acc:.4f}")
            
            # Análisis de recuperación
            loss_improvement = initial_loss - final_loss
            print(f"📈 Mejora en loss: {loss_improvement:.4f}")
            
            if final_loss < 7.0:
                print("🎉 RECUPERACIÓN EXITOSA - Loss bajando")
                recovery_success = True
            elif loss_improvement > 0.1:
                print("✅ RECUPERACIÓN PARCIAL - Alguna mejora")
                recovery_success = True
            else:
                print("⚠️ RECUPERACIÓN INCIERTA - Monitorear más épocas")
                recovery_success = False
            
            self.surgery_log.append({
                'timestamp': datetime.now().isoformat(),
                'stage': 'recovery_test',
                'initial_loss': float(initial_loss),
                'final_loss': float(final_loss),
                'final_accuracy': float(final_acc),
                'loss_improvement': float(loss_improvement),
                'recovery_success': recovery_success
            })
            
            return recovery_success, final_loss
            
        except Exception as e:
            print(f"❌ Error en test de recuperación: {e}")
            return False, float('inf')
    
    def full_emergency_surgery(self, sample_data: tuple = None):
        """
        🚨 PROTOCOLO COMPLETO DE CIRUGÍA DE EMERGENCIA
        
        Ejecuta todo el protocolo:
        1. Diagnóstico pre-quirúrgico
        2. Cirugía de gates
        3. Verificación post-cirugía
        4. Test de recuperación
        """
        print("🚨" + "="*48 + "🚨")
        print("  PROTOCOLO DE CIRUGÍA DE EMERGENCIA LSTM")
        print("  Creado por Bernard Orozco - Aslan Surgery Protocol")
        print("🚨" + "="*48 + "🚨")
        
        surgery_start = datetime.now()
        
        try:
            # Cargar paciente
            print("🏥 CARGANDO PACIENTE...")
            model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    'VariationalDropout': VariationalDropout,
                    'DropConnect': DropConnect
                }
            )
            print(f"✅ Paciente cargado: {self.model_path.name}")
            
            # FASE 1: Diagnóstico
            surgery_needed, issues, gate_stats = self.diagnose_patient(model)
            
            if not surgery_needed:
                print("✅ PACIENTE ESTABLE - No requiere cirugía")
                return model, self.surgery_log
            
            # FASE 2: Cirugía
            surgery_actions = self.perform_gate_surgery(model)
            
            # FASE 3: Verificación
            verification_passed = self.post_surgery_verification(model)
            
            if not verification_passed:
                print("❌ CIRUGÍA FALLÓ - Paciente no mejoró")
                return None, self.surgery_log
            
            # FASE 4: Test de recuperación (si hay datos de muestra)
            recovery_success = False
            final_loss = float('inf')
            
            if sample_data:
                X_sample, y_sample = sample_data
                recovery_success, final_loss = self.test_epoch_recovery(model, X_sample, y_sample)
            else:
                print("⚠️ No hay datos de muestra - Saltando test de recuperación")
            
            # Guardar modelo operado
            surgery_end = datetime.now()
            surgery_duration = surgery_end - surgery_start
            
            operated_model_path = self.model_path.parent / f"operated_{self.model_path.name}"
            model.save(operated_model_path)
            
            # Reporte final
            surgery_report = {
                'original_model': str(self.model_path),
                'operated_model': str(operated_model_path),
                'surgery_start': surgery_start.isoformat(),
                'surgery_end': surgery_end.isoformat(),
                'duration_minutes': surgery_duration.total_seconds() / 60,
                'issues_detected': len(issues),
                'surgery_actions': len(surgery_actions),
                'verification_passed': verification_passed,
                'recovery_success': recovery_success,
                'final_loss_after_test': final_loss,
                'full_log': self.surgery_log
            }
            
            # Guardar reporte
            report_path = self.model_path.parent / f"surgery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(surgery_report, f, indent=2, default=str)
            
            print(f"\n🏥 CIRUGÍA COMPLETADA:")
            print("=" * 30)
            print(f"   Duración: {surgery_duration.total_seconds()/60:.1f} minutos")
            print(f"   Problemas tratados: {len(issues)}")
            print(f"   Acciones realizadas: {len(surgery_actions)}")
            print(f"   Verificación: {'✅ EXITOSA' if verification_passed else '❌ FALLIDA'}")
            
            if sample_data:
                print(f"   Test recuperación: {'✅ EXITOSO' if recovery_success else '❌ FALLIDO'}")
                print(f"   Loss post-test: {final_loss:.4f}")
            
            print(f"   Modelo operado: {operated_model_path.name}")
            print(f"   Reporte: {report_path.name}")
            
            if verification_passed and (recovery_success or not sample_data):
                print("\n🎉 CIRUGÍA EXITOSA - Paciente estabilizado")
            else:
                print("\n⚠️ CIRUGÍA PARCIAL - Monitoreo continuo requerido")
            
            return model, surgery_report
            
        except Exception as e:
            print(f"❌ EMERGENCIA EN QUIRÓFANO: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def quick_surgery(model_path: str, sample_data: tuple = None):
    """
    ⚡ CIRUGÍA RÁPIDA - ENTRADA DIRECTA
    
    Args:
        model_path: Ruta al modelo crítico
        sample_data: (X, y) datos de muestra para test (opcional)
    """
    surgeon = LSTMGateSurgeon(model_path)
    return surgeon.full_emergency_surgery(sample_data)


if __name__ == "__main__":
    """
    🚨 MODO EMERGENCIA - Cirugía inmediata desde línea de comandos
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="🚨 Cirugía de Emergencia para Gates LSTM")
    parser.add_argument('--model', required=True, help='Modelo con gates saturados')
    parser.add_argument('--test-data', help='Datos para test de recuperación (opcional)')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Paciente no encontrado: {args.model}")
        sys.exit(1)
    
    print("🚨 INICIANDO CIRUGÍA DE EMERGENCIA...")
    
    # Preparar datos de test si se proporcionan
    sample_data = None
    if args.test_data and Path(args.test_data).exists():
        # Aquí cargarías tus datos específicos
        print(f"📊 Datos de test: {args.test_data}")
        # sample_data = load_your_test_data(args.test_data)
    
    # Ejecutar cirugía
    operated_model, report = quick_surgery(args.model, sample_data)
    
    if operated_model and report:
        print("\n🎉 CIRUGÍA DE EMERGENCIA COMPLETADA")
        print("🏥 Paciente transferido a cuidados post-operatorios")
    else:
        print("\n💀 CIRUGÍA FALLÓ - Paciente no recuperable")
        sys.exit(1)