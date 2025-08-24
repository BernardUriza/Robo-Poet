#!/usr/bin/env python3
"""
🔬 GRADIENT ANALYZER LITE - MÓDULO 2 TASK 1.1 (Sin pandas)
Creado por Bernard Orozco bajo tutela de Aslan

Versión simplificada del analizador que funciona solo con numpy y matplotlib
para el análisis fundamental de gradientes.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import csv

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
    from data_processor import TextProcessor
    print("✅ Módulos básicos cargados")
    
except Exception as e:
    print(f"❌ Error importando: {e}")
    sys.exit(1)


class GradientAnalyzerLite:
    """
    🔬 ANALIZADOR LIGERO DE GRADIENTES
    
    Implementa análisis esencial sin dependencias pesadas
    """
    
    def __init__(self, model_path: str, text_file: str = "alice_wonderland.txt"):
        self.model_path = Path(model_path)
        self.text_file = Path(text_file)
        self.gradient_records = []
        
        print("🔬 GRADIENT ANALYZER LITE INICIALIZADO")
        print("=" * 45)
    
    def load_model_and_data(self) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
        """Cargar modelo operado y datos"""
        print("🏥 Cargando modelo operado...")
        
        model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                'VariationalDropout': VariationalDropout,
                'DropConnect': DropConnect
            }
        )
        print(f"✅ Modelo cargado: {len(model.layers)} capas")
        
        # Preparar datos pequeños para análisis usando multi-corpus
        try:
            processor = TextProcessor(sequence_length=128, step_size=3)
            X_onehot, y_onehot = processor.prepare_data("corpus", max_length=15_000)  # Corpus pequeño para análisis
            
            X_train = np.argmax(X_onehot, axis=-1)
            y_train = np.argmax(y_onehot, axis=-1)
            print(f"✅ Datos multi-corpus cargados: {X_train.shape}")
            
        except Exception as e:
            print(f"⚠️ No se pudo cargar corpus, usando datos sintéticos: {e}")
            # Datos sintéticos como fallback
            X_train = np.random.randint(0, 1000, size=(100, 128))
            y_train = np.random.randint(0, 1000, size=(100,))
        
        print(f"   Datos preparados: {X_train.shape[0]} secuencias")
        return model, X_train, y_train
    
    def track_gradients_lite(self, model: tf.keras.Model, X_train: np.ndarray, 
                           y_train: np.ndarray, num_batches: int = 30) -> List[Dict]:
        """
        🎯 RASTREO BÁSICO DE GRADIENTES
        
        Versión simplificada del análisis de gradientes
        """
        print(f"\n🔍 Rastreando gradientes por {num_batches} batches...")
        
        # Compilar modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        
        # Preparar datasets
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(16).take(num_batches)
        
        records = []
        layer_names = []
        
        print("📈 Iniciando rastreo...")
        for batch_idx, (x_batch, y_batch) in enumerate(dataset):
            
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = model.compiled_loss(y_batch, predictions)
            
            gradients = tape.gradient(loss, model.trainable_weights)
            
            # Registro básico
            record = {
                'batch': batch_idx,
                'loss': float(loss),
                'gradient_norms': []
            }
            
            # Extraer gradient norms por capa
            for i, (layer, grad) in enumerate(zip(model.layers, gradients)):
                if grad is not None and len(grad.shape) > 0:
                    grad_norm = tf.norm(grad).numpy()
                    record['gradient_norms'].append({
                        'layer_name': layer.name,
                        'layer_idx': i,
                        'norm': float(grad_norm)
                    })
                    
                    if batch_idx == 0:  # Primera vez, guardar nombres
                        layer_names.append(layer.name)
            
            records.append(record)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"   Batch {batch_idx + 1:2d}/{num_batches}: Loss={loss:.4f}")
            
            # Aplicar gradientes
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        self.gradient_records = records
        self.layer_names = layer_names
        
        # Guardar CSV básico
        self.save_gradient_csv(records)
        
        print(f"📊 Rastreo completado: {len(records)} batches analizados")
        return records
    
    def save_gradient_csv(self, records: List[Dict]) -> str:
        """Guardar datos en CSV básico"""
        csv_path = f"gradient_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['batch', 'loss']
            if records and records[0]['gradient_norms']:
                for norm_info in records[0]['gradient_norms']:
                    header.append(f"{norm_info['layer_name']}_norm")
            writer.writerow(header)
            
            # Data
            for record in records:
                row = [record['batch'], record['loss']]
                for norm_info in record['gradient_norms']:
                    row.append(norm_info['norm'])
                writer.writerow(row)
        
        print(f"📄 CSV guardado: {csv_path}")
        return csv_path
    
    def detect_collapse_points(self, records: List[Dict]) -> Dict:
        """Detectar puntos de colapso de gradientes"""
        print("\n🔍 DETECTANDO PUNTOS DE COLAPSO...")
        
        collapse_info = {'layers_with_collapse': [], 'earliest_collapse': -1}
        
        # Organizar datos por capa
        layer_data = {}
        for record in records:
            for norm_info in record['gradient_norms']:
                layer_name = norm_info['layer_name']
                if layer_name not in layer_data:
                    layer_data[layer_name] = []
                layer_data[layer_name].append({
                    'batch': record['batch'],
                    'norm': norm_info['norm']
                })
        
        # Analizar cada capa
        earliest_collapse_batch = float('inf')
        
        for layer_name, data in layer_data.items():
            norms = [d['norm'] for d in data]
            batches = [d['batch'] for d in data]
            
            # Detectar colapso (norm < 1e-5)
            collapse_batches = [b for b, n in zip(batches, norms) if n < 1e-5]
            
            if collapse_batches:
                first_collapse = min(collapse_batches)
                collapse_info['layers_with_collapse'].append({
                    'layer_name': layer_name,
                    'collapse_batch': first_collapse,
                    'norm_at_collapse': min(norms)
                })
                
                earliest_collapse_batch = min(earliest_collapse_batch, first_collapse)
                print(f"🔴 {layer_name}: Colapso en batch {first_collapse}")
            else:
                # Check for declining trend
                if len(norms) >= 5:
                    trend = np.polyfit(range(len(norms)), norms, 1)[0]
                    if trend < -1e-6:
                        print(f"🟡 {layer_name}: Tendencia negativa (slope: {trend:.2e})")
                    else:
                        print(f"✅ {layer_name}: Estable")
        
        if earliest_collapse_batch != float('inf'):
            collapse_info['earliest_collapse'] = int(earliest_collapse_batch)
            print(f"\n💀 PRIMER COLAPSO GLOBAL: Batch {earliest_collapse_batch}")
        else:
            print(f"\n✅ No se detectó colapso crítico")
        
        return collapse_info
    
    def calculate_pascanu_ratios(self, records: List[Dict]) -> List[Dict]:
        """
        🎯 ANÁLISIS PASCANU BÁSICO
        
        Calcula ratios entre capas consecutivas
        """
        print("\n🔍 CALCULANDO RATIOS PASCANU...")
        
        ratio_records = []
        
        for record in records:
            batch_ratios = {'batch': record['batch'], 'ratios': []}
            
            norms = record['gradient_norms']
            
            # Calcular ratios entre capas consecutivas
            for i in range(len(norms) - 1):
                current_norm = norms[i]['norm']
                next_norm = norms[i + 1]['norm']
                
                if next_norm > 1e-10:
                    ratio = current_norm / next_norm
                else:
                    ratio = float('inf') if current_norm > 1e-10 else 1.0
                
                # Clasificación Pascanu
                if ratio < 0.1:
                    status = 'vanishing'
                elif ratio > 10.0:
                    status = 'exploding'
                else:
                    status = 'stable'
                
                batch_ratios['ratios'].append({
                    'from_layer': norms[i]['layer_name'],
                    'to_layer': norms[i + 1]['layer_name'], 
                    'ratio': float(ratio) if np.isfinite(ratio) else 999.0,
                    'status': status
                })
            
            ratio_records.append(batch_ratios)
        
        # Análisis de resultados
        if ratio_records:
            # Contar problemas por tipo
            vanishing_count = 0
            exploding_count = 0
            stable_count = 0
            
            for record in ratio_records:
                for ratio_info in record['ratios']:
                    if ratio_info['status'] == 'vanishing':
                        vanishing_count += 1
                    elif ratio_info['status'] == 'exploding':
                        exploding_count += 1
                    else:
                        stable_count += 1
            
            total = vanishing_count + exploding_count + stable_count
            print(f"📊 RESULTADOS PASCANU:")
            print(f"   Vanishing: {vanishing_count}/{total} ({vanishing_count/total*100:.1f}%)")
            print(f"   Exploding: {exploding_count}/{total} ({exploding_count/total*100:.1f}%)")
            print(f"   Stable: {stable_count}/{total} ({stable_count/total*100:.1f}%)")
        
        return ratio_records
    
    def create_basic_visualization(self, records: List[Dict]) -> str:
        """Crear visualización básica con matplotlib"""
        print("\n🎨 Creando visualización básica...")
        
        # Extraer datos para plotting
        batches = [r['batch'] for r in records]
        losses = [r['loss'] for r in records]
        
        # Organizar gradient norms por capa
        layer_norms = {}
        for record in records:
            for norm_info in record['gradient_norms']:
                layer_name = norm_info['layer_name']
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(norm_info['norm'])
        
        # Crear gráfico
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Análisis de Gradient Flow - Modelo Operado', fontsize=14)
        
        # Plot 1: Loss evolution
        axes[0, 0].plot(batches, losses, 'r-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Evolución del Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Gradient norms (log scale)
        axes[0, 1].set_yscale('log')
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_norms)))
        
        for i, (layer_name, norms) in enumerate(layer_norms.items()):
            axes[0, 1].plot(batches, norms, color=colors[i], 
                          label=layer_name[:15], marker='o', markersize=3)
        
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Gradient Norm (log scale)')
        axes[0, 1].set_title('Gradient Norms por Capa')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Problematic regions
        axes[1, 0].set_yscale('log')
        for i, (layer_name, norms) in enumerate(layer_norms.items()):
            # Marcar vanishing gradients
            vanishing_mask = np.array(norms) < 1e-5
            if np.any(vanishing_mask):
                vanishing_batches = np.array(batches)[vanishing_mask]
                vanishing_norms = np.array(norms)[vanishing_mask]
                axes[1, 0].scatter(vanishing_batches, vanishing_norms, 
                                 color='red', alpha=0.7, s=30, marker='x')
            
            # Plot normal points
            normal_mask = ~vanishing_mask
            if np.any(normal_mask):
                normal_batches = np.array(batches)[normal_mask]
                normal_norms = np.array(norms)[normal_mask]
                axes[1, 0].scatter(normal_batches, normal_norms, 
                                 color=colors[i], alpha=0.5, s=20)
        
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Gradient Norm (log scale)')
        axes[1, 0].set_title('Regiones Problemáticas (X = Vanishing)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss improvement
        if len(losses) > 1:
            loss_changes = np.diff(losses)
            axes[1, 1].plot(batches[1:], loss_changes, 'g-', linewidth=2, marker='s', markersize=4)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Loss Change')
            axes[1, 1].set_title('Cambio en Loss (Mejora = Negativo)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        plot_path = f"gradient_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráfico guardado: {plot_path}")
        return plot_path
    
    def run_complete_analysis(self, num_batches: int = 30) -> Dict:
        """Ejecutar análisis completo lite"""
        print("🔬" + "="*40 + "🔬")
        print("  GRADIENT ANALYSIS LITE - MÓDULO 2")
        print("  Creado por Bernard Orozco bajo tutela de Aslan")
        print("🔬" + "="*40 + "🔬")
        
        start_time = datetime.now()
        
        try:
            # Cargar modelo y datos
            model, X_train, y_train = self.load_model_and_data()
            
            # Rastrear gradientes
            records = self.track_gradients_lite(model, X_train, y_train, num_batches)
            
            # Detectar colapsos
            collapse_info = self.detect_collapse_points(records)
            
            # Análisis Pascanu
            pascanu_ratios = self.calculate_pascanu_ratios(records)
            
            # Visualización
            plot_path = self.create_basic_visualization(records)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # Compilar resultados
            results = {
                'analysis_metadata': {
                    'model_path': str(self.model_path),
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(), 
                    'duration_minutes': duration.total_seconds() / 60,
                    'batches_analyzed': len(records)
                },
                'collapse_analysis': collapse_info,
                'pascanu_analysis': {
                    'total_ratios': len(pascanu_ratios),
                    'has_vanishing': any(
                        any(r['status'] == 'vanishing' for r in rec['ratios']) 
                        for rec in pascanu_ratios
                    ),
                    'has_exploding': any(
                        any(r['status'] == 'exploding' for r in rec['ratios'])
                        for rec in pascanu_ratios
                    )
                },
                'visualization_path': plot_path
            }
            
            # Guardar reporte
            report_path = f"gradient_analysis_lite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Reporte final
            print(f"\n🎯 ANÁLISIS COMPLETADO:")
            print(f"   Duración: {duration.total_seconds()/60:.1f} minutos")
            print(f"   Batches: {len(records)}")
            print(f"   Reporte: {report_path}")
            print(f"   Gráfico: {plot_path}")
            
            if collapse_info['earliest_collapse'] >= 0:
                print(f"   🔴 Colapso detectado: Batch {collapse_info['earliest_collapse']}")
            else:
                print(f"   ✅ Sin colapso crítico detectado")
            
            return results
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="🔬 Gradient Analyzer Lite")
    parser.add_argument('--model', required=True, help='Modelo operado')
    parser.add_argument('--text', default='alice_wonderland.txt', help='Archivo de texto')
    parser.add_argument('--batches', type=int, default=30, help='Número de batches')
    
    args = parser.parse_args()
    
    analyzer = GradientAnalyzerLite(args.model, args.text)
    results = analyzer.run_complete_analysis(args.batches)
    
    if results:
        print("\n🎉 ANÁLISIS EXITOSO")
    else:
        sys.exit(1)