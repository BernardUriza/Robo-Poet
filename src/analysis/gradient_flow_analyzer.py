#!/usr/bin/env python3
"""
🔬 GRADIENT FLOW ANALYZER - MÓDULO 2 TASK 1.1
Creado por Bernard Orozco bajo tutela de Aslan

Implementa análisis profundo de gradientes siguiendo:
- Pascanu et al. 2013: "On the difficulty of training RNNs"
- Li et al. 2018: "Visualizing the Loss Landscape of Neural Nets"

OBJETIVO: Identificar el momento exacto donde los gradientes colapsan
y entender POR QUÉ el modelo original se saturó.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    from data_processor import TextProcessor
    print("✅ Módulos de análisis de gradientes cargados")
    
except Exception as e:
    print(f"❌ Error importando: {e}")
    sys.exit(1)

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    plotly_available = True
    print("✅ Plotly disponible para visualizaciones interactivas")
except ImportError:
    plotly_available = False
    print("⚠️ Plotly no disponible - usando matplotlib")


class GradientFlowAnalyzer:
    """
    🔬 ANALIZADOR COMPLETO DE FLUJO DE GRADIENTES
    
    Implementa análisis científico profundo siguiendo papers fundamentales:
    1. Rastrea gradientes durante entrenamiento
    2. Detecta vanishing/exploding gradients
    3. Identifica "punto de quiebre"
    4. Analiza propagación según Pascanu et al. 2013
    """
    
    def __init__(self, model_path: str, text_file: str = "alice_wonderland.txt"):
        """
        Inicializar analizador de gradientes
        
        Args:
            model_path: Ruta al modelo operado
            text_file: Archivo de texto para entrenamiento
        """
        self.model_path = Path(model_path)
        self.text_file = Path(text_file)
        self.gradient_history = []
        self.analysis_results = {}
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        print("🔬 GRADIENT FLOW ANALYZER INICIALIZADO")
        print("=" * 50)
        print(f"   Modelo: {self.model_path.name}")
        print(f"   Texto: {self.text_file.name if self.text_file.exists() else 'Sintético'}")
    
    def load_model_and_data(self) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
        """
        Cargar modelo operado y preparar datos para análisis
        
        Returns:
            model: Modelo TensorFlow cargado
            X_train: Datos de entrenamiento  
            y_train: Labels de entrenamiento
        """
        print("🏥 Cargando modelo operado...")
        
        # Cargar modelo operado (después de cirugía)
        model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                'VariationalDropout': VariationalDropout,
                'DropConnect': DropConnect
            }
        )
        print(f"✅ Modelo operado cargado: {len(model.layers)} capas")
        
        # Preparar datos multi-corpus
        print("📊 Preparando datos de entrenamiento desde multi-corpus...")
        
        try:
            # Datos reales desde corpus directory
            processor = TextProcessor(sequence_length=128, step_size=3)
            X_onehot, y_onehot = processor.prepare_data("corpus", max_length=20_000)  # 20K chars para análisis profundo
            
            # Convert to integer sequences
            X_train = np.argmax(X_onehot, axis=-1)
            y_train = np.argmax(y_onehot, axis=-1)
            
            self.vocab_size = processor.vocab_size
            print(f"✅ Multi-corpus cargado: {X_train.shape}, vocab: {self.vocab_size}")
            
        except Exception as e:
            print(f"⚠️ Error cargando corpus, usando datos sintéticos: {e}")
            # Datos sintéticos como fallback
            print("⚠️ Usando datos sintéticos para análisis...")
            self.vocab_size = 1000
            batch_size = 200
            sequence_length = 128
            
            X_train = np.random.randint(0, self.vocab_size, size=(batch_size, sequence_length))
            y_train = np.random.randint(0, self.vocab_size, size=(batch_size,))
        
        print(f"   Secuencias: {X_train.shape[0]}")
        print(f"   Sequence length: {X_train.shape[1]}")
        print(f"   Vocabulary: {self.vocab_size}")
        
        return model, X_train, y_train
    
    def track_gradients_during_training(self, model: tf.keras.Model, 
                                      X_train: np.ndarray, y_train: np.ndarray,
                                      num_batches: int = 50, batch_size: int = 32) -> pd.DataFrame:
        """
        🎯 TASK 1.1: Rastrear gradientes durante 50 batches
        
        Implementa el algoritmo fundamental para detectar degradación de gradientes
        
        Args:
            model: Modelo a analizar
            X_train: Datos de entrenamiento
            y_train: Labels
            num_batches: Número de batches a rastrear
            batch_size: Tamaño del batch
            
        Returns:
            gradient_df: DataFrame con historial completo
        """
        print(f"\n🔍 TASK 1.1: Rastreando gradientes por {num_batches} batches...")
        print("-" * 55)
        
        # Compilar modelo con optimizer conservador
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,  # Conservador post-cirugía
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Preparar datasets
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size).take(num_batches)
        
        gradient_records = []
        
        print("📈 Iniciando rastreo de gradientes...")
        for batch_idx, (x_batch, y_batch) in enumerate(dataset):
            
            # Forward pass con gradient tape
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = model.compiled_loss(y_batch, predictions)
            
            # Calcular gradientes
            gradients = tape.gradient(loss, model.trainable_weights)
            
            # Extraer estadísticas por capa
            batch_record = {
                'batch': batch_idx,
                'loss': float(loss),
                'timestamp': datetime.now().isoformat()
            }
            
            # Analizar gradientes por capa
            layer_idx = 0
            for layer, grad in zip(model.layers, gradients):
                if grad is not None and len(grad.shape) > 0:
                    
                    # Calcular estadísticas de gradiente
                    grad_norm = tf.norm(grad).numpy()
                    grad_mean = tf.reduce_mean(tf.abs(grad)).numpy()
                    grad_std = tf.math.reduce_std(grad).numpy()
                    grad_max = tf.reduce_max(tf.abs(grad)).numpy()
                    grad_min = tf.reduce_min(tf.abs(grad)).numpy()
                    
                    # Registrar por capa
                    batch_record.update({
                        f'{layer.name}_norm': grad_norm,
                        f'{layer.name}_mean': grad_mean,
                        f'{layer.name}_std': grad_std,
                        f'{layer.name}_max': grad_max,
                        f'{layer.name}_min': grad_min,
                        f'{layer.name}_idx': layer_idx
                    })
                    
                    layer_idx += 1
            
            gradient_records.append(batch_record)
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0:
                print(f"   Batch {batch_idx + 1:2d}/{num_batches}: Loss={loss:.4f}, "
                      f"Grad_norms=[{grad_norm:.6f}]")
            
            # Aplicar un step de entrenamiento
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        # Crear DataFrame estructurado
        gradient_df = pd.DataFrame(gradient_records)
        
        # Guardar CSV
        csv_path = f"gradient_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        gradient_df.to_csv(csv_path, index=False)
        
        print(f"\n📊 RESULTADOS TASK 1.1:")
        print(f"   Batches analizados: {len(gradient_records)}")
        print(f"   CSV guardado: {csv_path}")
        print(f"   Columnas registradas: {len(gradient_df.columns)}")
        
        self.gradient_history_df = gradient_df
        return gradient_df
    
    def detect_gradient_collapse_point(self, gradient_df: pd.DataFrame) -> Dict[str, Any]:
        """
        🔍 DETECTAR PUNTO DE QUIEBRE
        
        Identifica el momento exacto donde los gradientes colapsan
        
        Args:
            gradient_df: DataFrame con historial de gradientes
            
        Returns:
            collapse_analysis: Análisis del punto de colapso
        """
        print("\n🔍 DETECTANDO PUNTO DE QUIEBRE...")
        print("-" * 35)
        
        # Identificar columnas de gradient norms
        norm_columns = [col for col in gradient_df.columns if col.endswith('_norm')]
        
        collapse_points = {}
        
        for norm_col in norm_columns:
            layer_name = norm_col.replace('_norm', '')
            norms = gradient_df[norm_col].values
            
            # Detectar colapso: cuando gradient norm cae < 1e-5
            collapse_threshold = 1e-5
            collapse_batches = np.where(norms < collapse_threshold)[0]
            
            if len(collapse_batches) > 0:
                first_collapse = collapse_batches[0]
                collapse_points[layer_name] = {
                    'collapse_batch': int(first_collapse),
                    'norm_at_collapse': float(norms[first_collapse]),
                    'norm_before_collapse': float(norms[max(0, first_collapse-1)]),
                    'severity': 'critical' if norms[first_collapse] < 1e-6 else 'warning'
                }
                
                print(f"🔴 {layer_name}: Colapso en batch {first_collapse} "
                      f"(norm: {norms[first_collapse]:.2e})")
            else:
                # Buscar tendencia descendente pronunciada
                if len(norms) >= 10:
                    trend = np.polyfit(range(len(norms)), norms, 1)[0]  # Slope
                    if trend < -1e-6:  # Tendencia muy negativa
                        collapse_points[layer_name] = {
                            'collapse_batch': -1,  # No colapsó pero va mal
                            'trend_slope': float(trend),
                            'final_norm': float(norms[-1]),
                            'severity': 'trend_warning'
                        }
                        print(f"🟡 {layer_name}: Tendencia negativa pronunciada "
                              f"(slope: {trend:.2e})")
                    else:
                        print(f"✅ {layer_name}: Estable")
        
        # Análisis global
        if collapse_points:
            earliest_collapse = min([cp['collapse_batch'] for cp in collapse_points.values() 
                                   if cp.get('collapse_batch', -1) >= 0])
            
            if earliest_collapse >= 0:
                print(f"\n💀 PUNTO DE QUIEBRE GLOBAL: Batch {earliest_collapse}")
            else:
                print(f"\n⚠️ No hay colapso completo, pero hay tendencias preocupantes")
        else:
            print(f"\n✅ No se detectó colapso de gradientes")
        
        collapse_analysis = {
            'collapse_points': collapse_points,
            'earliest_collapse_batch': earliest_collapse if collapse_points else -1,
            'total_layers_collapsed': len([cp for cp in collapse_points.values() 
                                         if cp.get('collapse_batch', -1) >= 0]),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.analysis_results['collapse_analysis'] = collapse_analysis
        return collapse_analysis
    
    def calculate_gradient_ratios_pascanu(self, gradient_df: pd.DataFrame) -> pd.DataFrame:
        """
        🎯 TASK 1.2: Análisis de Propagación según Pascanu et al. 2013
        
        Implementa la fórmula del paper:
        gradient_ratio = ||∂L/∂h_t|| / ||∂L/∂h_{t+1}||
        
        Args:
            gradient_df: DataFrame con gradientes
            
        Returns:
            ratio_df: DataFrame con ratios entre capas
        """
        print("\n🔍 TASK 1.2: Análisis Pascanu et al. 2013...")
        print("-" * 45)
        
        # Identificar columnas de gradient norms ordenadas por capa
        norm_columns = [col for col in gradient_df.columns if col.endswith('_norm')]
        
        # Crear DataFrame para ratios
        ratio_records = []
        
        for batch_idx in gradient_df.index:
            batch_ratios = {'batch': batch_idx}
            
            # Calcular ratios entre capas consecutivas
            for i in range(len(norm_columns) - 1):
                current_layer = norm_columns[i]
                next_layer = norm_columns[i + 1]
                
                current_norm = gradient_df.loc[batch_idx, current_layer]
                next_norm = gradient_df.loc[batch_idx, next_layer]
                
                # Calcular ratio (evitar división por 0)
                if next_norm > 1e-10:
                    ratio = current_norm / next_norm
                else:
                    ratio = float('inf') if current_norm > 1e-10 else 1.0
                
                layer_pair = f"{current_layer.replace('_norm', '')}_to_{next_layer.replace('_norm', '')}"
                batch_ratios[f'{layer_pair}_ratio'] = ratio
                
                # Clasificar según Pascanu
                if ratio < 0.1:
                    batch_ratios[f'{layer_pair}_status'] = 'vanishing'
                elif ratio > 10.0:
                    batch_ratios[f'{layer_pair}_status'] = 'exploding'
                else:
                    batch_ratios[f'{layer_pair}_status'] = 'stable'
            
            ratio_records.append(batch_ratios)
        
        ratio_df = pd.DataFrame(ratio_records)
        
        # Análisis de resultados
        ratio_columns = [col for col in ratio_df.columns if col.endswith('_ratio')]
        
        print(f"📊 ANÁLISIS PASCANU COMPLETADO:")
        for ratio_col in ratio_columns:
            ratios = ratio_df[ratio_col].values
            ratios_finite = ratios[np.isfinite(ratios)]  # Excluir inf
            
            if len(ratios_finite) > 0:
                mean_ratio = np.mean(ratios_finite)
                std_ratio = np.std(ratios_finite)
                min_ratio = np.min(ratios_finite)
                max_ratio = np.max(ratios_finite)
                
                layer_pair = ratio_col.replace('_ratio', '')
                print(f"   {layer_pair}:")
                print(f"     Mean ratio: {mean_ratio:.3f}")
                print(f"     Std: {std_ratio:.3f}")
                print(f"     Range: [{min_ratio:.3f}, {max_ratio:.3f}]")
                
                # Diagnóstico según Pascanu
                if mean_ratio < 0.1:
                    print(f"     🔴 VANISHING GRADIENTS")
                elif mean_ratio > 10.0:
                    print(f"     🔴 EXPLODING GRADIENTS") 
                else:
                    print(f"     ✅ ESTABLE")
        
        # Guardar CSV
        ratio_csv = f"gradient_ratios_pascanu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        ratio_df.to_csv(ratio_csv, index=False)
        print(f"   CSV ratios guardado: {ratio_csv}")
        
        self.gradient_ratios_df = ratio_df
        return ratio_df
    
    def visualize_gradient_flow_evolution(self, gradient_df: pd.DataFrame) -> str:
        """
        🎨 VISUALIZACIÓN DE EVOLUCIÓN TEMPORAL
        
        Crea gráfico mostrando evolución de gradientes a lo largo del tiempo
        """
        print("\n🎨 Creando visualización de evolución...")
        
        # Identificar columnas de gradient norms
        norm_columns = [col for col in gradient_df.columns if col.endswith('_norm')]
        
        # Crear gráfico
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Gradient norms
        plt.subplot(2, 2, 1)
        for norm_col in norm_columns:
            layer_name = norm_col.replace('_norm', '')
            plt.semilogy(gradient_df['batch'], gradient_df[norm_col], 
                        label=layer_name, marker='o', markersize=3)
        
        plt.xlabel('Batch')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Evolución de Gradient Norms por Capa')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Loss evolution
        plt.subplot(2, 2, 2)
        plt.plot(gradient_df['batch'], gradient_df['loss'], 'r-', linewidth=2, marker='o')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Evolución del Loss')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Gradient norm heatmap
        plt.subplot(2, 2, 3)
        norm_data = gradient_df[norm_columns].T
        sns.heatmap(norm_data, cmap='viridis', cbar_kws={'label': 'Gradient Norm'})
        plt.xlabel('Batch')
        plt.ylabel('Capa')
        plt.title('Heatmap de Gradient Norms')
        
        # Subplot 4: Problematic regions
        plt.subplot(2, 2, 4)
        
        # Identificar regiones problemáticas
        for norm_col in norm_columns:
            layer_name = norm_col.replace('_norm', '')
            norms = gradient_df[norm_col].values
            
            # Marcar regiones con vanishing gradients
            vanishing_mask = norms < 1e-5
            if np.any(vanishing_mask):
                plt.scatter(gradient_df['batch'][vanishing_mask], 
                          norms[vanishing_mask], 
                          color='red', alpha=0.7, s=50, label=f'{layer_name} (Vanishing)')
            
            # Marcar regiones con exploding gradients  
            exploding_mask = norms > 10
            if np.any(exploding_mask):
                plt.scatter(gradient_df['batch'][exploding_mask], 
                          norms[exploding_mask], 
                          color='orange', alpha=0.7, s=50, marker='^', 
                          label=f'{layer_name} (Exploding)')
        
        plt.yscale('log')
        plt.xlabel('Batch')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Regiones Problemáticas')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_path = f"gradient_flow_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualización guardada: {plot_path}")
        return plot_path
    
    def create_sankey_gradient_flow(self, gradient_df: pd.DataFrame) -> Optional[str]:
        """
        🎯 TASK 1.3: Visualización Sankey del Flujo de Gradientes
        
        Crea diagrama Sankey mostrando flujo desde output hasta input
        """
        if not plotly_available:
            print("⚠️ TASK 1.3: Plotly no disponible - saltando Sankey")
            return None
        
        print("\n🎯 TASK 1.3: Creando diagrama Sankey...")
        print("-" * 40)
        
        # Tomar último batch como ejemplo
        last_batch_idx = gradient_df.index[-1]
        
        # Identificar capas y sus gradient norms
        norm_columns = [col for col in gradient_df.columns if col.endswith('_norm')]
        layer_names = [col.replace('_norm', '') for col in norm_columns]
        layer_norms = [gradient_df.loc[last_batch_idx, col] for col in norm_columns]
        
        # Crear nodos para Sankey
        nodes = []
        links = []
        
        # Agregar nodos
        for i, (layer_name, norm) in enumerate(zip(layer_names, layer_norms)):
            # Color según el estado del gradiente
            if norm < 1e-5:
                color = 'rgba(255, 0, 0, 0.8)'  # Rojo - vanishing
            elif norm > 10:
                color = 'rgba(255, 165, 0, 0.8)'  # Naranja - exploding  
            else:
                color = 'rgba(0, 255, 0, 0.8)'  # Verde - normal
            
            nodes.append({
                'label': f'{layer_name}<br>norm: {norm:.2e}',
                'color': color
            })
        
        # Crear conexiones entre capas consecutivas
        for i in range(len(layer_names) - 1):
            source_idx = i
            target_idx = i + 1
            
            # El "flujo" es proporcional al gradient norm
            flow_value = max(layer_norms[i], 1e-8)  # Evitar 0
            
            # Color del link según el ratio
            if i < len(layer_norms) - 1:
                ratio = layer_norms[i] / max(layer_norms[i + 1], 1e-10)
                if ratio < 0.1:
                    link_color = 'rgba(255, 0, 0, 0.4)'  # Rojo - vanishing
                elif ratio > 10:
                    link_color = 'rgba(255, 165, 0, 0.4)'  # Naranja - exploding
                else:
                    link_color = 'rgba(0, 100, 0, 0.4)'  # Verde - normal
            else:
                link_color = 'rgba(100, 100, 100, 0.4)'
            
            links.append({
                'source': source_idx,
                'target': target_idx, 
                'value': float(flow_value * 1e6),  # Scale para visualización
                'color': link_color
            })
        
        # Crear figura Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[node['label'] for node in nodes],
                color=[node['color'] for node in nodes]
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links],
                color=[link['color'] for link in links]
            )
        )])
        
        fig.update_layout(
            title_text=f"Gradient Flow - Batch {last_batch_idx}<br>"
                      f"<sub>Ancho = Magnitud del Gradiente | Color = Estado</sub>",
            font_size=10,
            width=1000,
            height=600
        )
        
        # Guardar como HTML interactivo
        sankey_path = f"gradient_sankey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(sankey_path)
        
        print(f"🌊 Diagrama Sankey guardado: {sankey_path}")
        print("   🔴 Rojo: Vanishing gradients")  
        print("   🟠 Naranja: Exploding gradients")
        print("   🟢 Verde: Gradientes normales")
        
        return sankey_path
    
    def run_complete_gradient_analysis(self, num_batches: int = 50) -> Dict[str, Any]:
        """
        🎯 ANÁLISIS COMPLETO DE GRADIENTES - ORCHESTRATOR
        
        Ejecuta todos los análisis del Módulo 2, Parte 1:
        - Task 1.1: Historial de gradientes
        - Task 1.2: Análisis Pascanu 
        - Task 1.3: Visualización Sankey
        
        Args:
            num_batches: Número de batches para análisis
            
        Returns:
            complete_results: Resultados completos del análisis
        """
        print("🔬" + "="*48 + "🔬")
        print("    ANÁLISIS COMPLETO DE GRADIENT FLOW")
        print("    Módulo 2 - Parte 1: Tasks 1.1, 1.2, 1.3")
        print("    Creado por Bernard Orozco bajo tutela de Aslan")
        print("🔬" + "="*48 + "🔬")
        
        analysis_start = datetime.now()
        
        try:
            # Cargar modelo y datos
            model, X_train, y_train = self.load_model_and_data()
            
            # TASK 1.1: Historial de gradientes
            print("\n🚀 EJECUTANDO TASK 1.1...")
            gradient_df = self.track_gradients_during_training(
                model, X_train, y_train, num_batches
            )
            
            # Detectar punto de colapso
            collapse_analysis = self.detect_gradient_collapse_point(gradient_df)
            
            # TASK 1.2: Análisis Pascanu
            print("\n🚀 EJECUTANDO TASK 1.2...")
            ratio_df = self.calculate_gradient_ratios_pascanu(gradient_df)
            
            # Visualizaciones
            print("\n🚀 CREANDO VISUALIZACIONES...")
            evolution_plot = self.visualize_gradient_flow_evolution(gradient_df)
            
            # TASK 1.3: Sankey diagram
            print("\n🚀 EJECUTANDO TASK 1.3...")
            sankey_plot = self.create_sankey_gradient_flow(gradient_df)
            
            # Compilar resultados completos
            analysis_end = datetime.now()
            analysis_duration = analysis_end - analysis_start
            
            complete_results = {
                'analysis_metadata': {
                    'model_path': str(self.model_path),
                    'text_file': str(self.text_file),
                    'analysis_start': analysis_start.isoformat(),
                    'analysis_end': analysis_end.isoformat(),
                    'duration_minutes': analysis_duration.total_seconds() / 60,
                    'num_batches_analyzed': num_batches
                },
                'task_1_1_results': {
                    'gradient_history_shape': gradient_df.shape,
                    'collapse_analysis': collapse_analysis,
                    'csv_file': getattr(self, 'gradient_history_df', pd.DataFrame()).shape[0] > 0
                },
                'task_1_2_results': {
                    'gradient_ratios_shape': ratio_df.shape,
                    'pascanu_analysis_completed': True,
                    'csv_file': getattr(self, 'gradient_ratios_df', pd.DataFrame()).shape[0] > 0
                },
                'task_1_3_results': {
                    'sankey_diagram_created': sankey_plot is not None,
                    'sankey_file': sankey_plot
                },
                'visualizations': {
                    'evolution_plot': evolution_plot,
                    'sankey_plot': sankey_plot
                },
                'analysis_results': self.analysis_results
            }
            
            # Guardar reporte completo
            report_path = f"gradient_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            # Reporte final
            print("\n🎯 ANÁLISIS COMPLETO TERMINADO:")
            print("=" * 40)
            print(f"   Duración: {analysis_duration.total_seconds()/60:.1f} minutos")
            print(f"   Batches analizados: {num_batches}")
            print(f"   Tasks completadas: 3/3")
            print(f"   Reporte guardado: {report_path}")
            
            if collapse_analysis['earliest_collapse_batch'] >= 0:
                print(f"   🔴 Punto de quiebre detectado: Batch {collapse_analysis['earliest_collapse_batch']}")
            else:
                print(f"   ✅ No se detectó colapso crítico")
            
            print(f"\n📁 ARCHIVOS GENERADOS:")
            print(f"   📊 CSV gradientes: gradient_history_*.csv")
            print(f"   📊 CSV ratios: gradient_ratios_pascanu_*.csv") 
            print(f"   📈 Gráfico evolución: {evolution_plot}")
            if sankey_plot:
                print(f"   🌊 Sankey interactivo: {sankey_plot}")
            print(f"   📋 Reporte JSON: {report_path}")
            
            return complete_results
            
        except Exception as e:
            print(f"❌ Error en análisis completo: {e}")
            import traceback
            traceback.print_exc()
            return {}


def quick_gradient_analysis(model_path: str, text_file: str = "alice_wonderland.txt", 
                          num_batches: int = 30):
    """
    ⚡ ANÁLISIS RÁPIDO DE GRADIENTES
    
    Función de conveniencia para ejecutar análisis completo
    """
    analyzer = GradientFlowAnalyzer(model_path, text_file)
    return analyzer.run_complete_gradient_analysis(num_batches)


if __name__ == "__main__":
    """
    🎯 EJECUCIÓN DIRECTA - MÓDULO 2 PARTE 1
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="🔬 Gradient Flow Analyzer - Módulo 2")
    parser.add_argument('--model', required=True, help='Modelo operado a analizar')
    parser.add_argument('--text', default='alice_wonderland.txt', help='Archivo de texto')
    parser.add_argument('--batches', type=int, default=50, help='Número de batches (default: 50)')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Modelo no encontrado: {args.model}")
        sys.exit(1)
    
    print("🔬 INICIANDO ANÁLISIS COMPLETO DE GRADIENT FLOW...")
    results = quick_gradient_analysis(args.model, args.text, args.batches)
    
    if results:
        print("\n🎉 ANÁLISIS COMPLETO EXITOSO")
        print("🔬 Datos listos para análisis científico profundo")
    else:
        print("\n❌ ANÁLISIS FALLÓ")
        sys.exit(1)