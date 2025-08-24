#!/usr/bin/env python3
"""
Ablation Experiments Analyzer - Task 3.1
Creado por Bernard Orozco

Implementación de experimentos de ablación sistemática para modelos LSTM
siguiendo la metodología de Melis et al. 2017 "On the State of the Art of 
Evaluation in Neural Language Models".

Teoría de Ablación:
- Ablación = remover sistemáticamente componentes del modelo
- Objetivo: identificar qué componentes contribuyen más al rendimiento
- Método: entrenar variantes del modelo con diferentes arquitecturas
- Métricas: comparar perplexity y loss entre variantes
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
from pathlib import Path
import shutil
import tempfile

class AblationExperimentRunner:
    """
    Runner de experimentos de ablación para modelos LSTM.
    
    Implementa ablaciones sistemáticas:
    1. Número de capas LSTM (1, 2, 3)
    2. Tamaño de unidades LSTM (128, 256, 512)
    3. Dropout rate (0.1, 0.3, 0.5)
    4. Embedding dimensions (64, 128, 256)
    5. Sequence length (50, 100, 200)
    """
    
    def __init__(self, base_model_path: str, training_data_path: str = None):
        """
        Inicializar runner de experimentos de ablación.
        
        Args:
            base_model_path: Ruta al modelo base entrenado
            training_data_path: Ruta a datos de entrenamiento (opcional)
        """
        self.base_model_path = base_model_path
        self.training_data_path = training_data_path
        self.base_model = None
        self.base_config = None
        self.results = {}
        
        # Cargar modelo base para extraer configuración
        self._load_base_model()
        
        # Configuraciones de experimentos
        self.ablation_configs = self._define_ablation_configs()
        
    def _load_base_model(self):
        """Cargar modelo base y extraer configuración."""
        try:
            self.base_model = tf.keras.models.load_model(self.base_model_path)
            
            # Extraer configuración del modelo base
            self.base_config = self._extract_model_config(self.base_model)
            print(f"✅ Modelo base cargado: {Path(self.base_model_path).name}")
            print(f"📊 Configuración detectada: {self.base_config}")
            
        except Exception as e:
            print(f"❌ Error cargando modelo base: {e}")
            raise
    
    def _extract_model_config(self, model: tf.keras.Model) -> Dict:
        """Extraer configuración del modelo existente."""
        config = {}
        
        # Detectar capas del modelo
        lstm_layers = [layer for layer in model.layers if 'lstm' in layer.name.lower()]
        embedding_layers = [layer for layer in model.layers if 'embedding' in layer.name.lower()]
        
        config['num_lstm_layers'] = len(lstm_layers)
        config['lstm_units'] = lstm_layers[0].units if lstm_layers else 256
        config['embedding_dim'] = embedding_layers[0].output_dim if embedding_layers else 128
        config['vocab_size'] = embedding_layers[0].input_dim if embedding_layers else 10000
        
        # Detectar dropout de las capas LSTM
        dropout_rate = 0.0
        for layer in lstm_layers:
            if hasattr(layer, 'dropout'):
                dropout_rate = layer.dropout
                break
        config['dropout_rate'] = dropout_rate
        
        # Detectar sequence length del input
        if model.input_shape and len(model.input_shape) > 1:
            config['sequence_length'] = model.input_shape[1] or 100
        else:
            config['sequence_length'] = 100
            
        return config
    
    def _define_ablation_configs(self) -> Dict[str, List]:
        """Definir configuraciones para experimentos de ablación."""
        base = self.base_config
        
        return {
            # Ablación de número de capas LSTM
            'lstm_layers': [
                {'num_lstm_layers': 1, 'name': 'single_layer'},
                {'num_lstm_layers': 2, 'name': 'double_layer'}, 
                {'num_lstm_layers': 3, 'name': 'triple_layer'}
            ],
            
            # Ablación de tamaño de unidades LSTM  
            'lstm_units': [
                {'lstm_units': 128, 'name': 'small_units'},
                {'lstm_units': 256, 'name': 'medium_units'},
                {'lstm_units': 512, 'name': 'large_units'}
            ],
            
            # Ablación de dropout rate
            'dropout_rate': [
                {'dropout_rate': 0.1, 'name': 'low_dropout'},
                {'dropout_rate': 0.3, 'name': 'medium_dropout'},
                {'dropout_rate': 0.5, 'name': 'high_dropout'}
            ],
            
            # Ablación de embedding dimensions
            'embedding_dim': [
                {'embedding_dim': 64, 'name': 'small_embedding'},
                {'embedding_dim': 128, 'name': 'medium_embedding'},
                {'embedding_dim': 256, 'name': 'large_embedding'}
            ],
            
            # Ablación de sequence length
            'sequence_length': [
                {'sequence_length': 50, 'name': 'short_sequences'},
                {'sequence_length': 100, 'name': 'medium_sequences'},
                {'sequence_length': 200, 'name': 'long_sequences'}
            ]
        }
    
    def run_ablation_study(self, experiment_types: List[str] = None, 
                          epochs: int = 5, quick_mode: bool = True) -> Dict:
        """
        Ejecutar estudio completo de ablación.
        
        Args:
            experiment_types: Lista de tipos de experimento a ejecutar
            epochs: Número de épocas para entrenar cada variante
            quick_mode: Si True, usa menos datos y épocas para rapidez
        
        Returns:
            Dict con resultados de todos los experimentos
        """
        if experiment_types is None:
            experiment_types = ['lstm_layers', 'lstm_units', 'dropout_rate']
        
        print("🧪 INICIANDO EXPERIMENTOS DE ABLACIÓN")
        print("=" * 60)
        print(f"🎯 Experimentos a ejecutar: {experiment_types}")
        print(f"⚡ Modo: {'Rápido' if quick_mode else 'Completo'}")
        print(f"📈 Épocas por experimento: {epochs}")
        print("=" * 60)
        
        all_results = {}
        
        for exp_type in experiment_types:
            print(f"\n🔬 EJECUTANDO ABLACIÓN: {exp_type.upper()}")
            print("-" * 40)
            
            experiment_results = self._run_experiment_type(
                exp_type, epochs, quick_mode
            )
            
            all_results[exp_type] = experiment_results
            
            # Mostrar resumen inmediato
            self._show_experiment_summary(exp_type, experiment_results)
        
        # Análisis comparativo final
        comparative_analysis = self._perform_comparative_analysis(all_results)
        all_results['comparative_analysis'] = comparative_analysis
        
        # Guardar resultados
        results_path = self._save_results(all_results)
        all_results['results_path'] = results_path
        
        print("\n🎉 EXPERIMENTOS DE ABLACIÓN COMPLETADOS")
        print(f"📁 Resultados guardados en: {results_path}")
        
        return all_results
    
    def _run_experiment_type(self, exp_type: str, epochs: int, 
                           quick_mode: bool) -> Dict:
        """Ejecutar un tipo específico de experimento de ablación."""
        configs = self.ablation_configs[exp_type]
        results = {}
        
        for i, config_variant in enumerate(configs, 1):
            variant_name = config_variant['name']
            print(f"   🧪 Variante {i}/{len(configs)}: {variant_name}")
            
            # Crear configuración completa combinando base + variante
            full_config = self.base_config.copy()
            full_config.update(config_variant)
            
            try:
                # Entrenar modelo con esta configuración
                model_metrics = self._train_ablation_model(
                    full_config, epochs, quick_mode
                )
                
                results[variant_name] = {
                    'config': full_config,
                    'metrics': model_metrics,
                    'success': True
                }
                
                print(f"      ✅ Completado - Loss final: {model_metrics['final_loss']:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                results[variant_name] = {
                    'config': full_config,
                    'metrics': None,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def _train_ablation_model(self, config: Dict, epochs: int, 
                            quick_mode: bool) -> Dict:
        """
        Entrenar un modelo individual para experimento de ablación.
        
        Nota: En modo rápido, usa datos sintéticos para acelerar.
        """
        # Construir modelo con la configuración especificada
        model = self._build_model_from_config(config)
        
        # Generar o cargar datos de entrenamiento
        if quick_mode or not self.training_data_path:
            # Usar datos sintéticos para rapidez
            X_train, y_train, X_val, y_val = self._generate_synthetic_data(config)
        else:
            # Cargar datos reales (implementación futura)
            X_train, y_train, X_val, y_val = self._load_real_training_data(config)
        
        # Configurar entrenamiento
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0,  # Silencioso para no saturar output
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=2, restore_best_weights=True
                )
            ]
        )
        
        # Calcular métricas finales
        final_loss = min(history.history['val_loss'])
        final_accuracy = max(history.history.get('val_accuracy', [0]))
        
        # Calcular perplexity
        perplexity = np.exp(final_loss)
        
        return {
            'final_loss': float(final_loss),
            'final_accuracy': float(final_accuracy),
            'perplexity': float(perplexity),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            },
            'total_params': model.count_params(),
            'epochs_trained': len(history.history['loss'])
        }
    
    def _build_model_from_config(self, config: Dict) -> tf.keras.Model:
        """Construir modelo LSTM basado en configuración."""
        model = tf.keras.Sequential()
        
        # Capa de embedding
        model.add(tf.keras.layers.Embedding(
            input_dim=config['vocab_size'],
            output_dim=config['embedding_dim'],
            input_length=config['sequence_length']
        ))
        
        # Capas LSTM
        for i in range(config['num_lstm_layers']):
            return_sequences = (i < config['num_lstm_layers'] - 1)
            
            model.add(tf.keras.layers.LSTM(
                units=config['lstm_units'],
                return_sequences=return_sequences,
                dropout=config['dropout_rate'],
                recurrent_dropout=config['dropout_rate']
            ))
        
        # Capa de salida
        model.add(tf.keras.layers.Dense(
            config['vocab_size'], 
            activation='softmax'
        ))
        
        return model
    
    def _generate_synthetic_data(self, config: Dict) -> Tuple:
        """Generar datos sintéticos para entrenamiento rápido."""
        vocab_size = config['vocab_size']
        sequence_length = config['sequence_length']
        
        # Generar secuencias aleatorias pero con cierta estructura
        num_samples = 1000 if config.get('quick_mode', True) else 5000
        
        # Datos de entrenamiento
        X_train = np.random.randint(0, vocab_size, (num_samples, sequence_length))
        y_train = np.random.randint(0, vocab_size, (num_samples, sequence_length))
        
        # Datos de validación
        X_val = np.random.randint(0, vocab_size, (200, sequence_length))
        y_val = np.random.randint(0, vocab_size, (200, sequence_length))
        
        return X_train, y_train, X_val, y_val
    
    def _load_real_training_data(self, config: Dict) -> Tuple:
        """Cargar datos reales de entrenamiento (implementación futura)."""
        # TODO: Implementar carga de datos reales
        # Por ahora, usar datos sintéticos
        return self._generate_synthetic_data(config)
    
    def _show_experiment_summary(self, exp_type: str, results: Dict):
        """Mostrar resumen de resultados de un experimento."""
        print(f"\n📊 RESUMEN - {exp_type.upper()}:")
        
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if not successful_results:
            print("   ❌ Ningún experimento completado exitosamente")
            return
        
        # Ordenar por perplexity (mejor = menor)
        sorted_results = sorted(
            successful_results.items(),
            key=lambda x: x[1]['metrics']['perplexity']
        )
        
        print("   Ranking por Perplexity (menor = mejor):")
        for rank, (name, data) in enumerate(sorted_results, 1):
            metrics = data['metrics']
            perplexity = metrics['perplexity']
            loss = metrics['final_loss']
            params = metrics['total_params']
            
            status = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
            
            print(f"   {status} {rank}. {name}:")
            print(f"      Perplexity: {perplexity:.2f} | Loss: {loss:.4f} | Params: {params:,}")
    
    def _perform_comparative_analysis(self, all_results: Dict) -> Dict:
        """Realizar análisis comparativo entre todos los experimentos."""
        analysis = {
            'best_overall': {},
            'parameter_efficiency': {},
            'ablation_insights': {}
        }
        
        # Encontrar mejor configuración general
        all_successful = []
        for exp_type, results in all_results.items():
            for variant_name, data in results.items():
                if data['success']:
                    all_successful.append({
                        'experiment': exp_type,
                        'variant': variant_name,
                        'metrics': data['metrics'],
                        'config': data['config']
                    })
        
        if all_successful:
            # Mejor por perplexity
            best_perplexity = min(all_successful, key=lambda x: x['metrics']['perplexity'])
            analysis['best_overall']['by_perplexity'] = best_perplexity
            
            # Mejor eficiencia (perplexity / parámetros)
            for result in all_successful:
                efficiency = result['metrics']['perplexity'] / result['metrics']['total_params'] * 1e6
                result['efficiency'] = efficiency
            
            best_efficiency = min(all_successful, key=lambda x: x['efficiency'])
            analysis['best_overall']['by_efficiency'] = best_efficiency
            
            # Insights de ablación
            analysis['ablation_insights'] = self._generate_ablation_insights(all_results)
        
        return analysis
    
    def _generate_ablation_insights(self, all_results: Dict) -> Dict:
        """Generar insights específicos de los experimentos de ablación."""
        insights = {}
        
        # Analizar impacto de cada componente
        for exp_type, results in all_results.items():
            successful = {k: v for k, v in results.items() if v['success']}
            
            if len(successful) >= 2:
                perplexities = [v['metrics']['perplexity'] for v in successful.values()]
                
                insights[exp_type] = {
                    'min_perplexity': float(min(perplexities)),
                    'max_perplexity': float(max(perplexities)),
                    'perplexity_range': float(max(perplexities) - min(perplexities)),
                    'impact_score': float((max(perplexities) - min(perplexities)) / min(perplexities))
                }
        
        return insights
    
    def _save_results(self, results: Dict) -> str:
        """Guardar resultados de experimentos."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ablation_study_{timestamp}.json"
        
        # Añadir metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'base_model': self.base_model_path,
            'base_config': self.base_config
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def generate_visualization(self, results: Dict) -> str:
        """Generar visualización de resultados de ablación."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ablation_visualization_{timestamp}.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resultados de Experimentos de Ablación', fontsize=16)
        
        # Plot 1: Perplexity por tipo de experimento
        ax1 = axes[0, 0]
        self._plot_perplexity_comparison(results, ax1)
        
        # Plot 2: Eficiencia de parámetros
        ax2 = axes[0, 1]
        self._plot_parameter_efficiency(results, ax2)
        
        # Plot 3: Impacto de ablaciones
        ax3 = axes[1, 0]
        self._plot_ablation_impact(results, ax3)
        
        # Plot 4: Histórico de entrenamiento del mejor modelo
        ax4 = axes[1, 1]
        self._plot_best_training_history(results, ax4)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _plot_perplexity_comparison(self, results: Dict, ax):
        """Plot comparativo de perplexity."""
        experiments = []
        variants = []
        perplexities = []
        
        for exp_type, exp_results in results.items():
            if exp_type in ['comparative_analysis', 'metadata', 'results_path']:
                continue
                
            for variant, data in exp_results.items():
                if data['success']:
                    experiments.append(exp_type)
                    variants.append(variant)
                    perplexities.append(data['metrics']['perplexity'])
        
        if perplexities:
            bars = ax.bar(range(len(perplexities)), perplexities)
            ax.set_xlabel('Experimentos')
            ax.set_ylabel('Perplexity')
            ax.set_title('Perplexity por Configuración')
            ax.set_xticks(range(len(variants)))
            ax.set_xticklabels([f"{e}\n{v}" for e, v in zip(experiments, variants)], 
                              rotation=45, ha='right')
            
            # Colorear mejor resultado
            min_idx = perplexities.index(min(perplexities))
            bars[min_idx].set_color('gold')
    
    def _plot_parameter_efficiency(self, results: Dict, ax):
        """Plot de eficiencia de parámetros."""
        params = []
        perplexities = []
        labels = []
        
        for exp_type, exp_results in results.items():
            if exp_type in ['comparative_analysis', 'metadata', 'results_path']:
                continue
                
            for variant, data in exp_results.items():
                if data['success']:
                    params.append(data['metrics']['total_params'])
                    perplexities.append(data['metrics']['perplexity'])
                    labels.append(f"{exp_type}_{variant}")
        
        if params:
            scatter = ax.scatter(params, perplexities, alpha=0.7, s=100)
            ax.set_xlabel('Número de Parámetros')
            ax.set_ylabel('Perplexity')
            ax.set_title('Eficiencia: Parámetros vs Rendimiento')
            
            # Añadir línea de tendencia
            if len(params) > 1:
                z = np.polyfit(params, perplexities, 1)
                p = np.poly1d(z)
                ax.plot(params, p(params), "r--", alpha=0.8)
    
    def _plot_ablation_impact(self, results: Dict, ax):
        """Plot del impacto de cada tipo de ablación."""
        comparative = results.get('comparative_analysis', {})
        insights = comparative.get('ablation_insights', {})
        
        if insights:
            exp_types = list(insights.keys())
            impacts = [insights[exp]['impact_score'] for exp in exp_types]
            
            bars = ax.bar(exp_types, impacts)
            ax.set_xlabel('Tipo de Experimento')
            ax.set_ylabel('Score de Impacto')
            ax.set_title('Impacto de cada Componente')
            ax.tick_params(axis='x', rotation=45)
            
            # Colorear por impacto
            for bar, impact in zip(bars, impacts):
                if impact > 0.1:
                    bar.set_color('red')
                elif impact > 0.05:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
    
    def _plot_best_training_history(self, results: Dict, ax):
        """Plot del histórico de entrenamiento del mejor modelo."""
        comparative = results.get('comparative_analysis', {})
        best_overall = comparative.get('best_overall', {})
        best_by_perplexity = best_overall.get('by_perplexity')
        
        if best_by_perplexity:
            history = best_by_perplexity['metrics']['training_history']
            epochs = range(1, len(history['loss']) + 1)
            
            ax.plot(epochs, history['loss'], 'b-', label='Training Loss')
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax.set_xlabel('Época')
            ax.set_ylabel('Loss')
            ax.set_title(f"Mejor Modelo: {best_by_perplexity['variant']}")
            ax.legend()
            ax.grid(True, alpha=0.3)


def run_quick_ablation_study(model_path: str, experiment_types: List[str] = None) -> Dict:
    """
    Función conveniente para ejecutar estudio rápido de ablación.
    
    Args:
        model_path: Ruta al modelo base
        experiment_types: Tipos de experimento a ejecutar
    
    Returns:
        Dict con resultados completos
    """
    print("🧪 INICIANDO ESTUDIO RÁPIDO DE ABLACIÓN")
    print("=" * 50)
    
    try:
        # Inicializar runner
        runner = AblationExperimentRunner(model_path)
        
        # Ejecutar experimentos
        results = runner.run_ablation_study(
            experiment_types=experiment_types or ['lstm_units', 'dropout_rate'],
            epochs=3,  # Rápido para demostración
            quick_mode=True
        )
        
        # Generar visualización
        viz_path = runner.generate_visualization(results)
        results['visualization_path'] = viz_path
        
        print(f"\n🎉 ESTUDIO COMPLETADO")
        print(f"📊 Visualización: {viz_path}")
        
        # Mostrar insights principales
        comparative = results.get('comparative_analysis', {})
        best_overall = comparative.get('best_overall', {})
        
        if 'by_perplexity' in best_overall:
            best = best_overall['by_perplexity']
            print(f"\n🏆 MEJOR CONFIGURACIÓN:")
            print(f"   Experimento: {best['experiment']}")
            print(f"   Variante: {best['variant']}")
            print(f"   Perplexity: {best['metrics']['perplexity']:.2f}")
            print(f"   Parámetros: {best['metrics']['total_params']:,}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en estudio de ablación: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python ablation_analyzer.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    results = run_quick_ablation_study(model_path)