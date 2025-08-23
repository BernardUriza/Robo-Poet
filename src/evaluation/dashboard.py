"""
Dashboard de métricas usando TensorBoard y visualizaciones personalizadas.

Implementa logging de métricas a TensorBoard y dashboard web
para monitoreo en tiempo real del entrenamiento.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

from .monitor import TrainingMetrics
from .metrics import EvaluationResults

logger = logging.getLogger(__name__)

# TensorBoard imports con fallback
try:
    import tensorflow as tf
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorFlow not available. TensorBoard logging disabled.")


class TensorBoardDashboard:
    """Dashboard de TensorBoard para métricas de entrenamiento."""
    
    def __init__(self, log_dir: str = "./logs/tensorboard", experiment_name: str = "robo_poet"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.available = TENSORBOARD_AVAILABLE
        
        if self.available:
            self._setup_tensorboard()
        else:
            logger.warning("TensorBoard not available - metrics will only be logged to file")
    
    def _setup_tensorboard(self):
        """Configura TensorBoard logging."""
        # Crear directorio de logs
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear writers para diferentes tipos de métricas
        timestamp = int(time.time())
        
        self.train_writer = tf.summary.create_file_writer(
            str(self.log_dir / f"{self.experiment_name}_train_{timestamp}")
        )
        
        self.val_writer = tf.summary.create_file_writer(
            str(self.log_dir / f"{self.experiment_name}_val_{timestamp}")
        )
        
        self.eval_writer = tf.summary.create_file_writer(
            str(self.log_dir / f"{self.experiment_name}_eval_{timestamp}")
        )
        
        logger.info(f"TensorBoard logging initialized at {self.log_dir}")
        logger.info(f"Start TensorBoard with: tensorboard --logdir {self.log_dir}")
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """Registra métricas de entrenamiento en TensorBoard."""
        if not self.available:
            return
        
        step = metrics.epoch * metrics.total_batches + metrics.batch
        
        # Métricas de entrenamiento
        with self.train_writer.as_default():
            tf.summary.scalar('loss', metrics.train_loss, step=step)
            tf.summary.scalar('accuracy', metrics.train_accuracy, step=step)
            tf.summary.scalar('perplexity', metrics.train_perplexity, step=step)
            tf.summary.scalar('learning_rate', metrics.learning_rate, step=step)
            tf.summary.scalar('training_speed', metrics.training_speed, step=step)
            tf.summary.scalar('gpu_memory_mb', metrics.gpu_memory_used, step=step)
        
        # Métricas de validación
        if metrics.val_loss > 0:
            with self.val_writer.as_default():
                tf.summary.scalar('loss', metrics.val_loss, step=step)
                tf.summary.scalar('accuracy', metrics.val_accuracy, step=step)
                tf.summary.scalar('perplexity', metrics.val_perplexity, step=step)
        
        # Métricas de evaluación
        if metrics.evaluation_results:
            self.log_evaluation_results(metrics.evaluation_results, step)
    
    def log_evaluation_results(self, results: EvaluationResults, step: int):
        """Registra resultados de evaluación en TensorBoard."""
        if not self.available:
            return
        
        with self.eval_writer.as_default():
            # Métricas principales
            tf.summary.scalar('bleu_score', results.bleu_score, step=step)
            tf.summary.scalar('perplexity', results.perplexity, step=step)
            
            # ROUGE scores
            tf.summary.scalar('rouge_1', results.rouge_1, step=step)
            tf.summary.scalar('rouge_2', results.rouge_2, step=step)
            tf.summary.scalar('rouge_l', results.rouge_l, step=step)
            
            # Diversidad
            tf.summary.scalar('unigram_diversity', results.unigram_diversity, step=step)
            tf.summary.scalar('bigram_diversity', results.bigram_diversity, step=step)
            tf.summary.scalar('trigram_diversity', results.trigram_diversity, step=step)
            
            # Métricas adicionales
            tf.summary.scalar('repetition_ratio', results.repetition_ratio, step=step)
            tf.summary.scalar('avg_sentence_length', results.avg_sentence_length, step=step)
            tf.summary.scalar('vocabulary_usage', results.vocabulary_usage, step=step)
            
            # Score resumen
            summary_score = results.get_summary_score()
            tf.summary.scalar('summary_score', summary_score, step=step)
    
    def log_text_samples(self, samples: Dict[str, List[str]], step: int, max_samples: int = 5):
        """Registra muestras de texto generado."""
        if not self.available:
            return
        
        references = samples.get('references', [])
        generated = samples.get('generated', [])
        
        if not references or not generated:
            return
        
        # Limitar número de muestras
        num_samples = min(len(references), len(generated), max_samples)
        
        with self.eval_writer.as_default():
            for i in range(num_samples):
                # Crear tabla de comparación
                comparison_text = f"""
**Reference:**
{references[i]}

**Generated:**
{generated[i]}

---
"""
                tf.summary.text(f'sample_{i}', comparison_text, step=step)
    
    def log_model_architecture(self, model_summary: str):
        """Registra resumen de arquitectura del modelo."""
        if not self.available:
            return
        
        with self.train_writer.as_default():
            tf.summary.text('model_architecture', model_summary, step=0)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Registra hiperparámetros."""
        if not self.available:
            return
        
        # Convertir a formato compatible con TensorBoard
        hparam_dict = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
            else:
                hparam_dict[key] = str(value)
        
        with self.train_writer.as_default():
            tf.summary.text('hyperparameters', json.dumps(hparam_dict, indent=2), step=0)
    
    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """Registra histograma de valores."""
        if not self.available:
            return
        
        with self.train_writer.as_default():
            tf.summary.histogram(name, values, step=step)
    
    def flush(self):
        """Fuerza escritura de logs pendientes."""
        if not self.available:
            return
        
        self.train_writer.flush()
        self.val_writer.flush()
        self.eval_writer.flush()
    
    def close(self):
        """Cierra writers de TensorBoard."""
        if not self.available:
            return
        
        self.train_writer.close()
        self.val_writer.close()
        self.eval_writer.close()


class MetricsDashboard:
    """Dashboard personalizado para métricas en tiempo real."""
    
    def __init__(self, output_dir: str = "./logs/dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de salida
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.summary_file = self.output_dir / "summary.json"
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Buffer de métricas
        self.metrics_history = []
        
        logger.info(f"Dashboard initialized at {self.output_dir}")
    
    def update_metrics(self, metrics: TrainingMetrics):
        """Actualiza dashboard con nuevas métricas."""
        # Añadir al historial
        self.metrics_history.append(metrics.to_dict())
        
        # Escribir a archivo JSONL
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
        
        # Actualizar resumen
        self._update_summary()
        
        # Generar plots cada cierto tiempo
        if len(self.metrics_history) % 10 == 0:
            self._generate_plots()
    
    def _update_summary(self):
        """Actualiza archivo de resumen."""
        if not self.metrics_history:
            return
        
        latest = self.metrics_history[-1]
        
        # Calcular estadísticas
        train_losses = [m['train_loss'] for m in self.metrics_history if m.get('train_loss', 0) > 0]
        val_losses = [m['val_loss'] for m in self.metrics_history if m.get('val_loss', 0) > 0]
        
        summary = {
            'experiment_info': {
                'total_epochs': latest.get('epoch', 0),
                'total_batches_processed': len(self.metrics_history),
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'current_metrics': latest,
            'best_metrics': {
                'best_train_loss': min(train_losses) if train_losses else float('inf'),
                'best_val_loss': min(val_losses) if val_losses else float('inf'),
            },
            'training_progress': {
                'epochs_completed': latest.get('epoch', 0),
                'current_batch': latest.get('batch', 0),
                'total_batches': latest.get('total_batches', 0),
            }
        }
        
        # Añadir métricas de evaluación si están disponibles
        if 'evaluation' in latest:
            eval_data = latest['evaluation']
            summary['latest_evaluation'] = eval_data
            summary['best_metrics']['best_bleu'] = eval_data.get('bleu_score', 0)
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_plots(self):
        """Genera plots de métricas."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Backend sin GUI
            
            # Plot de loss
            self._plot_loss_curves()
            
            # Plot de métricas de evaluación
            self._plot_evaluation_metrics()
            
            # Plot de métricas del sistema
            self._plot_system_metrics()
            
        except ImportError:
            logger.warning("Matplotlib not available - skipping plot generation")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
    
    def _plot_loss_curves(self):
        """Genera plot de curvas de loss."""
        import matplotlib.pyplot as plt
        
        epochs = [m['epoch'] for m in self.metrics_history]
        train_losses = [m.get('train_loss', 0) for m in self.metrics_history]
        val_losses = [m.get('val_loss', 0) for m in self.metrics_history]
        
        plt.figure(figsize=(12, 6))
        
        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', alpha=0.7)
        if any(val_losses):
            plt.plot(epochs, val_losses, label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Perplexity
        plt.subplot(1, 2, 2)
        train_perp = [m.get('train_perplexity', 0) for m in self.metrics_history]
        val_perp = [m.get('val_perplexity', 0) for m in self.metrics_history]
        
        plt.plot(epochs, train_perp, label='Train Perplexity', alpha=0.7)
        if any(val_perp):
            plt.plot(epochs, val_perp, label='Val Perplexity', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_evaluation_metrics(self):
        """Genera plot de métricas de evaluación."""
        import matplotlib.pyplot as plt
        
        # Filtrar métricas que tienen evaluación
        eval_metrics = [m for m in self.metrics_history if 'evaluation' in m]
        
        if not eval_metrics:
            return
        
        epochs = [m['epoch'] for m in eval_metrics]
        bleu_scores = [m['evaluation'].get('bleu_score', 0) for m in eval_metrics]
        rouge1_scores = [m['evaluation'].get('rouge_1', 0) for m in eval_metrics]
        diversity_scores = [m['evaluation'].get('bigram_diversity', 0) for m in eval_metrics]
        
        plt.figure(figsize=(15, 5))
        
        # BLEU scores
        plt.subplot(1, 3, 1)
        plt.plot(epochs, bleu_scores, 'o-', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score')
        plt.title('BLEU Score Evolution')
        plt.grid(True, alpha=0.3)
        
        # ROUGE scores
        plt.subplot(1, 3, 2)
        plt.plot(epochs, rouge1_scores, 'o-', alpha=0.7, label='ROUGE-1')
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE Score')
        plt.title('ROUGE Score Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Diversidad
        plt.subplot(1, 3, 3)
        plt.plot(epochs, diversity_scores, 'o-', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Bigram Diversity')
        plt.title('Text Diversity Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'evaluation_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_system_metrics(self):
        """Genera plot de métricas del sistema."""
        import matplotlib.pyplot as plt
        
        epochs = [m['epoch'] for m in self.metrics_history]
        learning_rates = [m.get('learning_rate', 0) for m in self.metrics_history]
        gpu_memory = [m.get('gpu_memory_used', 0) for m in self.metrics_history]
        training_speed = [m.get('training_speed', 0) for m in self.metrics_history]
        
        plt.figure(figsize=(15, 5))
        
        # Learning rate
        plt.subplot(1, 3, 1)
        plt.plot(epochs, learning_rates, alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # GPU Memory
        plt.subplot(1, 3, 2)
        if any(gpu_memory):
            plt.plot(epochs, gpu_memory, alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('GPU Memory (MB)')
            plt.title('GPU Memory Usage')
            plt.grid(True, alpha=0.3)
        
        # Training Speed
        plt.subplot(1, 3, 3)
        if any(training_speed):
            plt.plot(epochs, training_speed, alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Tokens/sec')
            plt.title('Training Speed')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'system_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> str:
        """Genera reporte de entrenamiento."""
        if not self.metrics_history:
            return "No metrics available"
        
        latest = self.metrics_history[-1]
        
        # Calcular estadísticas
        train_losses = [m['train_loss'] for m in self.metrics_history if m.get('train_loss', 0) > 0]
        
        report = f"""
# Robo-Poet Training Report

## Experiment Overview
- **Epochs Completed:** {latest.get('epoch', 0)}
- **Total Batches Processed:** {len(self.metrics_history)}
- **Last Update:** {time.strftime('%Y-%m-%d %H:%M:%S')}

## Current Performance
- **Training Loss:** {latest.get('train_loss', 0):.4f}
- **Validation Loss:** {latest.get('val_loss', 0):.4f}
- **Learning Rate:** {latest.get('learning_rate', 0):.6f}

## Best Performance
- **Best Training Loss:** {min(train_losses) if train_losses else 'N/A'}
- **Current Perplexity:** {latest.get('train_perplexity', 0):.2f}

## System Resources
- **GPU Memory Used:** {latest.get('gpu_memory_used', 0):.1f} MB
- **Training Speed:** {latest.get('training_speed', 0):.0f} tokens/sec
"""
        
        # Añadir métricas de evaluación si están disponibles
        if 'evaluation' in latest:
            eval_data = latest['evaluation']
            report += f"""
## Latest Evaluation Results
- **BLEU Score:** {eval_data.get('bleu_score', 0):.4f}
- **ROUGE-1:** {eval_data.get('rouge_1', 0):.4f}
- **ROUGE-2:** {eval_data.get('rouge_2', 0):.4f}
- **ROUGE-L:** {eval_data.get('rouge_l', 0):.4f}
- **Bigram Diversity:** {eval_data.get('bigram_diversity', 0):.4f}
- **Repetition Ratio:** {eval_data.get('repetition_ratio', 0):.4f}
"""
        
        return report
    
    def export_metrics(self, format: str = 'csv') -> str:
        """Exporta métricas a diferentes formatos."""
        if format == 'csv':
            return self._export_to_csv()
        elif format == 'json':
            return self._export_to_json()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_csv(self) -> str:
        """Exporta métricas a CSV."""
        import csv
        
        csv_file = self.output_dir / 'metrics_export.csv'
        
        if not self.metrics_history:
            return str(csv_file)
        
        # Obtener todas las claves únicas
        all_keys = set()
        for metrics in self.metrics_history:
            all_keys.update(metrics.keys())
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(self.metrics_history)
        
        return str(csv_file)
    
    def _export_to_json(self) -> str:
        """Exporta métricas a JSON."""
        json_file = self.output_dir / 'metrics_export.json'
        
        with open(json_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        return str(json_file)