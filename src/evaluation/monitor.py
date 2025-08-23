"""
Monitor de entrenamiento en tiempo real.

Implementa evaluación continua durante el entrenamiento y logging
de métricas para seguimiento del progreso del modelo.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from .metrics import MetricCalculator, EvaluationResults


logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento en tiempo real."""
    
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    
    # Métricas de entrenamiento
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    train_perplexity: float = 0.0
    
    # Métricas de validación
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_perplexity: float = 0.0
    
    # Métricas de evaluación
    evaluation_results: Optional[EvaluationResults] = None
    
    # Métricas del sistema
    learning_rate: float = 0.0
    gpu_memory_used: float = 0.0
    training_speed: float = 0.0  # tokens/sec
    
    # Timestamps
    timestamp: float = field(default_factory=time.time)
    epoch_start_time: float = 0.0
    batch_start_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para logging."""
        data = {
            'epoch': self.epoch,
            'batch': self.batch,
            'total_batches': self.total_batches,
            'train_loss': self.train_loss,
            'train_accuracy': self.train_accuracy,
            'train_perplexity': self.train_perplexity,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy,
            'val_perplexity': self.val_perplexity,
            'learning_rate': self.learning_rate,
            'gpu_memory_used': self.gpu_memory_used,
            'training_speed': self.training_speed,
            'timestamp': self.timestamp
        }
        
        if self.evaluation_results:
            data['evaluation'] = self.evaluation_results.to_dict()
        
        return data


class MetricLogger:
    """Logger de métricas con diferentes backends."""
    
    def __init__(self, log_file: Optional[str] = None, buffer_size: int = 100):
        self.log_file = log_file
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.callbacks: List[Callable[[TrainingMetrics], None]] = []
        
        # Configurar archivo de log si se especifica
        if self.log_file:
            self._setup_file_logger()
    
    def _setup_file_logger(self):
        """Configura logging a archivo."""
        import json
        self.file_handler = open(self.log_file, 'a')
    
    def add_callback(self, callback: Callable[[TrainingMetrics], None]):
        """Añade callback que se ejecuta cuando se registran métricas."""
        self.callbacks.append(callback)
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Registra métricas."""
        # Añadir al buffer
        self.metrics_buffer.append(metrics)
        
        # Logging a archivo
        if self.log_file and hasattr(self, 'file_handler'):
            import json
            self.file_handler.write(json.dumps(metrics.to_dict()) + '\n')
            self.file_handler.flush()
        
        # Ejecutar callbacks
        for callback in self.callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")
        
        # Log a consola
        logger.info(
            f"Epoch {metrics.epoch:03d} | "
            f"Batch {metrics.batch:04d}/{metrics.total_batches:04d} | "
            f"Loss: {metrics.train_loss:.4f} | "
            f"Val Loss: {metrics.val_loss:.4f} | "
            f"LR: {metrics.learning_rate:.6f}"
        )
    
    def get_recent_metrics(self, n: int = 10) -> List[TrainingMetrics]:
        """Obtiene las últimas n métricas."""
        return list(self.metrics_buffer)[-n:]
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Obtiene historial de una métrica específica."""
        history = []
        for metrics in self.metrics_buffer:
            if hasattr(metrics, metric_name):
                history.append(getattr(metrics, metric_name))
        return history
    
    def close(self):
        """Cierra el logger."""
        if hasattr(self, 'file_handler'):
            self.file_handler.close()


class RealTimeEvaluator:
    """Evaluador en tiempo real durante el entrenamiento."""
    
    def __init__(
        self,
        metric_calculator: MetricCalculator,
        validation_data: Optional[List[Dict[str, Any]]] = None,
        evaluation_frequency: int = 100,  # cada N batches
        max_samples: int = 50  # máximo samples para evaluación rápida
    ):
        self.metric_calculator = metric_calculator
        self.validation_data = validation_data or []
        self.evaluation_frequency = evaluation_frequency
        self.max_samples = max_samples
        
        # Estado interno
        self.batch_count = 0
        self.last_evaluation_time = 0
        
    def should_evaluate(self, batch_num: int, force: bool = False) -> bool:
        """Determina si se debe ejecutar evaluación."""
        if force:
            return True
        
        # Evaluar según frecuencia
        if batch_num % self.evaluation_frequency == 0:
            return True
        
        # Evaluar si ha pasado mucho tiempo
        current_time = time.time()
        if current_time - self.last_evaluation_time > 300:  # 5 minutos
            return True
        
        return False
    
    def evaluate_generated_samples(
        self,
        references: List[str],
        candidates: List[str],
        probabilities: Optional[List[np.ndarray]] = None
    ) -> EvaluationResults:
        """Evalúa samples generados en tiempo real."""
        
        # Limitar número de samples para evaluación rápida
        if len(candidates) > self.max_samples:
            indices = np.random.choice(len(candidates), self.max_samples, replace=False)
            references = [references[i] for i in indices]
            candidates = [candidates[i] for i in indices]
            if probabilities:
                probabilities = [probabilities[i] for i in indices]
        
        start_time = time.time()
        
        try:
            results = self.metric_calculator.evaluate_generation(
                references, candidates, probabilities
            )
            
            evaluation_time = time.time() - start_time
            results.evaluation_time = evaluation_time
            self.last_evaluation_time = time.time()
            
            return results
            
        except Exception as e:
            logger.error(f"Real-time evaluation failed: {e}")
            # Devolver resultados vacíos en caso de error
            return EvaluationResults(evaluation_time=time.time() - start_time)
    
    def quick_sample_evaluation(self, model_output: Dict[str, Any]) -> Dict[str, float]:
        """Evaluación rápida de una muestra."""
        try:
            reference = model_output.get('reference', '')
            candidate = model_output.get('generated', '')
            
            if not reference or not candidate:
                return {}
            
            return self.metric_calculator.quick_evaluate(reference, candidate)
            
        except Exception as e:
            logger.warning(f"Quick evaluation failed: {e}")
            return {}


class TrainingMonitor:
    """Monitor principal de entrenamiento con evaluación continua."""
    
    def __init__(
        self,
        metric_logger: MetricLogger,
        real_time_evaluator: Optional[RealTimeEvaluator] = None,
        dashboard_update_frequency: int = 10  # cada N batches
    ):
        self.metric_logger = metric_logger
        self.real_time_evaluator = real_time_evaluator
        self.dashboard_update_frequency = dashboard_update_frequency
        
        # Estado del entrenamiento
        self.current_epoch = 0
        self.current_batch = 0
        self.epoch_start_time = 0
        self.training_start_time = time.time()
        
        # Buffers para promedios móviles
        self.loss_buffer = deque(maxlen=100)
        self.accuracy_buffer = deque(maxlen=100)
        
        # Callbacks para eventos
        self.epoch_start_callbacks = []
        self.epoch_end_callbacks = []
        self.batch_end_callbacks = []
        
    def start_epoch(self, epoch: int, total_batches: int):
        """Inicia una nueva época."""
        self.current_epoch = epoch
        self.current_batch = 0
        self.epoch_start_time = time.time()
        
        logger.info(f"Started epoch {epoch} with {total_batches} batches")
        
        # Ejecutar callbacks
        for callback in self.epoch_start_callbacks:
            try:
                callback(epoch, total_batches)
            except Exception as e:
                logger.warning(f"Epoch start callback failed: {e}")
    
    def end_epoch(self, epoch_metrics: Dict[str, float]):
        """Termina una época."""
        epoch_duration = time.time() - self.epoch_start_time
        
        logger.info(
            f"Completed epoch {self.current_epoch} in {epoch_duration:.2f}s | "
            f"Avg Loss: {epoch_metrics.get('avg_loss', 0):.4f}"
        )
        
        # Ejecutar callbacks
        for callback in self.epoch_end_callbacks:
            try:
                callback(self.current_epoch, epoch_metrics, epoch_duration)
            except Exception as e:
                logger.warning(f"Epoch end callback failed: {e}")
    
    def update_batch_metrics(
        self,
        batch_num: int,
        total_batches: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        generated_samples: Optional[Dict[str, Any]] = None
    ):
        """Actualiza métricas después de procesar un batch."""
        
        self.current_batch = batch_num
        
        # Crear objeto de métricas
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            batch=batch_num,
            total_batches=total_batches,
            timestamp=time.time()
        )
        
        # Métricas de entrenamiento
        metrics.train_loss = train_metrics.get('loss', 0.0)
        metrics.train_accuracy = train_metrics.get('accuracy', 0.0)
        metrics.learning_rate = train_metrics.get('learning_rate', 0.0)
        
        # Calcular perplejidad de entrenamiento
        if metrics.train_loss > 0:
            metrics.train_perplexity = np.exp(metrics.train_loss)
        
        # Métricas de validación
        if val_metrics:
            metrics.val_loss = val_metrics.get('loss', 0.0)
            metrics.val_accuracy = val_metrics.get('accuracy', 0.0)
            if metrics.val_loss > 0:
                metrics.val_perplexity = np.exp(metrics.val_loss)
        
        # Actualizar buffers para promedios móviles
        self.loss_buffer.append(metrics.train_loss)
        self.accuracy_buffer.append(metrics.train_accuracy)
        
        # Calcular velocidad de entrenamiento
        if hasattr(self, 'last_batch_time'):
            batch_duration = time.time() - self.last_batch_time
            if batch_duration > 0:
                # Estimar tokens por segundo (asumiendo batch_size promedio)
                estimated_tokens = train_metrics.get('tokens_processed', 1000)
                metrics.training_speed = estimated_tokens / batch_duration
        
        self.last_batch_time = time.time()
        
        # Obtener uso de memoria GPU
        metrics.gpu_memory_used = self._get_gpu_memory_usage()
        
        # Evaluación en tiempo real si está disponible
        if (self.real_time_evaluator and generated_samples and 
            self.real_time_evaluator.should_evaluate(batch_num)):
            
            try:
                eval_results = self._run_realtime_evaluation(generated_samples)
                metrics.evaluation_results = eval_results
            except Exception as e:
                logger.warning(f"Real-time evaluation failed: {e}")
        
        # Registrar métricas
        self.metric_logger.log_metrics(metrics)
        
        # Ejecutar callbacks
        for callback in self.batch_end_callbacks:
            try:
                callback(batch_num, metrics)
            except Exception as e:
                logger.warning(f"Batch end callback failed: {e}")
    
    def _run_realtime_evaluation(self, generated_samples: Dict[str, Any]) -> EvaluationResults:
        """Ejecuta evaluación en tiempo real."""
        references = generated_samples.get('references', [])
        candidates = generated_samples.get('generated', [])
        probabilities = generated_samples.get('probabilities', None)
        
        return self.real_time_evaluator.evaluate_generated_samples(
            references, candidates, probabilities
        )
    
    def _get_gpu_memory_usage(self) -> float:
        """Obtiene uso actual de memoria GPU."""
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                # TensorFlow no proporciona API directa para memoria usada
                # Esto es una aproximación
                return 0.0
        except:
            pass
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return memory_info.used / 1024**2  # MB
        except:
            return 0.0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del entrenamiento."""
        total_duration = time.time() - self.training_start_time
        
        return {
            'current_epoch': self.current_epoch,
            'current_batch': self.current_batch,
            'total_duration': total_duration,
            'avg_loss': np.mean(self.loss_buffer) if self.loss_buffer else 0,
            'avg_accuracy': np.mean(self.accuracy_buffer) if self.accuracy_buffer else 0,
            'recent_metrics': self.metric_logger.get_recent_metrics(10)
        }
    
    def add_epoch_start_callback(self, callback: Callable):
        """Añade callback para inicio de época."""
        self.epoch_start_callbacks.append(callback)
    
    def add_epoch_end_callback(self, callback: Callable):
        """Añade callback para fin de época."""
        self.epoch_end_callbacks.append(callback)
    
    def add_batch_end_callback(self, callback: Callable):
        """Añade callback para fin de batch."""
        self.batch_end_callbacks.append(callback)


class ProgressTracker:
    """Tracker de progreso con estimaciones de tiempo."""
    
    def __init__(self, total_epochs: int, batches_per_epoch: int):
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.total_batches = total_epochs * batches_per_epoch
        
        self.start_time = time.time()
        self.completed_batches = 0
        self.batch_times = deque(maxlen=100)
    
    def update(self, epoch: int, batch: int):
        """Actualiza progreso."""
        current_time = time.time()
        
        # Calcular batches completados
        self.completed_batches = epoch * self.batches_per_epoch + batch
        
        # Registrar tiempo de batch
        if hasattr(self, 'last_update_time'):
            batch_time = current_time - self.last_update_time
            self.batch_times.append(batch_time)
        
        self.last_update_time = current_time
    
    def get_eta(self) -> Dict[str, Any]:
        """Calcula tiempo estimado de finalización."""
        if self.completed_batches == 0:
            return {'eta_seconds': 0, 'eta_formatted': 'Unknown', 'progress': 0.0}
        
        # Progreso
        progress = self.completed_batches / self.total_batches
        
        # Tiempo promedio por batch
        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times)
            remaining_batches = self.total_batches - self.completed_batches
            eta_seconds = remaining_batches * avg_batch_time
        else:
            elapsed = time.time() - self.start_time
            eta_seconds = (elapsed / progress - elapsed) if progress > 0 else 0
        
        # Formatear tiempo
        eta_formatted = self._format_time(eta_seconds)
        
        return {
            'eta_seconds': eta_seconds,
            'eta_formatted': eta_formatted,
            'progress': progress,
            'completed_batches': self.completed_batches,
            'total_batches': self.total_batches
        }
    
    def _format_time(self, seconds: float) -> str:
        """Formatea tiempo en formato legible."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"