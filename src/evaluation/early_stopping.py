"""
Sistema de Early Stopping basado en múltiples métricas.

Implementa estrategias avanzadas de early stopping que consideran
múltiples métricas para determinar cuándo detener el entrenamiento.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .metrics import EvaluationResults
from .monitor import TrainingMetrics


logger = logging.getLogger(__name__)


class MetricDirection(Enum):
    """Dirección de optimización de una métrica."""
    MINIMIZE = "minimize"  # Menor es mejor (ej: loss)
    MAXIMIZE = "maximize"  # Mayor es mejor (ej: BLEU)


@dataclass
class MetricCriterion:
    """Criterio para una métrica específica."""
    
    name: str  # Nombre de la métrica
    direction: MetricDirection  # Dirección de optimización
    patience: int  # Paciencia específica para esta métrica
    min_delta: float = 0.0  # Cambio mínimo considerado como mejora
    weight: float = 1.0  # Peso en la decisión combinada
    
    # Estado interno
    best_value: Optional[float] = None
    epochs_without_improvement: int = 0
    history: List[float] = field(default_factory=list)
    
    def update(self, value: float) -> bool:
        """
        Actualiza la métrica y retorna True si hay mejora.
        
        Args:
            value: Nuevo valor de la métrica
            
        Returns:
            True si hay mejora, False en caso contrario
        """
        self.history.append(value)
        
        # Primera evaluación
        if self.best_value is None:
            self.best_value = value
            self.epochs_without_improvement = 0
            return True
        
        # Verificar mejora
        if self.direction == MetricDirection.MINIMIZE:
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def should_stop(self) -> bool:
        """Determina si esta métrica indica que se debe parar."""
        return self.epochs_without_improvement >= self.patience
    
    def get_improvement_ratio(self) -> float:
        """Calcula ratio de mejora reciente."""
        if len(self.history) < 2:
            return 0.0
        
        recent_values = self.history[-5:]  # Últimos 5 valores
        if len(recent_values) < 2:
            return 0.0
        
        if self.direction == MetricDirection.MINIMIZE:
            return (recent_values[0] - recent_values[-1]) / max(recent_values[0], 1e-8)
        else:
            return (recent_values[-1] - recent_values[0]) / max(recent_values[0], 1e-8)


@dataclass
class EarlyStoppingCriteria:
    """Criterios de early stopping."""
    
    # Configuración básica
    patience: int = 15  # Paciencia global
    min_epochs: int = 10  # Mínimo de épocas antes de considerar parar
    max_epochs: int = 1000  # Máximo de épocas
    
    # Criterios por métrica
    metric_criteria: List[MetricCriterion] = field(default_factory=list)
    
    # Estrategia de combinación
    combination_strategy: str = "any"  # "any", "all", "weighted"
    min_metrics_for_decision: int = 1  # Mínimo de métricas para tomar decisión
    
    # Criterios adicionales
    loss_plateau_threshold: float = 1e-4  # Umbral de plateau en loss
    divergence_threshold: float = 10.0  # Umbral de divergencia
    
    def add_metric_criterion(self, criterion: MetricCriterion):
        """Añade criterio para una métrica."""
        self.metric_criteria.append(criterion)
    
    @classmethod
    def create_default(cls) -> 'EarlyStoppingCriteria':
        """Crea criterios por defecto."""
        criteria = cls(patience=15, min_epochs=10)
        
        # Criterio principal: validation loss
        criteria.add_metric_criterion(MetricCriterion(
            name="val_loss",
            direction=MetricDirection.MINIMIZE,
            patience=15,
            min_delta=0.001,
            weight=2.0
        ))
        
        # Criterio secundario: BLEU score
        criteria.add_metric_criterion(MetricCriterion(
            name="bleu_score",
            direction=MetricDirection.MAXIMIZE,
            patience=20,
            min_delta=0.01,
            weight=1.0
        ))
        
        return criteria
    
    @classmethod
    def create_aggressive(cls) -> 'EarlyStoppingCriteria':
        """Crea criterios agresivos (para menos tiempo)."""
        criteria = cls(patience=10, min_epochs=5)
        
        criteria.add_metric_criterion(MetricCriterion(
            name="val_loss",
            direction=MetricDirection.MINIMIZE,
            patience=10,
            min_delta=0.01,
            weight=2.0
        ))
        
        return criteria
    
    @classmethod
    def create_conservative(cls) -> 'EarlyStoppingCriteria':
        """Crea criterios conservadores (para más entrenamiento)."""
        criteria = cls(patience=25, min_epochs=20)
        
        criteria.add_metric_criterion(MetricCriterion(
            name="val_loss",
            direction=MetricDirection.MINIMIZE,
            patience=25,
            min_delta=0.0001,
            weight=1.0
        ))
        
        criteria.add_metric_criterion(MetricCriterion(
            name="bleu_score",
            direction=MetricDirection.MAXIMIZE,
            patience=30,
            min_delta=0.005,
            weight=1.0
        ))
        
        return criteria


class MultiMetricEarlyStopping:
    """Sistema de early stopping basado en múltiples métricas."""
    
    def __init__(self, criteria: EarlyStoppingCriteria):
        self.criteria = criteria
        self.current_epoch = 0
        self.best_epoch = 0
        self.should_stop_flag = False
        self.stop_reason = ""
        
        # Callbacks
        self.improvement_callbacks: List[Callable] = []
        self.plateau_callbacks: List[Callable] = []
        self.stop_callbacks: List[Callable] = []
        
        logger.info(f"Early stopping initialized with {len(criteria.metric_criteria)} criteria")
    
    def update(self, metrics: TrainingMetrics, evaluation_results: Optional[EvaluationResults] = None) -> bool:
        """
        Actualiza early stopping con nuevas métricas.
        
        Args:
            metrics: Métricas de entrenamiento
            evaluation_results: Resultados de evaluación (opcional)
            
        Returns:
            True si se debe detener el entrenamiento
        """
        self.current_epoch = metrics.epoch
        
        # No considerar early stopping antes del mínimo de épocas
        if self.current_epoch < self.criteria.min_epochs:
            return False
        
        # Parar si se alcanza el máximo de épocas
        if self.current_epoch >= self.criteria.max_epochs:
            self.should_stop_flag = True
            self.stop_reason = f"Maximum epochs reached ({self.criteria.max_epochs})"
            self._trigger_stop_callbacks()
            return True
        
        # Verificar divergencia
        if self._check_divergence(metrics):
            self.should_stop_flag = True
            self.stop_reason = "Training diverged"
            self._trigger_stop_callbacks()
            return True
        
        # Actualizar criterios de métricas
        metrics_values = self._extract_metric_values(metrics, evaluation_results)
        improvements = []
        
        for criterion in self.criteria.metric_criteria:
            if criterion.name in metrics_values:
                value = metrics_values[criterion.name]
                improved = criterion.update(value)
                improvements.append(improved)
                
                logger.debug(
                    f"Metric {criterion.name}: {value:.4f} "
                    f"(best: {criterion.best_value:.4f}, "
                    f"no improvement: {criterion.epochs_without_improvement})"
                )
        
        # Verificar si hay mejora global
        any_improvement = any(improvements)
        if any_improvement:
            self.best_epoch = self.current_epoch
            self._trigger_improvement_callbacks()
        
        # Decidir si parar basado en estrategia de combinación
        should_stop = self._should_stop_combined()
        
        if should_stop:
            self.should_stop_flag = True
            self.stop_reason = self._generate_stop_reason()
            self._trigger_stop_callbacks()
            return True
        
        # Verificar plateau
        if self._check_plateau():
            self._trigger_plateau_callbacks()
        
        return False
    
    def _extract_metric_values(
        self, 
        metrics: TrainingMetrics, 
        evaluation_results: Optional[EvaluationResults]
    ) -> Dict[str, float]:
        """Extrae valores de métricas relevantes."""
        values = {}
        
        # Métricas de entrenamiento
        values['train_loss'] = metrics.train_loss
        values['val_loss'] = metrics.val_loss
        values['train_accuracy'] = metrics.train_accuracy
        values['val_accuracy'] = metrics.val_accuracy
        values['train_perplexity'] = metrics.train_perplexity
        values['val_perplexity'] = metrics.val_perplexity
        
        # Métricas de evaluación
        if evaluation_results:
            values['bleu_score'] = evaluation_results.bleu_score
            values['rouge_1'] = evaluation_results.rouge_1
            values['rouge_2'] = evaluation_results.rouge_2
            values['rouge_l'] = evaluation_results.rouge_l
            values['perplexity'] = evaluation_results.perplexity
            values['unigram_diversity'] = evaluation_results.unigram_diversity
            values['bigram_diversity'] = evaluation_results.bigram_diversity
            values['trigram_diversity'] = evaluation_results.trigram_diversity
            values['summary_score'] = evaluation_results.get_summary_score()
        
        return values
    
    def _should_stop_combined(self) -> bool:
        """Determina si parar basado en estrategia de combinación."""
        if not self.criteria.metric_criteria:
            return False
        
        # Filtrar criterios que tienen suficientes datos
        active_criteria = [
            c for c in self.criteria.metric_criteria 
            if c.best_value is not None
        ]
        
        if len(active_criteria) < self.criteria.min_metrics_for_decision:
            return False
        
        if self.criteria.combination_strategy == "any":
            # Parar si cualquier métrica indica parar
            return any(c.should_stop() for c in active_criteria)
        
        elif self.criteria.combination_strategy == "all":
            # Parar solo si todas las métricas indican parar
            return all(c.should_stop() for c in active_criteria)
        
        elif self.criteria.combination_strategy == "weighted":
            # Parar basado en voto ponderado
            total_weight = sum(c.weight for c in active_criteria)
            stop_weight = sum(c.weight for c in active_criteria if c.should_stop())
            
            return stop_weight / total_weight > 0.5
        
        else:
            logger.warning(f"Unknown combination strategy: {self.criteria.combination_strategy}")
            return False
    
    def _check_divergence(self, metrics: TrainingMetrics) -> bool:
        """Verifica si el entrenamiento está divergiendo."""
        # Verificar si la loss se vuelve muy grande
        if metrics.train_loss > self.criteria.divergence_threshold:
            return True
        
        # Verificar si hay NaN o infinito
        if np.isnan(metrics.train_loss) or np.isinf(metrics.train_loss):
            return True
        
        return False
    
    def _check_plateau(self) -> bool:
        """Verifica si hay plateau en las métricas."""
        for criterion in self.criteria.metric_criteria:
            if len(criterion.history) >= 5:
                recent_improvement = criterion.get_improvement_ratio()
                if abs(recent_improvement) < self.criteria.loss_plateau_threshold:
                    return True
        
        return False
    
    def _generate_stop_reason(self) -> str:
        """Genera razón detallada para el stop."""
        reasons = []
        
        for criterion in self.criteria.metric_criteria:
            if criterion.should_stop():
                reasons.append(
                    f"{criterion.name}: {criterion.epochs_without_improvement} epochs "
                    f"without improvement (patience: {criterion.patience})"
                )
        
        if reasons:
            return "Early stopping triggered: " + "; ".join(reasons)
        else:
            return "Early stopping triggered by combined criteria"
    
    def add_improvement_callback(self, callback: Callable):
        """Añade callback para cuando hay mejora."""
        self.improvement_callbacks.append(callback)
    
    def add_plateau_callback(self, callback: Callable):
        """Añade callback para cuando hay plateau."""
        self.plateau_callbacks.append(callback)
    
    def add_stop_callback(self, callback: Callable):
        """Añade callback para cuando se detiene."""
        self.stop_callbacks.append(callback)
    
    def _trigger_improvement_callbacks(self):
        """Ejecuta callbacks de mejora."""
        for callback in self.improvement_callbacks:
            try:
                callback(self.current_epoch, self.best_epoch)
            except Exception as e:
                logger.warning(f"Improvement callback failed: {e}")
    
    def _trigger_plateau_callbacks(self):
        """Ejecuta callbacks de plateau."""
        for callback in self.plateau_callbacks:
            try:
                callback(self.current_epoch)
            except Exception as e:
                logger.warning(f"Plateau callback failed: {e}")
    
    def _trigger_stop_callbacks(self):
        """Ejecuta callbacks de stop."""
        for callback in self.stop_callbacks:
            try:
                callback(self.current_epoch, self.stop_reason)
            except Exception as e:
                logger.warning(f"Stop callback failed: {e}")
    
    def should_stop(self) -> bool:
        """Retorna si se debe detener el entrenamiento."""
        return self.should_stop_flag
    
    def get_best_epoch(self) -> int:
        """Retorna la mejor época encontrada."""
        return self.best_epoch
    
    def get_stop_reason(self) -> str:
        """Retorna la razón del stop."""
        return self.stop_reason
    
    def get_status_report(self) -> Dict[str, Any]:
        """Genera reporte de estado."""
        return {
            'current_epoch': self.current_epoch,
            'best_epoch': self.best_epoch,
            'should_stop': self.should_stop_flag,
            'stop_reason': self.stop_reason,
            'criteria_status': [
                {
                    'name': c.name,
                    'best_value': c.best_value,
                    'epochs_without_improvement': c.epochs_without_improvement,
                    'patience': c.patience,
                    'should_stop': c.should_stop()
                }
                for c in self.criteria.metric_criteria
            ]
        }
    
    def reset(self):
        """Reinicia el estado del early stopping."""
        self.current_epoch = 0
        self.best_epoch = 0
        self.should_stop_flag = False
        self.stop_reason = ""
        
        for criterion in self.criteria.metric_criteria:
            criterion.best_value = None
            criterion.epochs_without_improvement = 0
            criterion.history.clear()


class EarlyStoppingCallback:
    """Callback para TensorFlow/Keras compatible con early stopping."""
    
    def __init__(self, early_stopping: MultiMetricEarlyStopping):
        self.early_stopping = early_stopping
        self.model = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Callback al final de cada época."""
        if logs is None:
            logs = {}
        
        # Crear métricas de entrenamiento básicas
        from .monitor import TrainingMetrics
        
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=logs.get('loss', 0.0),
            val_loss=logs.get('val_loss', 0.0),
            train_accuracy=logs.get('accuracy', 0.0),
            val_accuracy=logs.get('val_accuracy', 0.0)
        )
        
        # Actualizar early stopping
        should_stop = self.early_stopping.update(metrics)
        
        if should_stop and self.model is not None:
            logger.info(f"Early stopping: {self.early_stopping.get_stop_reason()}")
            self.model.stop_training = True