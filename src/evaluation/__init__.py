"""
Sistema de Evaluación Continua para Robo-Poet.

Módulo que implementa métricas automáticas, monitoreo en tiempo real
y dashboard de seguimiento para entrenamiento de modelos.
"""

from .metrics import (
    BLEUMetric, PerplexityMetric, NGramDiversityMetric, ROUGEMetric,
    MetricCalculator, EvaluationResults
)
from .monitor import (
    TrainingMonitor, RealTimeEvaluator, MetricLogger
)
from .dashboard import (
    TensorBoardDashboard, MetricsDashboard
)
from .early_stopping import (
    MultiMetricEarlyStopping, EarlyStoppingCriteria
)

__all__ = [
    # Métricas
    'BLEUMetric', 'PerplexityMetric', 'NGramDiversityMetric', 'ROUGEMetric',
    'MetricCalculator', 'EvaluationResults',
    
    # Monitoreo
    'TrainingMonitor', 'RealTimeEvaluator', 'MetricLogger',
    
    # Dashboard
    'TensorBoardDashboard', 'MetricsDashboard',
    
    # Early Stopping
    'MultiMetricEarlyStopping', 'EarlyStoppingCriteria'
]