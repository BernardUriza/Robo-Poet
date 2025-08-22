"""
Query objects for training-related read operations.

These queries implement the Query side of CQRS pattern for read-only operations.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.domain.entities.generation_model import ModelStatus


@dataclass
class GetModelByIdQuery:
    """Query to get a specific model by ID."""
    model_id: str


@dataclass
class GetCorpusByIdQuery:
    """Query to get a specific corpus by ID."""
    corpus_id: str


@dataclass
class ListModelsQuery:
    """Query to list models with optional filtering."""
    status: Optional[ModelStatus] = None
    corpus_id: Optional[str] = None
    tag: Optional[str] = None
    limit: Optional[int] = None
    offset: int = 0
    sort_by: str = "created_at"
    sort_order: str = "desc"  # "asc" or "desc"


@dataclass
class ListCorpusesQuery:
    """Query to list corpuses with optional filtering."""
    language: Optional[str] = None
    tag: Optional[str] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    limit: Optional[int] = None
    offset: int = 0
    sort_by: str = "created_at"
    sort_order: str = "desc"


@dataclass
class GetTrainingMetricsQuery:
    """Query to get training metrics for a model."""
    model_id: str
    include_history: bool = False


@dataclass
class GetModelsByCorpusQuery:
    """Query to get all models trained on a specific corpus."""
    corpus_id: str
    status: Optional[ModelStatus] = None


@dataclass
class SearchModelsQuery:
    """Query to search models by name or description."""
    search_term: str
    search_in_description: bool = True
    status: Optional[ModelStatus] = None
    limit: Optional[int] = None


@dataclass
class SearchCorpusesQuery:
    """Query to search corpuses by name or content."""
    search_term: str
    search_in_content: bool = False
    language: Optional[str] = None
    limit: Optional[int] = None


@dataclass
class GetModelStatisticsQuery:
    """Query to get aggregate statistics about models."""
    include_by_status: bool = True
    include_by_corpus: bool = True
    include_training_times: bool = True


@dataclass
class GetCorpusStatisticsQuery:
    """Query to get aggregate statistics about corpuses."""
    include_by_language: bool = True
    include_size_distribution: bool = True
    include_usage_stats: bool = True


@dataclass
class GetRecentActivityQuery:
    """Query to get recent training and generation activity."""
    days: int = 7
    include_events: bool = True
    include_models: bool = True
    include_generations: bool = True
    limit: Optional[int] = 100


@dataclass
class GetModelComparisonQuery:
    """Query to compare metrics between multiple models."""
    model_ids: List[str]
    include_config: bool = True
    include_metrics: bool = True
    include_performance: bool = True