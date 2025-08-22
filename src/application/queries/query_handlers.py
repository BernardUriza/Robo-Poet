"""
Query handlers for CQRS pattern.

These handlers execute read-only queries and return formatted data.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from src.domain.entities.generation_model import ModelStatus
from src.domain.repositories import UnitOfWork
from src.application.queries.training_queries import (
    GetModelByIdQuery, GetCorpusByIdQuery, ListModelsQuery, ListCorpusesQuery,
    GetTrainingMetricsQuery, GetModelsByCorpusQuery, SearchModelsQuery,
    SearchCorpusesQuery, GetModelStatisticsQuery, GetCorpusStatisticsQuery,
    GetRecentActivityQuery, GetModelComparisonQuery
)


class TrainingQueryHandler:
    """Handler for training-related queries."""
    
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
    
    def handle_get_model_by_id(self, query: GetModelByIdQuery) -> Optional[Dict[str, Any]]:
        """Handle GetModelByIdQuery."""
        with self.uow:
            model = self.uow.model_repository.get_by_id(query.model_id)
            if not model:
                return None
            
            corpus = self.uow.corpus_repository.get_by_id(model.corpus_id)
            
            return {
                'id': model.id,
                'name': model.name,
                'status': model.status.value,
                'description': model.description,
                'config': model.config.to_dict(),
                'created_at': model.created_at.isoformat(),
                'training_started_at': model.training_started_at.isoformat() if model.training_started_at else None,
                'training_completed_at': model.training_completed_at.isoformat() if model.training_completed_at else None,
                'training_metrics': model.training_metrics.to_dict() if model.training_metrics else None,
                'error_message': model.error_message,
                'corpus': {
                    'id': corpus.id if corpus else None,
                    'name': corpus.name if corpus else None,
                    'language': corpus.language if corpus else None
                },
                'tags': model.tags
            }
    
    def handle_get_corpus_by_id(self, query: GetCorpusByIdQuery) -> Optional[Dict[str, Any]]:
        """Handle GetCorpusByIdQuery."""
        with self.uow:
            corpus = self.uow.corpus_repository.get_by_id(query.corpus_id)
            if not corpus:
                return None
            
            # Get models using this corpus
            models = self.uow.model_repository.get_by_corpus_id(query.corpus_id)
            
            return {
                'id': corpus.id,
                'name': corpus.name,
                'content_size': len(corpus.content),
                'content_hash': corpus.content_hash,
                'language': corpus.language,
                'source_path': corpus.source_path,
                'created_at': corpus.created_at.isoformat(),
                'updated_at': corpus.updated_at.isoformat(),
                'tags': corpus.tags,
                'preprocessing_status': corpus.preprocessing_status,
                'models_count': len(models),
                'models': [
                    {
                        'id': model.id,
                        'name': model.name,
                        'status': model.status.value,
                        'created_at': model.created_at.isoformat()
                    }
                    for model in models
                ]
            }
    
    def handle_list_models(self, query: ListModelsQuery) -> Dict[str, Any]:
        """Handle ListModelsQuery."""
        with self.uow:
            # Apply filters
            models = self.uow.model_repository.get_all()
            
            if query.status:
                models = [m for m in models if m.status == query.status]
            
            if query.corpus_id:
                models = [m for m in models if m.corpus_id == query.corpus_id]
            
            if query.tag:
                models = [m for m in models if query.tag in m.tags]
            
            # Sort
            if query.sort_by == "created_at":
                models.sort(key=lambda m: m.created_at, reverse=(query.sort_order == "desc"))
            elif query.sort_by == "name":
                models.sort(key=lambda m: m.name.lower(), reverse=(query.sort_order == "desc"))
            
            # Paginate
            total_count = len(models)
            if query.offset:
                models = models[query.offset:]
            if query.limit:
                models = models[:query.limit]
            
            return {
                'models': [
                    {
                        'id': model.id,
                        'name': model.name,
                        'status': model.status.value,
                        'corpus_id': model.corpus_id,
                        'created_at': model.created_at.isoformat(),
                        'description': model.description,
                        'tags': model.tags,
                        'final_loss': model.training_metrics.final_loss if model.training_metrics else None
                    }
                    for model in models
                ],
                'total_count': total_count,
                'offset': query.offset,
                'limit': query.limit
            }
    
    def handle_list_corpuses(self, query: ListCorpusesQuery) -> Dict[str, Any]:
        """Handle ListCorpusesQuery."""
        with self.uow:
            corpuses = self.uow.corpus_repository.get_all()
            
            # Apply filters
            if query.language:
                corpuses = [c for c in corpuses if c.language == query.language]
            
            if query.tag:
                corpuses = [c for c in corpuses if query.tag in c.tags]
            
            if query.min_size:
                corpuses = [c for c in corpuses if len(c.content) >= query.min_size]
            
            if query.max_size:
                corpuses = [c for c in corpuses if len(c.content) <= query.max_size]
            
            # Sort
            if query.sort_by == "created_at":
                corpuses.sort(key=lambda c: c.created_at, reverse=(query.sort_order == "desc"))
            elif query.sort_by == "name":
                corpuses.sort(key=lambda c: c.name.lower(), reverse=(query.sort_order == "desc"))
            elif query.sort_by == "size":
                corpuses.sort(key=lambda c: len(c.content), reverse=(query.sort_order == "desc"))
            
            # Paginate
            total_count = len(corpuses)
            if query.offset:
                corpuses = corpuses[query.offset:]
            if query.limit:
                corpuses = corpuses[:query.limit]
            
            return {
                'corpuses': [
                    {
                        'id': corpus.id,
                        'name': corpus.name,
                        'size': len(corpus.content),
                        'language': corpus.language,
                        'created_at': corpus.created_at.isoformat(),
                        'tags': corpus.tags,
                        'source_path': corpus.source_path
                    }
                    for corpus in corpuses
                ],
                'total_count': total_count,
                'offset': query.offset,
                'limit': query.limit
            }
    
    def handle_get_training_metrics(self, query: GetTrainingMetricsQuery) -> Optional[Dict[str, Any]]:
        """Handle GetTrainingMetricsQuery."""
        with self.uow:
            model = self.uow.model_repository.get_by_id(query.model_id)
            if not model or not model.training_metrics:
                return None
            
            result = {
                'model_id': model.id,
                'model_name': model.name,
                'status': model.status.value,
                'metrics': model.training_metrics.to_dict(),
                'training_started_at': model.training_started_at.isoformat() if model.training_started_at else None,
                'training_completed_at': model.training_completed_at.isoformat() if model.training_completed_at else None
            }
            
            if query.include_history:
                # TODO: Implement training history from events
                result['history'] = []
            
            return result
    
    def handle_search_models(self, query: SearchModelsQuery) -> Dict[str, Any]:
        """Handle SearchModelsQuery."""
        with self.uow:
            models = self.uow.model_repository.get_all()
            
            # Filter by search term
            search_term_lower = query.search_term.lower()
            filtered_models = []
            
            for model in models:
                if search_term_lower in model.name.lower():
                    filtered_models.append(model)
                elif query.search_in_description and search_term_lower in model.description.lower():
                    filtered_models.append(model)
            
            # Filter by status if specified
            if query.status:
                filtered_models = [m for m in filtered_models if m.status == query.status]
            
            # Limit results
            if query.limit:
                filtered_models = filtered_models[:query.limit]
            
            return {
                'models': [
                    {
                        'id': model.id,
                        'name': model.name,
                        'status': model.status.value,
                        'description': model.description,
                        'created_at': model.created_at.isoformat(),
                        'tags': model.tags
                    }
                    for model in filtered_models
                ],
                'search_term': query.search_term,
                'total_found': len(filtered_models)
            }
    
    def handle_get_model_statistics(self, query: GetModelStatisticsQuery) -> Dict[str, Any]:
        """Handle GetModelStatisticsQuery."""
        with self.uow:
            models = self.uow.model_repository.get_all()
            
            stats = {
                'total_models': len(models),
                'created_today': len([m for m in models if m.created_at.date() == datetime.now().date()])
            }
            
            if query.include_by_status:
                status_counts = {}
                for status in ModelStatus:
                    count = len([m for m in models if m.status == status])
                    status_counts[status.value] = count
                stats['by_status'] = status_counts
            
            if query.include_by_corpus:
                corpus_counts = {}
                for model in models:
                    corpus_counts[model.corpus_id] = corpus_counts.get(model.corpus_id, 0) + 1
                stats['by_corpus'] = corpus_counts
            
            if query.include_training_times:
                completed_models = [m for m in models if m.status == ModelStatus.TRAINED and m.training_metrics]
                if completed_models:
                    training_times = [m.training_metrics.training_time for m in completed_models]
                    stats['training_times'] = {
                        'average': sum(training_times) / len(training_times),
                        'min': min(training_times),
                        'max': max(training_times),
                        'count': len(training_times)
                    }
            
            return stats
    
    def handle_get_recent_activity(self, query: GetRecentActivityQuery) -> Dict[str, Any]:
        """Handle GetRecentActivityQuery."""
        with self.uow:
            cutoff_date = datetime.now() - timedelta(days=query.days)
            
            activity = {
                'period_days': query.days,
                'from_date': cutoff_date.isoformat(),
                'to_date': datetime.now().isoformat()
            }
            
            if query.include_models:
                recent_models = [
                    m for m in self.uow.model_repository.get_all()
                    if m.created_at >= cutoff_date
                ]
                activity['recent_models'] = [
                    {
                        'id': model.id,
                        'name': model.name,
                        'status': model.status.value,
                        'created_at': model.created_at.isoformat()
                    }
                    for model in recent_models
                ]
            
            if query.include_events:
                # Get recent events from event repository
                recent_events = self.uow.event_repository.get_recent(query.limit or 100)
                activity['recent_events'] = recent_events
            
            return activity