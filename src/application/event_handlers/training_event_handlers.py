"""
Event handlers for training-related domain events.

These handlers respond to training events and perform side effects
like logging, notifications, and cleanup operations.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from src.domain.events.training_events import (
    TrainingStarted, TrainingCompleted, TrainingFailed,
    CorpusCreated, ModelArchived
)
from src.domain.repositories import UnitOfWork


logger = logging.getLogger(__name__)


class TrainingEventHandler:
    """Handler for training-related events."""
    
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
    
    def handle_training_started(self, event: TrainingStarted) -> None:
        """Handle training started event."""
        logger.info(
            f"Training started for model {event.model_id} "
            f"using corpus {event.corpus_id}"
        )
        
        # Could trigger additional actions like:
        # - Send notification to user
        # - Update monitoring dashboard
        # - Start resource monitoring
        # - Send webhook to external systems
        
        self._log_training_event("training_started", {
            'model_id': event.model_id,
            'corpus_id': event.corpus_id,
            'config': event.config,
            'timestamp': event.timestamp.isoformat()
        })
    
    def handle_training_completed(self, event: TrainingCompleted) -> None:
        """Handle training completed event."""
        logger.info(
            f"Training completed for model {event.model_id} "
            f"with final loss: {event.final_loss:.4f} "
            f"after {event.epochs_completed} epochs "
            f"in {event.training_time:.1f} seconds"
        )
        
        # Could trigger additional actions like:
        # - Send success notification
        # - Generate training report
        # - Update model registry
        # - Trigger model validation tests
        # - Clean up temporary training files
        
        self._log_training_event("training_completed", {
            'model_id': event.model_id,
            'final_loss': event.final_loss,
            'epochs_completed': event.epochs_completed,
            'training_time': event.training_time,
            'timestamp': event.timestamp.isoformat()
        })
        
        # Auto-cleanup old checkpoints for completed models
        self._cleanup_training_artifacts(event.model_id)
    
    def handle_training_failed(self, event: TrainingFailed) -> None:
        """Handle training failed event."""
        logger.error(
            f"Training failed for model {event.model_id}: {event.error_message}"
        )
        
        # Could trigger additional actions like:
        # - Send failure notification
        # - Create error report
        # - Trigger debugging collection
        # - Alert monitoring systems
        # - Clean up partial training artifacts
        
        self._log_training_event("training_failed", {
            'model_id': event.model_id,
            'error_message': event.error_message,
            'timestamp': event.timestamp.isoformat()
        })
        
        # Clean up failed training artifacts
        self._cleanup_training_artifacts(event.model_id, failed=True)
    
    def handle_corpus_created(self, event: CorpusCreated) -> None:
        """Handle corpus created event."""
        logger.info(
            f"New corpus created: {event.name} ({event.size} chars, {event.language})"
        )
        
        # Could trigger additional actions like:
        # - Run corpus analysis
        # - Generate corpus statistics
        # - Validate corpus quality
        # - Send notification
        # - Update corpus registry
        
        self._log_training_event("corpus_created", {
            'corpus_id': event.corpus_id,
            'name': event.name,
            'size': event.size,
            'language': event.language,
            'timestamp': event.timestamp.isoformat()
        })
        
        # Trigger corpus preprocessing if configured
        self._trigger_corpus_preprocessing(event.corpus_id)
    
    def handle_model_archived(self, event: ModelArchived) -> None:
        """Handle model archived event."""
        logger.info(f"Model {event.model_id} archived: {event.reason}")
        
        # Could trigger additional actions like:
        # - Move model files to archive storage
        # - Update model registry
        # - Send notification
        # - Clean up active model cache
        # - Update resource allocation
        
        self._log_training_event("model_archived", {
            'model_id': event.model_id,
            'reason': event.reason,
            'timestamp': event.timestamp.isoformat()
        })
        
        # Clean up model artifacts
        self._cleanup_model_cache(event.model_id)
    
    def _log_training_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log training event to structured logs."""
        logger.info(
            f"Training event: {event_type}",
            extra={
                'event_type': event_type,
                'event_data': data,
                'service': 'robo_poet_training'
            }
        )
    
    def _cleanup_training_artifacts(self, model_id: str, failed: bool = False) -> None:
        """Clean up training artifacts for a model."""
        try:
            # This would integrate with actual file system cleanup
            # For now, just log the action
            action = "failed training" if failed else "completed training"
            logger.info(f"Cleaning up {action} artifacts for model {model_id}")
            
            # TODO: Implement actual cleanup logic:
            # - Remove temporary checkpoints
            # - Clean up training logs
            # - Remove partial model files (if failed)
            # - Update disk usage metrics
            
        except Exception as e:
            logger.warning(f"Failed to cleanup artifacts for model {model_id}: {e}")
    
    def _cleanup_model_cache(self, model_id: str) -> None:
        """Clean up cached model data."""
        try:
            logger.info(f"Cleaning up model cache for {model_id}")
            
            # TODO: Implement actual cache cleanup:
            # - Remove from memory cache
            # - Clean up temporary files
            # - Update cache statistics
            
        except Exception as e:
            logger.warning(f"Failed to cleanup model cache for {model_id}: {e}")
    
    def _trigger_corpus_preprocessing(self, corpus_id: str) -> None:
        """Trigger corpus preprocessing if needed."""
        try:
            logger.info(f"Triggering preprocessing for corpus {corpus_id}")
            
            # TODO: Implement corpus preprocessing:
            # - Text cleaning and normalization
            # - Tokenization and vocabulary building
            # - Quality assessment
            # - Statistical analysis
            
        except Exception as e:
            logger.warning(f"Failed to trigger preprocessing for corpus {corpus_id}: {e}")


class GenerationEventHandler:
    """Handler for generation-related events."""
    
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
    
    def handle_generation_requested(self, event) -> None:
        """Handle generation requested event."""
        logger.info(f"Generation requested: {event.request_id} for model {event.model_id}")
        
        # Could trigger:
        # - Load model if not cached
        # - Log request for analytics
        # - Start performance monitoring
        
    def handle_generation_completed(self, event) -> None:
        """Handle generation completed event."""
        logger.info(
            f"Generation completed: {event.request_id} "
            f"({event.generated_length} chars in {event.generation_time:.2f}s)"
        )
        
        # Could trigger:
        # - Log performance metrics
        # - Update usage statistics
        # - Cache results if appropriate
        
    def handle_generation_failed(self, event) -> None:
        """Handle generation failed event."""
        logger.error(f"Generation failed: {event.request_id} - {event.error_message}")
        
        # Could trigger:
        # - Error reporting
        # - Model health checks
        # - Fallback generation attempts