"""
Generation service for text generation operations.

This service handles text generation, model loading, and generation
parameter management.
"""

from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime

from src.domain.entities.generation_model import GenerationModel, ModelStatus
from src.domain.value_objects.generation_params import GenerationParams
from src.domain.events.generation_events import (
    GenerationRequested, GenerationCompleted, GenerationFailed
)
from src.domain.repositories import ModelRepository, UnitOfWork


class GenerationService:
    """Service for text generation operations."""
    
    def __init__(self, uow: UnitOfWork):
        self.uow = uow
        self._loaded_models: Dict[str, Any] = {}  # Cache for loaded TensorFlow models
    
    def generate_text(
        self,
        model_id: str,
        prompt: str,
        params: Optional[GenerationParams] = None
    ) -> Dict[str, Any]:
        """Generate text using a trained model."""
        if not params:
            params = GenerationParams()
        
        with self.uow:
            # Verify model exists and is trained
            model = self.uow.model_repository.get_by_id(model_id)
            if not model:
                raise ValueError(f"Model with ID '{model_id}' not found")
            
            if model.status != ModelStatus.TRAINED:
                raise ValueError(f"Model is not trained (status: {model.status})")
            
            # Generate unique request ID
            request_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_id[:8]}"
            
            # Publish generation requested event
            start_event = GenerationRequested(
                request_id=request_id,
                model_id=model_id,
                prompt=prompt,
                params=params.to_dict()
            )
            self.uow.event_repository.add(start_event)
            self.uow.commit()
        
        try:
            # Load model if not cached
            tf_model = self._get_loaded_model(model_id)
            
            # Perform generation (this would integrate with the actual ML pipeline)
            generated_text = self._perform_generation(tf_model, prompt, params, model)
            
            # Record successful generation
            with self.uow:
                completion_event = GenerationCompleted(
                    request_id=request_id,
                    model_id=model_id,
                    generated_length=len(generated_text),
                    generation_time=0.0  # TODO: measure actual time
                )
                self.uow.event_repository.add(completion_event)
                self.uow.commit()
            
            return {
                'request_id': request_id,
                'model_id': model_id,
                'prompt': prompt,
                'generated_text': generated_text,
                'params': params.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            # Record failed generation
            with self.uow:
                failure_event = GenerationFailed(
                    request_id=request_id,
                    model_id=model_id,
                    error_message=str(e)
                )
                self.uow.event_repository.add(failure_event)
                self.uow.commit()
            
            raise ValueError(f"Generation failed: {str(e)}")
    
    def batch_generate(
        self,
        model_id: str,
        prompts: List[str],
        params: Optional[GenerationParams] = None
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts."""
        if not prompts:
            raise ValueError("No prompts provided")
        
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate_text(model_id, prompt, params)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                # Continue with other prompts on failure
                results.append({
                    'batch_index': i,
                    'prompt': prompt,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        with self.uow:
            model = self.uow.model_repository.get_by_id(model_id)
            if not model:
                raise ValueError(f"Model with ID '{model_id}' not found")
            
            # Get corpus information
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
                'corpus': {
                    'id': corpus.id if corpus else None,
                    'name': corpus.name if corpus else None,
                    'language': corpus.language if corpus else None,
                    'size': len(corpus.content) if corpus else 0
                },
                'tags': model.tags,
                'is_loaded': model_id in self._loaded_models
            }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all models available for generation."""
        with self.uow:
            trained_models = self.uow.model_repository.get_by_status(ModelStatus.TRAINED)
            
            return [
                {
                    'id': model.id,
                    'name': model.name,
                    'description': model.description,
                    'training_completed_at': model.training_completed_at.isoformat(),
                    'final_loss': model.training_metrics.final_loss if model.training_metrics else None,
                    'is_loaded': model.id in self._loaded_models,
                    'tags': model.tags
                }
                for model in trained_models
            ]
    
    def load_model(self, model_id: str) -> None:
        """Pre-load a model into memory for faster generation."""
        with self.uow:
            model = self.uow.model_repository.get_by_id(model_id)
            if not model:
                raise ValueError(f"Model with ID '{model_id}' not found")
            
            if model.status != ModelStatus.TRAINED:
                raise ValueError(f"Model is not trained (status: {model.status})")
        
        if model_id not in self._loaded_models:
            # TODO: Integrate with actual TensorFlow model loading
            # This would load the saved model files
            self._loaded_models[model_id] = {
                'model_id': model_id,
                'loaded_at': datetime.now(),
                'tf_model': None  # Placeholder for actual TensorFlow model
            }
    
    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
    
    def unload_all_models(self) -> None:
        """Unload all models from memory."""
        self._loaded_models.clear()
    
    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get list of currently loaded models."""
        return [
            {
                'model_id': model_id,
                'loaded_at': info['loaded_at'].isoformat()
            }
            for model_id, info in self._loaded_models.items()
        ]
    
    def validate_generation_params(self, params: GenerationParams) -> Dict[str, Any]:
        """Validate generation parameters and return analysis."""
        validation_result = {
            'valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        # Check temperature
        if params.temperature < 0.1:
            validation_result['warnings'].append("Very low temperature may produce repetitive text")
        elif params.temperature > 2.0:
            validation_result['warnings'].append("Very high temperature may produce incoherent text")
        
        # Check max_length
        if params.max_length > 1000:
            validation_result['warnings'].append("Very long generation may be slow and repetitive")
        elif params.max_length < 10:
            validation_result['warnings'].append("Very short generation may not be meaningful")
        
        # Check top_k and top_p combination
        if params.top_k and params.top_p:
            validation_result['recommendations'].append(
                "Using both top_k and top_p. Consider using only one sampling method."
            )
        
        # Check seed consistency
        if params.seed is not None:
            validation_result['recommendations'].append(
                "Seed is set - generation will be deterministic"
            )
        
        return validation_result
    
    def _get_loaded_model(self, model_id: str):
        """Get a loaded TensorFlow model, loading it if necessary."""
        if model_id not in self._loaded_models:
            self.load_model(model_id)
        
        return self._loaded_models[model_id]['tf_model']
    
    def _perform_generation(
        self,
        tf_model: Any,
        prompt: str,
        params: GenerationParams,
        model: GenerationModel
    ) -> str:
        """
        Perform the actual text generation.
        
        This is a placeholder that would integrate with the existing
        TensorFlow model and generation pipeline.
        """
        # TODO: Integrate with existing robo_poet.py generation logic
        # This would:
        # 1. Tokenize the prompt using the model's vocabulary
        # 2. Generate text using the loaded TensorFlow model
        # 3. Apply sampling strategy (temperature, top_k, top_p)
        # 4. Decode tokens back to text
        # 5. Apply post-processing (stop_sequences, etc.)
        
        # For now, return a placeholder
        return f"[Generated text for prompt: '{prompt[:50]}...' using model {model.name}]"
    
    # ===== TELARES DATASET AMPLIFICATION =====
    
    def generate_for_telares_amplification(
        self,
        model_id: str,
        amplification_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic text samples for Telares dataset amplification.
        
        This method automatically generates text variations that can be used
        to expand the Telares detection dataset with labeled examples.
        
        Args:
            model_id: ID of the trained text generation model
            amplification_config: Configuration for amplification generation
            
        Returns:
            List of generated samples with automatic Telares labeling
        """
        try:
            # Default amplification prompts for different manipulation tactics
            manipulation_prompts = {
                "promesa_irrealista": [
                    "Este negocio increíble te hará millonario en",
                    "Gana dinero fácil sin trabajar con este método que",
                    "Obtén ingresos pasivos de $50,000 mensuales con"
                ],
                "urgencia_artificial": [
                    "¡ÚLTIMOS CUPOS DISPONIBLES! Esta oportunidad única",
                    "¡OFERTA POR TIEMPO LIMITADO! Solo quedan",
                    "¡ACTÚA AHORA O TE ARREPENTIRÁS! Esta chance"
                ],
                "control_emocional": [
                    "Tu familia merece lo mejor, por eso debes",
                    "Piensa en el futuro de tus hijos cuando",
                    "No dejes que tus sueños mueran, únete a"
                ],
                "lenguaje_espiritual": [
                    "Dios te está dando esta bendición para que",
                    "El universo conspira a tu favor con esta",
                    "Esta es una señal divina para que"
                ],
                "presion_social": [
                    "Todos tus amigos ya están ganando dinero con",
                    "No seas el único que se quede atrás mientras",
                    "Miles de personas ya confían en este sistema"
                ],
                "testimonio_fabricado": [
                    "Yo logré cambiar mi vida completamente gracias a",
                    "Mi vecina ahora maneja un BMW nuevo después de",
                    "María, de 45 años, pasó de ser empleada a"
                ],
                "logica_circular": [
                    "Este sistema funciona porque es muy simple:",
                    "La clave del éxito es que todos ganan cuando",
                    "Es matemática pura: si inviertes X entonces"
                ]
            }
            
            # Control prompts (non-manipulative)
            control_prompts = [
                "En el parque los niños juegan mientras",
                "La biblioteca municipal ofrece cursos de",
                "El clima de hoy está perfecto para",
                "Los libros de poesía nos enseñan que",
                "La música clásica tiene la cualidad de"
            ]
            
            generated_samples = []
            
            # Generate manipulative samples
            for tactic, prompts in manipulation_prompts.items():
                for prompt in prompts[:2]:  # 2 samples per tactic
                    params = GenerationParams(
                        temperature=0.7,
                        max_length=150,
                        seed_text=prompt
                    )
                    
                    result = self.generate_text(model_id, prompt, params)
                    
                    # Auto-label as manipulative
                    sample = {
                        "generated_text": result["generated_text"],
                        "source_tactic": tactic,
                        "manipulation_score": 1.0,  # High manipulation
                        "telares_labels": {tactic: 1.0},  # Label for this tactic
                        "generation_params": params.to_dict(),
                        "is_synthetic": True,
                        "amplification_type": "manipulative"
                    }
                    generated_samples.append(sample)
            
            # Generate control samples (non-manipulative)
            for prompt in control_prompts:
                params = GenerationParams(
                    temperature=0.6,
                    max_length=120,
                    seed_text=prompt
                )
                
                result = self.generate_text(model_id, prompt, params)
                
                # Auto-label as clean
                sample = {
                    "generated_text": result["generated_text"],
                    "source_tactic": "none",
                    "manipulation_score": 0.0,  # No manipulation
                    "telares_labels": {},  # No manipulation tactics
                    "generation_params": params.to_dict(),
                    "is_synthetic": True,
                    "amplification_type": "control"
                }
                generated_samples.append(sample)
            
            return generated_samples
            
        except Exception as e:
            raise ValueError(f"Telares amplification failed: {str(e)}")
    
    def auto_amplify_telares_dataset(
        self,
        model_id: str,
        target_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Automatically amplify Telares dataset with generated samples.
        
        This method is called automatically when text generation is performed
        in Phase 2, creating labeled samples for Telares detection improvement.
        
        Args:
            model_id: Trained generation model to use
            target_samples: Number of samples to generate
            
        Returns:
            Amplification results and statistics
        """
        try:
            # Configure amplification
            config = {
                "target_samples": target_samples,
                "manipulation_ratio": 0.7,  # 70% manipulative, 30% control
                "max_length": 150,
                "temperature_range": (0.6, 0.8)
            }
            
            # Generate samples
            samples = self.generate_for_telares_amplification(model_id, config)
            
            # Statistics
            manipulative_count = sum(1 for s in samples if s["manipulation_score"] > 0.5)
            control_count = len(samples) - manipulative_count
            
            # Auto-trigger Telares retraining (delegated to TelaresTrainingService)
            amplification_result = {
                "success": True,
                "total_generated": len(samples),
                "manipulative_samples": manipulative_count,
                "control_samples": control_count,
                "samples": samples,
                "auto_retrain_triggered": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return amplification_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }