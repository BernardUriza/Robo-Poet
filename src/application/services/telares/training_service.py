"""
Telares Training Service
Business logic for training and managing ML models for pyramid scheme detection
"""

import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from src.domain.telares.entities import TelaresMessage, ManipulationTactics
from src.infrastructure.telares.ml_detector import TelaresMLDetector
from src.infrastructure.telares.data_loader import TelaresDataLoader


class TelaresTrainingService:
    """
    Application service for training Telares detection models
    Orchestrates training workflow with data loading and model persistence
    """
    
    def __init__(self):
        self.data_loader = TelaresDataLoader()
        self.ml_detector = TelaresMLDetector()
        self.training_metrics = {}
    
    def train_standard_model(self, epochs: int = None) -> Dict[str, any]:
        """
        Train model using standard telares dataset only
        
        Args:
            epochs: Not used in scikit-learn, kept for API compatibility
            
        Returns:
            Training metrics and results
        """
        start_time = time.time()
        
        # Load training data
        print("📊 Cargando dataset de telares...")
        X_train, y_train, metadata = self.data_loader.load_telares_dataset()
        
        if not X_train:
            raise ValueError("No se pudo cargar el dataset de entrenamiento")
        
        print(f"✅ Dataset cargado: {len(X_train)} mensajes")
        print(f"📋 Tácticas: {metadata.get('tactic_names', [])}")
        
        # Train model
        print("🔥 Entrenando detector...")
        success = self.ml_detector.train(X_train, y_train)
        
        if not success:
            raise RuntimeError("Fallo en el entrenamiento del modelo")
        
        # Save model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "telares_detector.joblib"
        vectorizer_path = models_dir / "telares_vectorizer.joblib" 
        
        self.ml_detector.save_model(str(model_path), str(vectorizer_path))
        
        training_time = time.time() - start_time
        
        # Calculate training metrics
        metrics = {
            "training_time": training_time,
            "dataset_size": len(X_train),
            "model_saved": True,
            "model_path": str(model_path),
            "vectorizer_path": str(vectorizer_path),
            "tactic_names": metadata.get('tactic_names', []),
            "success": True
        }
        
        self.training_metrics = metrics
        return metrics
    
    def train_hybrid_model(self, corpus_dir: str = "corpus") -> Dict[str, any]:
        """
        Train hybrid model using telares + poetic corpus as negative controls
        
        Args:
            corpus_dir: Directory containing poetic text files
            
        Returns:
            Training metrics and results
        """
        start_time = time.time()
        
        # Load telares data (positive examples)
        print("📊 Cargando dataset de telares (manipulativos)...")
        X_telares, y_telares, metadata = self.data_loader.load_telares_dataset()
        
        if not X_telares:
            raise ValueError("No se pudo cargar el dataset de telares")
        
        # Load poetic corpus as negative controls
        print("📚 Cargando corpus poético (controles negativos)...")
        X_poetic = self.data_loader.load_poetic_corpus(corpus_dir)
        
        # Create labels for poetic texts (all zeros - no manipulation)
        y_poetic = np.zeros((len(X_poetic), len(metadata['tactic_names'])))
        
        # Combine datasets
        X_combined = X_telares + X_poetic
        y_combined = np.vstack([y_telares, y_poetic]) if len(X_poetic) > 0 else y_telares
        
        print(f"✅ Dataset híbrido creado:")
        print(f"   📨 Mensajes telares: {len(X_telares)}")
        print(f"   📚 Fragmentos poéticos: {len(X_poetic)}")
        print(f"   📊 Total entrenamiento: {len(X_combined)}")
        
        # Train hybrid model
        print("🔥 Entrenando detector híbrido...")
        success = self.ml_detector.train(X_combined, y_combined)
        
        if not success:
            raise RuntimeError("Fallo en el entrenamiento del modelo híbrido")
        
        # Save hybrid model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "telares_hybrid_detector.joblib"
        vectorizer_path = models_dir / "telares_hybrid_vectorizer.joblib"
        
        self.ml_detector.save_model(str(model_path), str(vectorizer_path))
        
        training_time = time.time() - start_time
        
        # Calculate hybrid training metrics  
        metrics = {
            "training_time": training_time,
            "telares_messages": len(X_telares),
            "poetic_fragments": len(X_poetic),
            "total_dataset_size": len(X_combined),
            "model_type": "hybrid",
            "model_saved": True,
            "model_path": str(model_path),
            "vectorizer_path": str(vectorizer_path),
            "tactic_names": metadata.get('tactic_names', []),
            "success": True
        }
        
        self.training_metrics = metrics
        return metrics
    
    def validate_model(self, test_messages: List[str] = None) -> Dict[str, any]:
        """
        Validate trained model with test messages
        
        Args:
            test_messages: Optional test messages, uses built-in if None
            
        Returns:
            Validation metrics
        """
        if not self.ml_detector.is_loaded():
            raise ValueError("Modelo no entrenado - ejecute training primero")
        
        # Use built-in test messages if none provided
        if test_messages is None:
            test_messages = [
                "Únete a este negocio increíble y gana millones sin esfuerzo",
                "Esta es una oportunidad única que cambiará tu vida para siempre", 
                "Los poemas de Neruda expresan profundas emociones humanas",
                "Hoy es un día hermoso para caminar por el parque"
            ]
        
        print(f"🧪 Validando modelo con {len(test_messages)} mensajes de prueba...")
        
        # Get predictions
        predictions = self.ml_detector.predict(test_messages)
        
        validation_results = []
        for i, message in enumerate(test_messages):
            scores = predictions[i] if i < len(predictions) else [0.0] * 7
            total_score = sum(scores)
            
            validation_results.append({
                "message": message[:50] + "..." if len(message) > 50 else message,
                "total_score": total_score,
                "risk_level": "ALTO" if total_score > 3.0 else "BAJO",
                "individual_scores": scores
            })
        
        return {
            "validation_completed": True,
            "test_messages_count": len(test_messages),
            "results": validation_results,
            "model_responsive": True
        }
    
    def get_training_status(self) -> Dict[str, any]:
        """Get current training status and metrics"""
        return {
            "model_loaded": self.ml_detector.is_loaded(),
            "model_ready": self.ml_detector.is_ready(),
            "last_training_metrics": self.training_metrics,
            "data_loader_ready": True,
            "supported_formats": ["CSV", "TXT"],
            "available_datasets": self._get_available_datasets()
        }
    
    def _get_available_datasets(self) -> List[str]:
        """List available datasets for training"""
        datasets = []
        
        # Check for telares dataset
        telares_dataset = Path("src/data/telares_dataset_135.csv")
        if telares_dataset.exists():
            datasets.append(f"Telares Dataset ({telares_dataset.name})")
        
        # Check for poetic corpus
        corpus_dir = Path("corpus")
        if corpus_dir.exists():
            txt_files = list(corpus_dir.glob("*.txt"))
            if txt_files:
                datasets.append(f"Poetic Corpus ({len(txt_files)} files)")
        
        return datasets
    
    # ===== AUTO-AMPLIFICATION WITH GENERATED SAMPLES =====
    
    def integrate_generated_samples(self, generated_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Integrate generated samples from GenerationService into Telares training.
        
        This method processes synthetic samples created by the text generation model
        and incorporates them into the Telares detection training pipeline.
        
        Args:
            generated_samples: List of generated samples with automatic labeling
            
        Returns:
            Integration results and updated training metrics
        """
        try:
            # Load existing telares dataset
            print("📊 Cargando dataset Telares existente...")
            X_original, y_original, metadata = self.data_loader.load_telares_dataset()
            
            if not X_original:
                raise ValueError("No se pudo cargar dataset original de Telares")
            
            # Process generated samples
            print(f"🔄 Procesando {len(generated_samples)} muestras generadas...")
            
            X_generated = []
            y_generated_list = []
            
            for sample in generated_samples:
                # Extract text
                text = sample.get("generated_text", "")
                if not text or len(text.strip()) < 20:
                    continue  # Skip too short samples
                
                X_generated.append(text)
                
                # Convert telares_labels to array format
                labels_array = np.zeros(len(self.data_loader.tactic_names))
                
                telares_labels = sample.get("telares_labels", {})
                for tactic, score in telares_labels.items():
                    if tactic in self.data_loader.tactic_names:
                        tactic_index = self.data_loader.tactic_names.index(tactic)
                        labels_array[tactic_index] = float(score)
                
                y_generated_list.append(labels_array)
            
            if not X_generated:
                raise ValueError("No se generaron muestras válidas")
            
            y_generated = np.array(y_generated_list)
            
            # Combine datasets (original + generated)
            X_combined = X_original + X_generated
            y_combined = np.vstack([y_original, y_generated])
            
            print(f"✅ Dataset combinado creado:")
            print(f"   📨 Mensajes originales: {len(X_original)}")
            print(f"   🤖 Muestras generadas: {len(X_generated)}")
            print(f"   📊 Total combinado: {len(X_combined)}")
            
            # Train with combined dataset
            print("🔥 Re-entrenando detector con dataset ampliado...")
            success = self.ml_detector.train(X_combined, y_combined)
            
            if not success:
                raise RuntimeError("Fallo en el re-entrenamiento con dataset ampliado")
            
            # Save amplified model
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            model_path = models_dir / "telares_amplified_detector.joblib"
            vectorizer_path = models_dir / "telares_amplified_vectorizer.joblib"
            
            self.ml_detector.save_model(str(model_path), str(vectorizer_path))
            
            # Calculate improvement metrics
            original_size = len(X_original)
            amplified_size = len(X_combined)
            improvement_ratio = amplified_size / original_size
            
            # Analyze generated samples
            manipulative_generated = sum(1 for sample in generated_samples 
                                       if sample.get("manipulation_score", 0) > 0.5)
            control_generated = len(generated_samples) - manipulative_generated
            
            integration_result = {
                "success": True,
                "original_dataset_size": original_size,
                "generated_samples_added": len(X_generated),
                "total_dataset_size": amplified_size,
                "improvement_ratio": improvement_ratio,
                "manipulative_generated": manipulative_generated,
                "control_generated": control_generated,
                "model_path": str(model_path),
                "vectorizer_path": str(vectorizer_path),
                "training_time": time.time(),  # Placeholder
                "auto_amplification": True
            }
            
            # Update training metrics
            self.training_metrics = integration_result
            
            print("✅ INTEGRACIÓN DE MUESTRAS GENERADAS COMPLETADA")
            print(f"🎯 Mejora del dataset: {improvement_ratio:.1f}x")
            print(f"🤖 Muestras manipulativas: {manipulative_generated}")
            print(f"🧪 Muestras de control: {control_generated}")
            
            return integration_result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
            print(f"❌ Error en integración de muestras: {e}")
            return error_result
    
    def auto_retrain_with_amplification(self, generation_service_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically retrain Telares detector when Generation Service creates new samples.
        
        This method is triggered automatically when Phase 2 generation creates
        synthetic samples that can improve Telares detection.
        
        Args:
            generation_service_result: Result from GenerationService amplification
            
        Returns:
            Auto-retraining results
        """
        try:
            if not generation_service_result.get("success", False):
                return {
                    "auto_retrain": False,
                    "reason": "Generation service failed",
                    "original_error": generation_service_result.get("error", "Unknown")
                }
            
            generated_samples = generation_service_result.get("samples", [])
            if not generated_samples:
                return {
                    "auto_retrain": False,
                    "reason": "No generated samples available"
                }
            
            print("🔄 AUTO-REENTRENAMIENTO TELARES DETECTOR ACTIVADO")
            print(f"🤖 Integrando {len(generated_samples)} muestras sintéticas...")
            
            # Integrate generated samples
            integration_result = self.integrate_generated_samples(generated_samples)
            
            if integration_result["success"]:
                print("✅ AUTO-REENTRENAMIENTO COMPLETADO")
                print("🛡️ Telares detector mejorado automáticamente")
                
                return {
                    "auto_retrain": True,
                    "integration_result": integration_result,
                    "improvement_stats": {
                        "dataset_growth": integration_result["improvement_ratio"],
                        "new_samples": integration_result["generated_samples_added"],
                        "total_size": integration_result["total_dataset_size"]
                    },
                    "timestamp": time.time()
                }
            else:
                return {
                    "auto_retrain": False,
                    "reason": "Integration failed",
                    "error": integration_result.get("error", "Unknown integration error")
                }
                
        except Exception as e:
            return {
                "auto_retrain": False,
                "reason": "Auto-retrain exception",
                "error": str(e),
                "timestamp": time.time()
            }