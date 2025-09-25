"""
Telares ML Detector Infrastructure
Handles scikit-learn model training, prediction, and persistence
"""

from typing import List, Optional, Tuple
from pathlib import Path
import joblib
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.metrics import classification_report, multilabel_confusion_matrix
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError("scikit-learn required for Telares ML Detector. Install with: pip install scikit-learn")


class TelaresMLDetector:
    """
    Machine Learning infrastructure for pyramid scheme detection
    Uses TF-IDF + Multi-Output Logistic Regression for WSL2 compatibility
    """
    
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.model_version = "1.0"
        self.tactic_names = [
            "control_emocional",
            "presion_social", 
            "lenguaje_espiritual",
            "logica_circular",
            "urgencia_artificial",
            "testimonio_fabricado", 
            "promesa_irrealista"
        ]
        self._is_trained = False
    
    def train(self, X_train: List[str], y_train: np.ndarray) -> bool:
        """
        Train the ML model with text and labels
        
        Args:
            X_train: List of message texts
            y_train: Multi-label array (n_samples, n_tactics)
            
        Returns:
            True if training successful
        """
        try:
            print(f"[FIRE] Iniciando entrenamiento ML...")
            print(f"   [CHART] Mensajes: {len(X_train)}")
            print(f"     Tácticas: {y_train.shape[1] if y_train.ndim > 1 else 1}")
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                stop_words=None,  # Keep Spanish stop words
                lowercase=True,
                strip_accents='unicode'
            )
            
            # Vectorize training texts
            X_vectorized = self.vectorizer.fit_transform(X_train)
            print(f"    Features: {X_vectorized.shape[1]}")
            
            # Initialize multi-output classifier
            base_classifier = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'  # Handle class imbalance
            )
            
            self.classifier = MultiOutputClassifier(base_classifier)
            
            # Train model
            print("   [BRAIN] Entrenando clasificador...")
            self.classifier.fit(X_vectorized, y_train)
            
            self._is_trained = True
            print("[OK] Entrenamiento completado")
            
            # Quick validation
            self._validate_training(X_train[:5], y_train[:5])
            
            return True
            
        except Exception as e:
            print(f"[X] Error en entrenamiento: {e}")
            return False
    
    def predict(self, messages: List[str]) -> List[List[float]]:
        """
        Predict manipulation tactics for messages
        
        Args:
            messages: List of message texts to analyze
            
        Returns:
            List of prediction arrays, one per message
        """
        if not self.is_ready():
            print("WARNING: Modelo no entrenado")
            return [[0.0] * 7 for _ in messages]
        
        try:
            # Vectorize input messages
            X_vectorized = self.vectorizer.transform(messages)
            
            # Get predictions (probabilities)
            predictions = self.classifier.predict_proba(X_vectorized)
            
            # Extract positive class probabilities for each output
            result = []
            for i in range(len(messages)):
                message_scores = []
                for j in range(len(self.tactic_names)):
                    # Get probability of positive class (index 1)
                    if hasattr(predictions[j][i], '__len__') and len(predictions[j][i]) > 1:
                        prob = predictions[j][i][1]  # Probability of class 1
                    else:
                        prob = predictions[j][i][0] if predictions[j][i][0] > 0.5 else 0.0
                    message_scores.append(float(prob))
                result.append(message_scores)
            
            return result
            
        except Exception as e:
            print(f"WARNING: Error en predicción: {e}")
            return [[0.0] * 7 for _ in messages]
    
    def predict_single(self, message: str) -> List[float]:
        """Predict manipulation tactics for a single message"""
        predictions = self.predict([message])
        return predictions[0] if predictions else [0.0] * 7
    
    def save_model(self, model_path: str, vectorizer_path: str) -> bool:
        """
        Save trained model and vectorizer to disk
        
        Args:
            model_path: Path to save classifier
            vectorizer_path: Path to save vectorizer
            
        Returns:
            True if successful
        """
        try:
            if not self.is_ready():
                print("WARNING: Modelo no entrenado - no se puede guardar")
                return False
            
            # Save classifier
            joblib.dump(self.classifier, model_path)
            
            # Save vectorizer
            joblib.dump(self.vectorizer, vectorizer_path)
            
            print(f"[SAVE] Modelo guardado en: {model_path}")
            print(f"[SAVE] Vectorizador guardado en: {vectorizer_path}")
            
            return True
            
        except Exception as e:
            print(f"[X] Error guardando modelo: {e}")
            return False
    
    def load_model(self, model_path: str, vectorizer_path: str) -> bool:
        """
        Load trained model and vectorizer from disk
        
        Args:
            model_path: Path to classifier file
            vectorizer_path: Path to vectorizer file
            
        Returns:
            True if successful
        """
        try:
            model_file = Path(model_path)
            vectorizer_file = Path(vectorizer_path)
            
            if not model_file.exists():
                print(f"WARNING: Archivo de modelo no encontrado: {model_path}")
                return False
                
            if not vectorizer_file.exists():
                print(f"WARNING: Archivo de vectorizador no encontrado: {vectorizer_path}")
                return False
            
            # Load classifier
            self.classifier = joblib.load(model_path)
            
            # Load vectorizer
            self.vectorizer = joblib.load(vectorizer_path)
            
            self._is_trained = True
            
            print(f"[OK] Modelo cargado desde: {model_path}")
            print(f"[OK] Vectorizador cargado desde: {vectorizer_path}")
            
            return True
            
        except Exception as e:
            print(f"[X] Error cargando modelo: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.classifier is not None and self.vectorizer is not None
    
    def is_ready(self) -> bool:
        """Check if model is ready for prediction"""
        return self.is_loaded() and self._is_trained
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model_version": self.model_version,
            "is_loaded": self.is_loaded(),
            "is_trained": self._is_trained,
            "tactic_names": self.tactic_names,
            "num_tactics": len(self.tactic_names),
            "vectorizer_features": self.vectorizer.max_features if self.vectorizer else None,
            "classifier_type": "Multi-Output Logistic Regression"
        }
    
    def _validate_training(self, X_sample: List[str], y_sample: np.ndarray) -> None:
        """Quick validation with training sample"""
        try:
            predictions = self.predict(X_sample)
            print(f"    Validación: {len(predictions)} predicciones generadas")
            
            # Check if predictions are reasonable
            for i, pred in enumerate(predictions[:2]):  # Show first 2
                message_preview = X_sample[i][:50] + "..." if len(X_sample[i]) > 50 else X_sample[i]
                total_score = sum(pred)
                print(f"   [DOC] '{message_preview}' -> Score: {total_score:.2f}")
                
        except Exception as e:
            print(f"   WARNING: Validación falló: {e}")
    
    def evaluate_performance(self, X_test: List[str], y_test: np.ndarray) -> dict:
        """
        Evaluate model performance on test set
        
        Args:
            X_test: Test messages
            y_test: True labels
            
        Returns:
            Performance metrics
        """
        if not self.is_ready():
            return {"error": "Model not trained"}
        
        try:
            # Get predictions
            predictions = self.predict(X_test)
            y_pred = np.array(predictions)
            
            # Convert probabilities to binary predictions
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate metrics for each tactic
            metrics = {}
            for i, tactic_name in enumerate(self.tactic_names):
                # Skip if no positive examples
                if y_test[:, i].sum() == 0:
                    metrics[tactic_name] = {"precision": 0, "recall": 0, "f1": 0}
                    continue
                
                # Calculate basic metrics
                tp = ((y_pred_binary[:, i] == 1) & (y_test[:, i] == 1)).sum()
                fp = ((y_pred_binary[:, i] == 1) & (y_test[:, i] == 0)).sum()
                fn = ((y_pred_binary[:, i] == 0) & (y_test[:, i] == 1)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[tactic_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
            
            # Overall metrics
            overall_accuracy = (y_pred_binary == y_test).mean()
            
            return {
                "overall_accuracy": overall_accuracy,
                "tactic_metrics": metrics,
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            return {"error": f"Evaluation failed: {e}"}