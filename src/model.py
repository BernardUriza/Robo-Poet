"""
Neural network model definitions for text generation.

Implements LSTM-based architecture optimized for character-level language modeling,
with configurable hyperparameters and GPU optimization support.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Optional
import numpy as np

class LSTMTextGenerator:
    """LSTM-based neural network for character-level text generation."""
    
    def __init__(self, vocab_size: int, sequence_length: int, 
                 lstm_units: int = 128, dropout_rate: float = 0.2):
        """
        Initialize LSTM text generator.
        
        Args:
            vocab_size: Size of character vocabulary
            sequence_length: Length of input sequences
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model: Optional[Model] = None
        
    def build_model(self) -> Model:
        """
        Construct LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        print(f"🧠 Building LSTM model...")
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   LSTM units: {self.lstm_units}")
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.vocab_size))
        
        # LSTM layers with dropout
        lstm_out = layers.LSTM(
            self.lstm_units,
            dropout=self.dropout_rate,
            return_sequences=False,
            name='lstm_layer'
        )(inputs)
        
        # Output layer with softmax activation
        outputs = layers.Dense(
            self.vocab_size,
            activation='softmax',
            name='output_layer'
        )(lstm_out)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='lstm_text_generator')
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Model summary
        param_count = self.model.count_params()
        print(f"✅ Model built: {param_count:,} parameters")
        
        return self.model
    
    def get_model_summary(self) -> str:
        """
        Get detailed model architecture summary.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet. Call build_model() first."
        
        # Capture summary as string
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary

class ModelTrainer:
    """Handles model training with callbacks and monitoring."""
    
    def __init__(self, model: Model, device: str = '/CPU:0'):
        """
        Initialize model trainer.
        
        Args:
            model: Keras model to train
            device: Device to use for training
        """
        self.model = model
        self.device = device
        self.history = None
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              batch_size: int = 128, epochs: int = 5,
              validation_split: float = 0.2) -> tf.keras.callbacks.History:
        """
        Train the model with specified parameters.
        
        Args:
            X: Input training data
            y: Target training data
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            
        Returns:
            Training history object
        """
        print(f"⚡ Starting training on {self.device}")
        print(f"   Data shape: X{X.shape}, y{y.shape}")
        print(f"   Batch size: {batch_size}, Epochs: {epochs}")
        print(f"   Validation split: {validation_split:.1%}")
        
        # Setup callbacks
        callbacks = self._setup_callbacks(patience=10)
        
        # Train on specified device
        with tf.device(self.device):
            self.history = self.model.fit(
                X, y,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
        
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"✅ Training completed!")
        print(f"   Final training loss: {final_loss:.4f}")
        print(f"   Final validation loss: {final_val_loss:.4f}")
        
        return self.history
    
    def _setup_callbacks(self, patience: int = 10) -> list:
        """
        Setup training callbacks.
        
        Args:
            patience: Early stopping patience
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/checkpoint_epoch_{epoch:02d}_loss_{val_loss:.4f}.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate trained model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("📊 Evaluating model...")
        
        with tf.device(self.device):
            test_loss, test_accuracy = self.model.evaluate(
                X_test, y_test, verbose=0
            )
        
        # Calculate perplexity
        perplexity = np.exp(test_loss)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'perplexity': perplexity
        }
        
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Perplexity: {perplexity:.2f}")
        
        return metrics

class ModelManager:
    """Handles model saving and loading operations."""
    
    @staticmethod
    def save_model(model: Model, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained Keras model
            filepath: Path to save model
        """
        try:
            model.save(filepath)
            print(f"💾 Model saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    @staticmethod
    def load_model(filepath: str) -> Model:
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded Keras model
        """
        try:
            model = tf.keras.models.load_model(filepath)
            print(f"📁 Model loaded from: {filepath}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise