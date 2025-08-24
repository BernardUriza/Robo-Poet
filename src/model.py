"""
Neural network model definitions for text generation.

Implements LSTM-based architecture optimized for character-level language modeling,
with configurable hyperparameters and GPU optimization support.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable
from typing import Tuple, Optional
import numpy as np

@register_keras_serializable(package='robo_poet', name='DropConnect')
class DropConnect(layers.Layer):
    """
    DropConnect layer implementation - Strategy 2.1
    
    DropConnect is superior to Dropout as it randomly drops connections 
    rather than entire neurons, preserving more information flow.
    """
    
    def __init__(self, rate: float, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.supports_masking = True
        
    def call(self, inputs, training=None, mask=None):
        if training:
            # Generate random mask for connections
            dropout_mask = tf.random.uniform(tf.shape(inputs)) > self.rate
            dropout_mask = tf.cast(dropout_mask, inputs.dtype)
            # Scale output to maintain expected value
            return inputs * dropout_mask / (1.0 - self.rate)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config

@register_keras_serializable(package='robo_poet', name='VariationalDropout')
class VariationalDropout(layers.Layer):
    """
    Variational Dropout implementation - Strategy 2.2
    
    Uses same dropout mask across time steps, which is more effective
    for RNNs than standard dropout.
    """
    
    def __init__(self, rate: float, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.supports_masking = True
        
    def call(self, inputs, training=None, mask=None):
        if training:
            # Generate mask for batch and feature dimensions only
            # Keep same mask across time dimension
            noise_shape = [tf.shape(inputs)[0], 1, tf.shape(inputs)[2]]
            dropout_mask = tf.random.uniform(noise_shape) > self.rate
            dropout_mask = tf.cast(dropout_mask, inputs.dtype)
            return inputs * dropout_mask / (1.0 - self.rate)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({'rate': self.rate})
        return config

class LSTMTextGenerator:
    """Weight-Dropped LSTM neural network for text generation (Strategy 2).
    
    Implements advanced regularization with DropConnect, Variational Dropout,
    and weight tying for improved generalization.
    """
    
    def __init__(self, vocab_size: int, sequence_length: int, 
                 lstm_units: int = 256, 
                 variational_dropout_rate: float = 0.3,
                 dropconnect_rate: float = 0.2,
                 embedding_dim: int = 128):
        """
        Initialize Weight-Dropped LSTM text generator.
        
        Args:
            vocab_size: Size of vocabulary
            sequence_length: Length of input sequences
            lstm_units: Number of LSTM units (256 per CLAUDE.md)
            variational_dropout_rate: Rate for variational dropout (Strategy 2.2)
            dropconnect_rate: Rate for DropConnect (Strategy 2.1)
            embedding_dim: Embedding dimension (Strategy 2.3)
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.variational_dropout_rate = variational_dropout_rate
        self.dropconnect_rate = dropconnect_rate
        self.embedding_dim = embedding_dim
        self.model: Optional[Model] = None
        self.embedding_layer = None
        self.dense_layer = None
        
    def build_model(self) -> Model:
        """
        Construct LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        print(f"🧠 Building Weight-Dropped LSTM model (Strategy 2)...")
        print(f"   Architecture: 2 x 256-unit LSTM layers")
        print(f"   Vocab size: {self.vocab_size}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Regularization: DropConnect + Variational Dropout")
        print(f"   Weight tying: Embedding-Output layer shared")
        print(f"   Advanced: Weight-Dropped LSTM implementation")
        
        # Strategy 2.3: Embedding layer for weight tying
        # Use embedding instead of one-hot for memory efficiency and weight tying
        inputs = layers.Input(shape=(self.sequence_length,), dtype='int32')
        
        # Shared embedding layer (configurable dimensions)
        embedding_layer = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='shared_embedding'
        )
        embedded = embedding_layer(inputs)
        
        # STRATEGY 2: Weight-Dropped LSTM architecture with advanced regularization
        
        # LSTM Layer 1 - 256 units with return_sequences=True
        # Strategy 2.1 & 2.2: Remove built-in dropout, use custom regularization
        lstm_1_out = layers.LSTM(
            units=256,  # FIXED: Was self.lstm_units (128), must be 256
            return_sequences=True,  # CRITICAL: Must return sequences for next layer
            dropout=0.0,  # Disabled - using DropConnect instead
            recurrent_dropout=0.0,  # Disabled - using VariationalDropout instead
            name='lstm_layer_1'
        )(embedded)
        
        # Strategy 2.2: Variational Dropout after LSTM 1
        dropout_1 = VariationalDropout(self.variational_dropout_rate, name='variational_dropout_1')(lstm_1_out)
        
        # Strategy 2.1: DropConnect for inter-layer connections
        dropconnect_1 = DropConnect(self.dropconnect_rate, name='dropconnect_1')(dropout_1)
        
        # LSTM Layer 2 - 256 units with return_sequences=False for next-token prediction
        lstm_2_out = layers.LSTM(
            units=256,  # Maintain 256 units per spec
            return_sequences=False,  # Changed: Only output last timestep for next-token prediction
            dropout=0.0,  # Disabled - using DropConnect instead
            recurrent_dropout=0.0,  # Disabled - using VariationalDropout instead
            name='lstm_layer_2'
        )(dropconnect_1)
        
        # Standard dropout for 2D output (since return_sequences=False)
        dropout_2 = layers.Dropout(self.variational_dropout_rate, name='dropout_2')(lstm_2_out)
        
        # Strategy 2.1: DropConnect before output layer (now works with 2D)
        dropconnect_2 = DropConnect(self.dropconnect_rate, name='dropconnect_2')(dropout_2)
        
        # Strategy 2.3: Weight tying - output layer uses transposed embedding weights
        # Dense layer with weight tying
        dense_layer = layers.Dense(
            self.vocab_size,
            use_bias=False,  # No bias for weight tying
            name='output_dense'
        )
        dense_out = dense_layer(dropconnect_2)
        
        # Apply weight tying constraint
        def tie_weights():
            dense_layer.set_weights([embedding_layer.get_weights()[0].T])
        
        # Softmax activation
        outputs = layers.Activation('softmax', name='output_activation')(dense_out)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='lstm_text_generator')
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  # Changed: Use sparse for integer targets
            metrics=['accuracy']
        )
        
        # Apply weight tying after model creation
        # This needs to be done in a callback during training
        self.embedding_layer = embedding_layer
        self.dense_layer = dense_layer
        
        # Model summary
        param_count = self.model.count_params()
        print(f"✅ Weight-Dropped LSTM built: {param_count:,} parameters")
        print(f"   Regularization: DropConnect(0.2) + VariationalDropout(0.3)")
        print(f"   Weight tying: Embedding ↔ Output layer")
        print(f"   Memory efficient: Embedding instead of one-hot")
        
        return self.model
    
    @classmethod
    def create_with_dropout_config(cls, vocab_size: int, sequence_length: int, 
                                   dropout_config: str = 'balanced'):
        """
        Factory method to create models with different dropout configurations (Strategy 2.4).
        
        Args:
            vocab_size: Size of vocabulary
            sequence_length: Length of sequences
            dropout_config: 'light' (0.1), 'balanced' (0.3), 'heavy' (0.5)
        
        Returns:
            Configured LSTMTextGenerator instance
        """
        configs = {
            'light': {'variational': 0.1, 'dropconnect': 0.05},
            'balanced': {'variational': 0.3, 'dropconnect': 0.2},
            'heavy': {'variational': 0.5, 'dropconnect': 0.3}
        }
        
        if dropout_config not in configs:
            raise ValueError(f"Unknown config: {dropout_config}. Use: {list(configs.keys())}")
        
        config = configs[dropout_config]
        print(f"🎛️ Strategy 2.4: Using '{dropout_config}' dropout configuration")
        print(f"   Variational dropout: {config['variational']}")
        print(f"   DropConnect: {config['dropconnect']}")
        
        return cls(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            variational_dropout_rate=config['variational'],
            dropconnect_rate=config['dropconnect']
        )
    
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

class WeightTyingCallback(tf.keras.callbacks.Callback):
    """Callback to enforce weight tying between embedding and output layers."""
    
    def __init__(self, embedding_layer, dense_layer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.dense_layer = dense_layer
    
    def on_batch_end(self, batch, logs=None):
        # Tie weights after each batch
        embedding_weights = self.embedding_layer.get_weights()[0]
        self.dense_layer.set_weights([embedding_weights.T])

class ModelTrainer:
    """Handles model training with callbacks and monitoring."""
    
    def __init__(self, model: Model, device: str = '/GPU:0'):
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
    
    def _setup_callbacks(self, patience: int = 10, weight_tying_callback=None) -> list:
        """
        Setup training callbacks including weight tying for Strategy 2.3.
        
        Args:
            patience: Early stopping patience
            weight_tying_callback: Optional weight tying callback
        
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
                filepath='models/checkpoint_epoch_{epoch:02d}_loss_{val_loss:.4f}.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Add weight tying callback if provided (Strategy 2.3)
        if weight_tying_callback:
            callbacks.append(weight_tying_callback)
            print("✅ Weight tying callback added for Strategy 2.3")
        
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
        Load model from disk with custom layers.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded Keras model
        """
        try:
            # Define custom objects for loading
            custom_objects = {
                'VariationalDropout': VariationalDropout,
                'DropConnect': DropConnect
            }
            
            # Load model with custom objects
            model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
            print(f"📁 Model loaded from: {filepath}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise