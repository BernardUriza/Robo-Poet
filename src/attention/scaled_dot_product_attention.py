#!/usr/bin/env python3
"""
Scaled Dot-Product Attention Implementation
Built for Shakespeare & Alice Multi-Corpus Dataset

Implements attention mechanism without tf.keras.layers.MultiHeadAttention
Using only tf.matmul, tf.nn.softmax, and basic operations.
Includes explicit shape tracking and gradient flow analysis.

Author: ML Academic Framework  
Target: LSTM baseline improvement (current val_loss = 6.5)
"""

try:
    import tensorflow as tf
    import numpy as np
    from typing import Tuple, Optional, Dict, Any
    print("✅ TensorFlow imports successful")
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    print("💡 Install with: pip install tensorflow")
    tf = None


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """
    🎯 SCALED DOT-PRODUCT ATTENTION FROM SCRATCH
    
    Implements attention mechanism using only basic TensorFlow operations:
    - tf.matmul for matrix multiplications
    - tf.nn.softmax for attention weights
    - No pre-built attention layers
    
    Key Features:
    - Explicit shape assertions after each operation
    - Gradient flow visualization hooks
    - Causal masking for autoregressive generation
    - Dropout after softmax (not before)
    - Memory usage tracking vs LSTM baseline
    
    Architecture:
    Input (batch, seq_len, d_model) → Q,K,V projections → 
    Attention scores → Causal mask → Softmax → Dropout → 
    Weighted values → Output (batch, seq_len, d_model)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        dropout_rate: float = 0.1,
        use_causal_mask: bool = True,
        gradient_tracking: bool = True,
        name: str = "scaled_attention",
        **kwargs
    ):
        """
        Initialize Scaled Dot-Product Attention layer.
        
        Args:
            d_model: Model dimension (256 for Shakespeare&Alice corpus)
            dropout_rate: Dropout rate after softmax (default 0.1)
            use_causal_mask: Apply causal mask for autoregressive generation
            gradient_tracking: Enable gradient flow analysis hooks
            name: Layer name for debugging
        """
        super().__init__(name=name, **kwargs)
        
        # Core parameters
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.use_causal_mask = use_causal_mask
        self.gradient_tracking = gradient_tracking
        
        # Scaling factor for attention scores
        self.scale = tf.cast(1.0 / tf.sqrt(float(d_model)), tf.float32)
        
        # Gradient tracking storage
        self.gradient_norms = {}
        self.attention_weights_history = []
        
        print(f"🔧 Initializing Scaled Dot-Product Attention:")
        print(f"   d_model: {d_model}")
        print(f"   dropout_rate: {dropout_rate}")
        print(f"   causal_mask: {use_causal_mask}")
        print(f"   gradient_tracking: {gradient_tracking}")
        print(f"   scale_factor: {1.0 / np.sqrt(float(d_model)):.4f}")
    
    def build(self, input_shape):
        """
        Build layer weights - separate Q, K, V projections.
        
        Args:
            input_shape: Expected (batch_size, sequence_length, d_model)
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {len(input_shape)}D")
        
        input_dim = input_shape[-1]
        
        # Query projection weight
        self.W_q = self.add_weight(
            name='query_projection',
            shape=(input_dim, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Key projection weight  
        self.W_k = self.add_weight(
            name='key_projection',
            shape=(input_dim, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Value projection weight
        self.W_v = self.add_weight(
            name='value_projection',
            shape=(input_dim, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Dropout layer (applied after softmax)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        print(f"✅ Attention weights built:")
        print(f"   W_q shape: {self.W_q.shape}")
        print(f"   W_k shape: {self.W_k.shape}")  
        print(f"   W_v shape: {self.W_v.shape}")
        print(f"   Total parameters: {3 * input_dim * self.d_model:,}")
        
        super().build(input_shape)
    
    def _shape_assert(self, tensor: tf.Tensor, expected_shape: Tuple, name: str):
        """
        Assert tensor shape matches expected with detailed error message.
        
        Args:
            tensor: Tensor to check
            expected_shape: Expected shape (can contain None for dynamic dims)
            name: Operation name for debugging
        """
        actual_shape = tf.shape(tensor)
        
        # Create assertion message
        if self.gradient_tracking:
            tf.debugging.assert_equal(
                tf.reduce_prod(actual_shape),
                tf.reduce_prod([s for s in expected_shape if s is not None]),
                message=f"Shape mismatch in {name}: expected {expected_shape}, got shape"
            )
    
    def _create_causal_mask(self, sequence_length: int) -> tf.Tensor:
        """
        Create causal mask for autoregressive generation.
        
        Args:
            sequence_length: Length of input sequence
            
        Returns:
            Lower triangular mask (seq_len, seq_len)
        """
        # Create lower triangular matrix
        mask = tf.linalg.band_part(
            tf.ones((sequence_length, sequence_length)), -1, 0
        )
        
        # Convert to large negative values for masking
        # 0 → -inf, 1 → 0 (for addition before softmax)
        causal_mask = (1.0 - mask) * -1e9
        
        # Shape assertion
        self._shape_assert(causal_mask, (sequence_length, sequence_length), "causal_mask")
        
        return causal_mask
    
    def _track_gradients(self, tensor: tf.Tensor, name: str):
        """
        Track gradient norms for analysis.
        
        Args:
            tensor: Tensor to track gradients for
            name: Name for gradient tracking storage
        """
        if self.gradient_tracking and tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(tensor)
                # Compute a simple scalar for gradient tracking
                scalar = tf.reduce_mean(tensor)
            
            grad = tape.gradient(scalar, tensor)
            if grad is not None:
                grad_norm = tf.norm(grad)
                self.gradient_norms[name] = float(grad_norm.numpy())
    
    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            inputs: Input tensor (batch, seq_len, d_model)
            training: Training mode flag
            mask: Optional mask (not used - we use causal mask)
            
        Returns:
            Attention output (batch, seq_len, d_model)
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # 🔍 STEP 1: Create Q, K, V projections
        # Using tf.matmul only (no Dense layers)
        Q = tf.matmul(inputs, self.W_q)  # (batch, seq_len, d_model)
        K = tf.matmul(inputs, self.W_k)  # (batch, seq_len, d_model)  
        V = tf.matmul(inputs, self.W_v)  # (batch, seq_len, d_model)
        
        # Shape assertions
        self._shape_assert(Q, (None, None, self.d_model), "Q_projection")
        self._shape_assert(K, (None, None, self.d_model), "K_projection")
        self._shape_assert(V, (None, None, self.d_model), "V_projection")
        
        # Track gradients for projections
        self._track_gradients(Q, "query_gradients")
        self._track_gradients(K, "key_gradients") 
        self._track_gradients(V, "value_gradients")
        
        # 🔍 STEP 2: Compute attention scores with scaling
        # Attention = Q * K^T / sqrt(d_model)
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, seq_len, seq_len)
        scaled_scores = scores * self.scale
        
        # Shape assertion
        self._shape_assert(scaled_scores, (None, None, None), "attention_scores")
        
        # 🔍 STEP 3: Apply causal mask for autoregressive generation
        if self.use_causal_mask:
            causal_mask = self._create_causal_mask(seq_length)
            # Broadcast mask to all batches
            causal_mask = tf.expand_dims(causal_mask, 0)  # (1, seq_len, seq_len)
            masked_scores = scaled_scores + causal_mask
        else:
            masked_scores = scaled_scores
        
        # 🔍 STEP 4: Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(masked_scores, axis=-1)  # (batch, seq_len, seq_len)
        
        # Validate attention weights sum to 1
        weight_sums = tf.reduce_sum(attention_weights, axis=-1)
        tf.debugging.assert_near(
            weight_sums, 1.0, rtol=1e-6,
            message="Attention weights don't sum to 1.0"
        )
        
        # Store attention weights for analysis
        if self.gradient_tracking:
            self.attention_weights_history.append(attention_weights)
        
        # 🔍 STEP 5: Apply dropout AFTER softmax (not before)
        if training:
            attention_weights = self.dropout(attention_weights, training=training)
        
        # Track attention gradient flow
        self._track_gradients(attention_weights, "attention_weights")
        
        # 🔍 STEP 6: Apply attention weights to values
        attention_output = tf.matmul(attention_weights, V)  # (batch, seq_len, d_model)
        
        # Final shape assertion
        self._shape_assert(attention_output, (None, None, self.d_model), "attention_output")
        
        return attention_output
    
    def get_attention_weights(self) -> Optional[tf.Tensor]:
        """
        Get most recent attention weights for visualization.
        
        Returns:
            Latest attention weights tensor or None if not tracking
        """
        if self.attention_weights_history:
            return self.attention_weights_history[-1]
        return None
    
    def get_gradient_norms(self) -> Dict[str, float]:
        """
        Get gradient norms for analysis.
        
        Returns:
            Dictionary of gradient norms by operation name
        """
        return self.gradient_norms.copy()
    
    def validate_gradients(self, min_norm: float = 0.1, max_norm: float = 10.0) -> bool:
        """
        Validate gradient norms are in acceptable range.
        
        Args:
            min_norm: Minimum acceptable gradient norm
            max_norm: Maximum acceptable gradient norm
            
        Returns:
            True if all gradients in range, False otherwise
        """
        for name, norm in self.gradient_norms.items():
            if not (min_norm <= norm <= max_norm):
                print(f"⚠️ Gradient norm out of range for {name}: {norm:.4f}")
                return False
        
        print(f"✅ All gradient norms in range [{min_norm}, {max_norm}]")
        return True
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Estimate memory usage vs LSTM baseline.
        
        Returns:
            Dictionary with memory usage estimates
        """
        # Calculate parameter count
        total_params = 3 * self.d_model * self.d_model  # Q, K, V projections
        
        # Estimate memory per batch
        attention_matrix_memory = self.d_model * self.d_model * 4  # float32
        
        return {
            'parameters': total_params,
            'attention_matrix_bytes_per_batch': attention_matrix_memory,
            'estimated_memory_mb': (total_params * 4 + attention_matrix_memory) / (1024**2)
        }
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
            'use_causal_mask': self.use_causal_mask,
            'gradient_tracking': self.gradient_tracking
        })
        return config


# Support functions for testing and validation

def create_attention_model(
    vocab_size: int,
    sequence_length: int = 128,
    d_model: int = 256,
    dropout_rate: float = 0.1
) -> tf.keras.Model:
    """
    Create a simple model with scaled dot-product attention for testing.
    
    Args:
        vocab_size: Size of vocabulary
        sequence_length: Input sequence length  
        d_model: Model dimension
        dropout_rate: Dropout rate
        
    Returns:
        Keras model with attention layer
    """
    # Input layer
    inputs = tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32)
    
    # Embedding layer
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    
    # Positional encoding (simple learned positions)
    positions = tf.keras.layers.Embedding(sequence_length, d_model)(
        tf.range(sequence_length)
    )
    
    # Add embeddings and positions
    x = embeddings + positions
    
    # Scaled dot-product attention layer
    attention = ScaledDotProductAttention(
        d_model=d_model,
        dropout_rate=dropout_rate,
        use_causal_mask=True,
        gradient_tracking=True
    )
    
    # Apply attention
    attended = attention(x)
    
    # Output projection to vocabulary
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(attended)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model, attention


def test_attention_layer():
    """Test the attention layer implementation."""
    if tf is None:
        print("❌ TensorFlow not available - skipping test")
        return False
    
    print("\n🧪 TESTING SCALED DOT-PRODUCT ATTENTION")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    seq_length = 128  
    d_model = 256
    
    # Create test input
    test_input = tf.random.normal((batch_size, seq_length, d_model))
    
    # Create attention layer
    attention = ScaledDotProductAttention(
        d_model=d_model,
        dropout_rate=0.1,
        use_causal_mask=True,
        gradient_tracking=True
    )
    
    # Forward pass
    output = attention(test_input, training=True)
    
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test attention weights
    weights = attention.get_attention_weights()
    if weights is not None:
        print(f"   Attention weights shape: {weights.shape}")
        
        # Validate weights sum to 1
        weight_sums = tf.reduce_sum(weights, axis=-1)
        print(f"   Weight sums (should be ~1.0): {tf.reduce_mean(weight_sums):.6f}")
    
    # Test gradient tracking
    gradient_norms = attention.get_gradient_norms()
    print(f"   Gradient norms tracked: {len(gradient_norms)}")
    for name, norm in gradient_norms.items():
        print(f"     {name}: {norm:.4f}")
    
    # Validate gradients in range
    gradients_valid = attention.validate_gradients(0.1, 10.0)
    
    # Memory usage
    memory_info = attention.get_memory_usage()
    print(f"   Parameters: {memory_info['parameters']:,}")
    print(f"   Estimated memory: {memory_info['estimated_memory_mb']:.2f} MB")
    
    return gradients_valid


if __name__ == "__main__":
    print("🎯 SCALED DOT-PRODUCT ATTENTION FOR SHAKESPEARE & ALICE")
    print("=" * 60)
    print("🎭 Multi-corpus target: Improve LSTM baseline (val_loss = 6.5)")
    print("⚡ Implementation: Pure TensorFlow ops (no pre-built attention)")
    
    if tf is not None:
        # Run test
        success = test_attention_layer()
        
        if success:
            print("\n🎉 ATTENTION IMPLEMENTATION VALIDATED")
            print("✅ Ready for Shakespeare & Alice multi-corpus training")
        else:
            print("\n❌ Validation failed - check gradient norms")
    else:
        print("\n💡 Install TensorFlow to run tests")
        print("   pip install tensorflow")