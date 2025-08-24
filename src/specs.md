# Task: Implement Scaled Dot-Product Attention

## Context
- Building attention mechanism for small Shakespeare dataset
- Current LSTM baseline: val_loss = 6.5
- Need explicit shape tracking and gradient flow analysis

## Requirements
1. Implement attention WITHOUT tf.keras.layers.MultiHeadAttention
2. Use only tf.matmul, tf.nn.softmax, basic ops
3. Add shape assertions after each operation
4. Include gradient flow visualization hooks

## Implementation Steps
- [ ] Create query, key, value projections (separate weights)
- [ ] Compute attention scores with scaling
- [ ] Apply causal mask for autoregressive generation  
- [ ] Add dropout AFTER softmax (not before)
- [ ] Test with sequence_length=128, d_model=256

## Validation
- Attention weights should sum to 1.0 per query
- Gradient norm should stay in [0.1, 10] range
- Compare memory usage vs LSTM baseline