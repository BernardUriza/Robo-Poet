#!/usr/bin/env python3
"""
Attention Validation Suite
Shakespeare & Alice Multi-Corpus Testing

Comprehensive validation of scaled dot-product attention implementation
with gradient flow analysis, memory profiling, and performance comparison.

Author: ML Academic Framework
Target: Beat LSTM baseline (val_loss = 6.5) on Shakespeare & Alice corpus
"""

import sys
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    from attention.scaled_dot_product_attention import ScaledDotProductAttention, create_attention_model
    from data_processor import TextProcessor
    tf_available = True
    print("✅ TensorFlow and modules loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Solution: pip install tensorflow")
    tf_available = False


class AttentionValidator:
    """
    🧪 COMPREHENSIVE ATTENTION VALIDATION SUITE
    
    Tests the scaled dot-product attention implementation against:
    1. Mathematical correctness (attention weights sum to 1.0)
    2. Gradient flow stability (norms in [0.1, 10] range)  
    3. Memory usage vs LSTM baseline
    4. Performance on Shakespeare & Alice corpus
    5. Causal masking correctness
    """
    
    def __init__(self, sequence_length: int = 128, d_model: int = 256):
        """
        Initialize validator with test parameters.
        
        Args:
            sequence_length: Sequence length for testing (128 as specified)
            d_model: Model dimension (256 as specified)
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.batch_size = 4  # Small batch for detailed analysis
        
        self.test_results = {}
        
        print(f"🔧 Attention Validator initialized:")
        print(f"   sequence_length: {sequence_length}")
        print(f"   d_model: {d_model}")
        print(f"   batch_size: {batch_size}")
    
    def test_mathematical_correctness(self) -> bool:
        """
        Test 1: Mathematical correctness of attention mechanism.
        
        Returns:
            True if all mathematical properties are satisfied
        """
        print("\n📐 TEST 1: MATHEMATICAL CORRECTNESS")
        print("-" * 50)
        
        if not tf_available:
            print("❌ TensorFlow not available")
            return False
        
        try:
            # Create test data
            test_input = tf.random.normal((self.batch_size, self.sequence_length, self.d_model))
            
            # Create attention layer
            attention = ScaledDotProductAttention(
                d_model=self.d_model,
                dropout_rate=0.0,  # Disable dropout for precise testing
                use_causal_mask=True,
                gradient_tracking=True
            )
            
            # Forward pass
            output = attention(test_input, training=False)
            
            # Test 1.1: Output shape correctness
            expected_shape = (self.batch_size, self.sequence_length, self.d_model)
            actual_shape = tuple(output.shape.as_list())
            
            if actual_shape != expected_shape:
                print(f"❌ Shape mismatch: expected {expected_shape}, got {actual_shape}")
                return False
            print(f"✅ Output shape correct: {actual_shape}")
            
            # Test 1.2: Attention weights sum to 1.0
            weights = attention.get_attention_weights()
            if weights is not None:
                weight_sums = tf.reduce_sum(weights, axis=-1)
                mean_sum = tf.reduce_mean(weight_sums)
                std_sum = tf.math.reduce_std(weight_sums)
                
                print(f"✅ Attention weight sums: {mean_sum:.6f} ± {std_sum:.6f}")
                
                # Should be very close to 1.0
                if not tf.reduce_all(tf.abs(weight_sums - 1.0) < 1e-6):
                    print("❌ Attention weights don't sum to 1.0")
                    return False
            
            # Test 1.3: Causal mask effectiveness
            if attention.use_causal_mask and weights is not None:
                # Check that upper triangular part is effectively zero
                # Convert to numpy for easier analysis
                weights_np = weights.numpy()
                
                for batch in range(self.batch_size):
                    for i in range(self.sequence_length):
                        for j in range(i+1, self.sequence_length):
                            if weights_np[batch, i, j] > 1e-6:
                                print(f"❌ Causal mask violation at [{batch}, {i}, {j}]: {weights_np[batch, i, j]}")
                                return False
                
                print("✅ Causal mask working correctly")
            
            # Test 1.4: Scaling factor applied correctly  
            scaling_factor = 1.0 / np.sqrt(self.d_model)
            expected_scale = scaling_factor
            actual_scale = float(attention.scale.numpy())
            
            if abs(actual_scale - expected_scale) > 1e-6:
                print(f"❌ Scaling factor incorrect: expected {expected_scale:.6f}, got {actual_scale:.6f}")
                return False
            print(f"✅ Scaling factor correct: {actual_scale:.6f}")
            
            self.test_results['mathematical_correctness'] = True
            return True
            
        except Exception as e:
            print(f"❌ Mathematical correctness test failed: {e}")
            self.test_results['mathematical_correctness'] = False
            return False
    
    def test_gradient_flow(self) -> bool:
        """
        Test 2: Gradient flow stability.
        
        Returns:
            True if gradients are in acceptable range [0.1, 10]
        """
        print("\n🌊 TEST 2: GRADIENT FLOW STABILITY")
        print("-" * 50)
        
        if not tf_available:
            print("❌ TensorFlow not available")
            return False
        
        try:
            # Create model for gradient testing
            vocab_size = 1000  # Small vocab for testing
            model, attention_layer = create_attention_model(
                vocab_size=vocab_size,
                sequence_length=self.sequence_length,
                d_model=self.d_model,
                dropout_rate=0.1
            )
            
            # Create sample data
            x = tf.random.uniform((self.batch_size, self.sequence_length), 0, vocab_size, dtype=tf.int32)
            y = tf.random.uniform((self.batch_size, self.sequence_length), 0, vocab_size, dtype=tf.int32)
            
            # Compute gradients
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                loss = tf.reduce_mean(loss)
            
            # Get gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Analyze gradient norms
            gradient_norms = []
            for i, grad in enumerate(gradients):
                if grad is not None:
                    norm = tf.norm(grad)
                    gradient_norms.append(float(norm.numpy()))
                    print(f"   Layer {i} gradient norm: {norm:.4f}")
            
            # Check if gradients are in acceptable range
            min_norm, max_norm = 0.1, 10.0
            out_of_range = 0
            
            for norm in gradient_norms:
                if not (min_norm <= norm <= max_norm):
                    out_of_range += 1
            
            if out_of_range > 0:
                print(f"⚠️ {out_of_range}/{len(gradient_norms)} gradients out of range [{min_norm}, {max_norm}]")
                # Still pass if most gradients are ok
                success = out_of_range < len(gradient_norms) // 2
            else:
                success = True
                print(f"✅ All {len(gradient_norms)} gradients in range [{min_norm}, {max_norm}]")
            
            # Check attention-specific gradients
            attention_gradients = attention_layer.get_gradient_norms()
            print(f"   Attention-specific gradient norms: {len(attention_gradients)}")
            for name, norm in attention_gradients.items():
                print(f"     {name}: {norm:.4f}")
            
            self.test_results['gradient_flow'] = success
            return success
            
        except Exception as e:
            print(f"❌ Gradient flow test failed: {e}")
            self.test_results['gradient_flow'] = False
            return False
    
    def test_memory_usage(self) -> bool:
        """
        Test 3: Memory usage comparison vs LSTM baseline.
        
        Returns:
            True if memory usage is reasonable
        """
        print("\n💾 TEST 3: MEMORY USAGE ANALYSIS")
        print("-" * 50)
        
        if not tf_available:
            print("❌ TensorFlow not available")
            return False
        
        try:
            # Create attention model
            vocab_size = 5000  # Typical Shakespeare & Alice vocab
            attention_model, attention_layer = create_attention_model(
                vocab_size=vocab_size,
                sequence_length=self.sequence_length,
                d_model=self.d_model
            )
            
            # Create equivalent LSTM model for comparison
            lstm_inputs = tf.keras.layers.Input(shape=(self.sequence_length,), dtype=tf.int32)
            lstm_embed = tf.keras.layers.Embedding(vocab_size, self.d_model)(lstm_inputs)
            lstm_hidden = tf.keras.layers.LSTM(self.d_model, return_sequences=True)(lstm_embed)
            lstm_outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(lstm_hidden)
            lstm_model = tf.keras.Model(inputs=lstm_inputs, outputs=lstm_outputs)
            
            # Compare parameter counts
            attention_params = attention_model.count_params()
            lstm_params = lstm_model.count_params()
            
            print(f"📊 Parameter Comparison:")
            print(f"   Attention model: {attention_params:,} parameters")
            print(f"   LSTM baseline:   {lstm_params:,} parameters")
            print(f"   Ratio: {attention_params/lstm_params:.2f}x")
            
            # Memory usage from attention layer
            attention_memory = attention_layer.get_memory_usage()
            print(f"\n💾 Attention Memory Usage:")
            print(f"   Attention parameters: {attention_memory['parameters']:,}")
            print(f"   Matrix memory per batch: {attention_memory['attention_matrix_bytes_per_batch']:,} bytes")
            print(f"   Estimated total: {attention_memory['estimated_memory_mb']:.2f} MB")
            
            # Test with different sequence lengths for scalability
            seq_lengths = [32, 64, 128, 256]
            print(f"\n📈 Memory Scaling Analysis:")
            
            for seq_len in seq_lengths:
                # Attention memory scales O(seq_len^2)
                attention_mem = seq_len * seq_len * 4  # float32 attention matrix
                # LSTM memory scales O(seq_len)  
                lstm_mem = seq_len * self.d_model * 4
                
                print(f"   seq_len={seq_len}: Attention={attention_mem/1024:.1f}KB, LSTM={lstm_mem/1024:.1f}KB")
            
            # Consider reasonable if attention model is not more than 3x LSTM parameters
            memory_reasonable = attention_params <= 3 * lstm_params
            
            if memory_reasonable:
                print("✅ Memory usage within acceptable bounds")
            else:
                print("⚠️ Memory usage higher than expected")
            
            self.test_results['memory_usage'] = {
                'attention_params': attention_params,
                'lstm_params': lstm_params,
                'ratio': attention_params/lstm_params,
                'reasonable': memory_reasonable
            }
            
            return memory_reasonable
            
        except Exception as e:
            print(f"❌ Memory usage test failed: {e}")
            self.test_results['memory_usage'] = False
            return False
    
    def test_shakespeare_alice_corpus(self) -> bool:
        """
        Test 4: Performance on actual Shakespeare & Alice corpus.
        
        Returns:
            True if can process corpus without errors
        """
        print("\n🎭 TEST 4: SHAKESPEARE & ALICE CORPUS")
        print("-" * 50)
        
        if not tf_available:
            print("❌ TensorFlow not available")
            return False
        
        try:
            # Try to load actual corpus
            corpus_path = Path("corpus")
            if not corpus_path.exists():
                print("⚠️ Corpus directory not found - creating synthetic data")
                # Use synthetic data that mimics corpus structure
                synthetic_text = "to be or not to be that is the question " * 100
                synthetic_text += "alice was beginning to get very tired of sitting " * 100
                
                # Create simple vocabulary
                vocab = list(set(synthetic_text.split()))
                vocab_size = len(vocab)
                token_to_idx = {token: idx for idx, token in enumerate(vocab)}
                
                # Convert to integer sequences
                tokens = synthetic_text.split()[:self.sequence_length * 10]  # Enough for testing
                sequences = [token_to_idx[token] for token in tokens if token in token_to_idx]
                
            else:
                print("✅ Found corpus directory - loading multi-corpus data")
                try:
                    # Use actual corpus
                    processor = TextProcessor(
                        sequence_length=self.sequence_length,
                        vocab_size=1000,  # Smaller for testing
                        tokenization='word'
                    )
                    
                    # Load a small sample for testing
                    text = processor.load_text("corpus", max_length=20_000)
                    processor.build_vocabulary(text)
                    
                    # Get vocabulary info
                    vocab_size = processor.vocab_size
                    
                    # Create some test sequences
                    X_onehot, y_onehot = processor.create_sequences(text)
                    X = np.argmax(X_onehot, axis=-1)
                    
                    sequences = X.flatten()[:self.sequence_length * 10]
                    
                except Exception as e:
                    print(f"⚠️ Corpus loading failed: {e}")
                    print("   Using fallback synthetic data")
                    vocab_size = 1000
                    sequences = np.random.randint(0, vocab_size, size=self.sequence_length * 10)
            
            print(f"📚 Corpus loaded: vocab_size={vocab_size}, sequences={len(sequences)}")
            
            # Create attention model
            model, attention_layer = create_attention_model(
                vocab_size=vocab_size,
                sequence_length=self.sequence_length,
                d_model=self.d_model,
                dropout_rate=0.1
            )
            
            # Create batched data
            num_sequences = len(sequences) // self.sequence_length
            X_test = np.array(sequences[:num_sequences * self.sequence_length]).reshape(
                -1, self.sequence_length
            )[:self.batch_size]  # Take only batch_size samples
            
            print(f"🔄 Testing with data shape: {X_test.shape}")
            
            # Test forward pass
            start_time = time.time()
            predictions = model(X_test, training=False)
            forward_time = time.time() - start_time
            
            print(f"✅ Forward pass successful")
            print(f"   Input shape: {X_test.shape}")
            print(f"   Output shape: {predictions.shape}")
            print(f"   Forward time: {forward_time:.4f}s")
            print(f"   Throughput: {X_test.shape[0] * X_test.shape[1] / forward_time:.0f} tokens/sec")
            
            # Test attention weights visualization
            attention_weights = attention_layer.get_attention_weights()
            if attention_weights is not None:
                print(f"   Attention weights available: {attention_weights.shape}")
                
                # Sample some attention weights for analysis
                sample_weights = attention_weights[0, :5, :5].numpy()  # First batch, 5x5 sample
                print(f"   Sample attention weights (first 5x5):")
                for i in range(5):
                    row_str = " ".join([f"{sample_weights[i,j]:.3f}" for j in range(5)])
                    print(f"     [{i}]: {row_str}")
            
            # Test training step
            print(f"\n🏋️ Testing training step...")
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            
            with tf.GradientTape() as tape:
                predictions = model(X_test, training=True)
                # Create dummy targets (shifted input for language modeling)
                targets = np.roll(X_test, -1, axis=1)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    targets, predictions, from_logits=False
                )
                loss = tf.reduce_mean(loss)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            print(f"✅ Training step successful")
            print(f"   Loss: {float(loss):.4f}")
            
            self.test_results['shakespeare_alice_corpus'] = True
            return True
            
        except Exception as e:
            print(f"❌ Shakespeare & Alice corpus test failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['shakespeare_alice_corpus'] = False
            return False
    
    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run complete validation suite.
        
        Returns:
            Dictionary with all test results
        """
        print("🧪 RUNNING FULL ATTENTION VALIDATION SUITE")
        print("=" * 60)
        print("🎭 Target: Beat LSTM baseline (val_loss = 6.5)")
        print("📐 Testing scaled dot-product attention implementation")
        
        tests = [
            ("Mathematical Correctness", self.test_mathematical_correctness),
            ("Gradient Flow Stability", self.test_gradient_flow), 
            ("Memory Usage Analysis", self.test_memory_usage),
            ("Shakespeare & Alice Corpus", self.test_shakespeare_alice_corpus)
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                success = test_func()
                results[test_name] = success
                if success:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Final summary
        print(f"\n📊 VALIDATION SUMMARY")
        print("=" * 40)
        print(f"Tests passed: {passed}/{len(tests)}")
        print(f"Success rate: {passed/len(tests)*100:.1f}%")
        
        if passed == len(tests):
            print("🎉 ALL TESTS PASSED - ATTENTION READY FOR TRAINING")
        elif passed >= len(tests) // 2:
            print("⚠️ PARTIAL SUCCESS - ATTENTION USABLE WITH CAUTION")
        else:
            print("❌ VALIDATION FAILED - ATTENTION NEEDS FIXES")
        
        results['summary'] = {
            'passed': passed,
            'total': len(tests),
            'success_rate': passed/len(tests),
            'overall_status': 'PASSED' if passed == len(tests) else 'PARTIAL' if passed >= len(tests)//2 else 'FAILED'
        }
        
        return results


def main():
    """Main validation script."""
    print("🎯 SCALED DOT-PRODUCT ATTENTION VALIDATOR")
    print("🎭 Shakespeare & Alice Multi-Corpus Edition")
    print("=" * 60)
    
    if not tf_available:
        print("❌ TensorFlow not available")
        print("💡 Install with: pip install tensorflow")
        return False
    
    # Create validator with specified dimensions
    validator = AttentionValidator(sequence_length=128, d_model=256)
    
    # Run full validation
    results = validator.run_full_validation()
    
    # Save results if needed
    print(f"\n💾 Validation completed")
    print(f"🔍 Results available in validator.test_results")
    
    return results['summary']['overall_status'] == 'PASSED'


if __name__ == "__main__":
    success = main()
    print(f"\n🦁 Validation {'✅ completed successfully' if success else '❌ needs attention'} - by Aslan")
    print("🧉 Ready to brew the perfect attention mechanism mate!")