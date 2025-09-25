"""
Attention Mechanisms Module
Shakespeare & Alice Multi-Corpus Implementation

Contains custom attention implementations built from basic TensorFlow operations
for educational purposes and detailed gradient analysis.
"""

# Make attention classes available at module level
try:
    from .scaled_dot_product_attention import ScaledDotProductAttention, create_attention_model, test_attention_layer
    print("[OK] Scaled Dot-Product Attention module loaded")
except ImportError as e:
    print(f"WARNING: Attention module import warning: {e}")
    # Define empty classes for graceful degradation
    class ScaledDotProductAttention:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow required for attention mechanisms")
    
    def create_attention_model(*args, **kwargs):
        raise ImportError("TensorFlow required for attention mechanisms")
    
    def test_attention_layer():
        print("[X] TensorFlow not available - cannot test attention layer")
        return False

__all__ = [
    'ScaledDotProductAttention',
    'create_attention_model', 
    'test_attention_layer'
]