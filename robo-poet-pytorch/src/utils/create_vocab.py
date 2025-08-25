"""
Create character-level vocabulary for PyTorch implementation.
Created by Bernard Orozco - TensorFlow to PyTorch Migration
"""

import json
from pathlib import Path
from typing import Dict, Tuple
from collections import Counter


def create_character_vocabulary(text_path: str, output_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create character-level vocabulary from text file.
    
    Args:
        text_path: Path to unified corpus text file
        output_path: Path to save vocabulary JSON
        
    Returns:
        Tuple of (char_to_idx, idx_to_char) mappings
    """
    print(f"📚 Creating character vocabulary from {text_path}")
    
    # Read text
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"📖 Text length: {len(text):,} characters")
    
    # Get unique characters
    unique_chars = sorted(list(set(text)))
    print(f"🔤 Unique characters: {len(unique_chars)}")
    
    # Create mappings (starting from index 0)
    char_to_idx = {}
    idx_to_char = {}
    
    # Add special tokens first
    special_tokens = ['<|PAD|>', '<|UNK|>', '<|START|>', '<|END|>']
    for i, token in enumerate(special_tokens):
        char_to_idx[token] = i
        idx_to_char[i] = token
    
    # Add regular characters
    start_idx = len(special_tokens)
    for i, char in enumerate(unique_chars):
        char_to_idx[char] = start_idx + i
        idx_to_char[start_idx + i] = char
    
    vocab_size = len(char_to_idx)
    print(f"🎯 Total vocabulary size: {vocab_size}")
    
    # Count character frequencies for analysis
    char_counts = Counter(text)
    top_chars = char_counts.most_common(10)
    print(f"📊 Most frequent characters:")
    for char, count in top_chars:
        char_repr = repr(char) if char in ['\n', '\t', ' '] else char
        print(f"   {char_repr}: {count:,}")
    
    # Create vocabulary data structure
    vocab_data = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
        'special_tokens': special_tokens,
        'text_length': len(text),
        'unique_chars': len(unique_chars)
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Vocabulary saved to {output_path}")
    
    return char_to_idx, idx_to_char


if __name__ == "__main__":
    # Create vocabulary from unified corpus
    text_path = "../data/processed/unified_corpus.txt"
    vocab_path = "../data/processed/char_vocabulary.json"
    
    char_to_idx, idx_to_char = create_character_vocabulary(text_path, vocab_path)
    
    # Test the vocabulary
    print(f"\n🧪 Testing vocabulary:")
    test_text = "To be or not to be"
    encoded = [char_to_idx.get(c, char_to_idx['<|UNK|>']) for c in test_text]
    decoded = ''.join(idx_to_char[i] for i in encoded)
    
    print(f"   Original: {test_text}")
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: {decoded}")
    print(f"   Match: {test_text == decoded}")
    
    print("\n✅ Character vocabulary creation completed!")