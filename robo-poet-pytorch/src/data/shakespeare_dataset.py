"""
PyTorch Dataset for Shakespeare + Alice corpus with document markers.
Created by Bernard Orozco - TensorFlow to PyTorch Migration
"""

import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for Shakespeare + Alice corpus with unified preprocessing.
    
    Features:
    - Character-level tokenization 
    - Document markers for style control (<|startdoc|>Shakespeare|...|<|content|>)
    - Sliding window sequences (context_length=128)
    - Train/Val/Test splits preservation
    """
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "train",
        context_length: int = 128,
        stride: int = 1,
        device: torch.device = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to processed data directory
            split: 'train', 'validation', or 'test'  
            context_length: Sequence length for training
            stride: Stride for sliding window
            device: Target device for tensors
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.context_length = context_length
        self.stride = stride
        self.device = device or torch.device('cpu')
        
        # Load vocabulary and metadata
        self._load_vocabulary()
        self._load_metadata()
        
        # Load and prepare text data
        self._load_text_data()
        self._create_sequences()
        
        print(f"📚 {self.split.upper()} dataset initialized:")
        print(f"   📖 Text length: {len(self.text):,} characters")
        print(f"   🔢 Vocabulary size: {self.vocab_size}")
        print(f"   📝 Sequences: {len(self.sequences):,}")
        print(f"   🎯 Context length: {self.context_length}")
    
    def _load_vocabulary(self) -> None:
        """Load vocabulary mappings from JSON."""
        # Try character-level vocabulary first
        vocab_path = self.data_dir / "char_vocabulary.json"
        
        if not vocab_path.exists():
            # Fallback to creating vocabulary from text
            print("⚠️  Character vocabulary not found, creating from text...")
            self._create_vocabulary_from_text()
            return
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        self.vocab_size = vocab_data['vocab_size']
        
        print(f"🔤 Vocabulary loaded: {self.vocab_size} tokens")
    
    def _create_vocabulary_from_text(self) -> None:
        """Create character vocabulary from text if not exists."""
        # Load text first
        text_path = self.data_dir / "processed" / "simple_corpus.txt"
        if not text_path.exists():
            raise FileNotFoundError(f"Cannot create vocabulary: {text_path} not found")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Get unique characters
        unique_chars = sorted(list(set(text)))
        
        # Create mappings
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Add special tokens first
        special_tokens = ['<|PAD|>', '<|UNK|>', '<|START|>', '<|END|>']
        for i, token in enumerate(special_tokens):
            self.char_to_idx[token] = i
            self.idx_to_char[i] = token
        
        # Add regular characters
        start_idx = len(special_tokens)
        for i, char in enumerate(unique_chars):
            self.char_to_idx[char] = start_idx + i
            self.idx_to_char[start_idx + i] = char
        
        self.vocab_size = len(self.char_to_idx)
        print(f"🔤 Created vocabulary from text: {self.vocab_size} tokens")
    
    def _load_metadata(self) -> None:
        """Load dataset metadata."""
        metadata_path = self.data_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            print("⚠️  No metadata found, using defaults")
    
    def _load_text_data(self) -> None:
        """Load text data for specified split."""
        if self.split == "train":
            # Use unified corpus for training
            text_path = self.data_dir / "processed" / "simple_corpus.txt"
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    self.text = f.read()
                print(f"📖 Loaded simple corpus: {len(self.text):,} chars")
            else:
                # Fallback to splits
                text_path = self.data_dir / "splits" / "train.txt"
                with open(text_path, 'r', encoding='utf-8') as f:
                    self.text = f.read()
                print(f"📖 Loaded train split: {len(self.text):,} chars")
        else:
            # Use specific splits for val/test
            text_path = self.data_dir / "splits" / f"{self.split}.txt"
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    self.text = f.read()
                print(f"📖 Loaded {self.split} split: {len(self.text):,} chars")
            else:
                # Create dummy data if split doesn't exist
                self.text = "To be or not to be, that is the question."
                print(f"⚠️  No {self.split} data found, using dummy text")
    
    def _create_sequences(self) -> None:
        """Create sliding window sequences from text."""
        # Convert text to indices
        self.encoded_text = self._encode_text(self.text)
        
        # Create sequences with sliding window
        self.sequences = []
        
        for i in range(0, len(self.encoded_text) - self.context_length, self.stride):
            # Input sequence (context_length tokens)
            input_seq = self.encoded_text[i:i + self.context_length]
            # Target sequence (shifted by 1 for next-token prediction)
            target_seq = self.encoded_text[i + 1:i + self.context_length + 1]
            
            if len(input_seq) == self.context_length and len(target_seq) == self.context_length:
                self.sequences.append((input_seq, target_seq))
        
        print(f"🔢 Created {len(self.sequences):,} sequences")
    
    def _encode_text(self, text: str) -> List[int]:
        """Convert text to list of token indices."""
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # Use UNK token for unknown characters
                encoded.append(self.char_to_idx.get('<|UNK|>', 0))
        return encoded
    
    def _decode_indices(self, indices: List[int]) -> str:
        """Convert list of indices back to text."""
        return ''.join(self.idx_to_char.get(idx, '<|UNK|>') for idx in indices)
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of sequence to retrieve
        
        Returns:
            Tuple of (input_tensor, target_tensor)
            Both tensors have shape [context_length]
        """
        input_seq, target_seq = self.sequences[idx]
        
        # Convert to tensors (always on CPU to avoid multiprocessing CUDA issues)
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)
        
        return input_tensor, target_tensor
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text string to tensor."""
        indices = self._encode_text(text)
        return torch.tensor(indices, dtype=torch.long, device=self.device)
    
    def decode(self, tensor: torch.Tensor) -> str:
        """Decode tensor to text string."""
        if tensor.dim() > 1:
            tensor = tensor.flatten()
        indices = tensor.cpu().tolist()
        return self._decode_indices(indices)
    
    def get_sample(self, idx: int = 0) -> Dict[str, str]:
        """
        Get a sample sequence for inspection.
        
        Args:
            idx: Index of sequence to sample
            
        Returns:
            Dict with 'input' and 'target' text
        """
        if idx >= len(self.sequences):
            idx = 0
        
        input_seq, target_seq = self.sequences[idx]
        
        return {
            'input': self._decode_indices(input_seq),
            'target': self._decode_indices(target_seq),
            'input_length': len(input_seq),
            'target_length': len(target_seq)
        }


def create_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 32,
    context_length: int = 128,
    num_workers: int = 4,
    device: torch.device = None
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for training
        context_length: Sequence length
        num_workers: Number of worker processes for data loading
        device: Target device for tensors
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = ShakespeareDataset(
        data_dir=data_dir,
        split="train",
        context_length=context_length,
        stride=context_length // 2,  # 50% overlap for more training examples
        device=device
    )
    
    val_dataset = ShakespeareDataset(
        data_dir=data_dir,
        split="validation", 
        context_length=context_length,
        stride=context_length,  # No overlap for validation
        device=device
    )
    
    test_dataset = ShakespeareDataset(
        data_dir=data_dir,
        split="test",
        context_length=context_length,
        stride=context_length,  # No overlap for testing
        device=device
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device and device.type == 'cuda' else False,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device and device.type == 'cuda' else False,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device and device.type == 'cuda' else False,
        drop_last=False
    )
    
    print(f"🚀 DataLoaders created:")
    print(f"   📊 Train batches: {len(train_loader)}")
    print(f"   📊 Val batches: {len(val_loader)}")
    print(f"   📊 Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("🧪 Testing ShakespeareDataset...")
    
    # Test dataset creation
    dataset = ShakespeareDataset(
        data_dir="../../data/processed",
        split="train",
        context_length=64
    )
    
    # Test sample retrieval
    sample = dataset.get_sample(0)
    print(f"\n📝 Sample input (first 100 chars): {sample['input'][:100]}...")
    print(f"📝 Sample target (first 100 chars): {sample['target'][:100]}...")
    
    # Test dataloader
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="../../data/processed",
        batch_size=4,
        context_length=64
    )
    
    # Test batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"\n🎯 Batch {batch_idx}:")
        print(f"   Input shape: {inputs.shape}")
        print(f"   Target shape: {targets.shape}")
        print(f"   Input sample: {dataset.decode(inputs[0][:20])}...")
        break
    
    print("\n✅ ShakespeareDataset test completed!")