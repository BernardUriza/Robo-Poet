"""
High-performance streaming data pipeline with tf.data for large text corpora.

Implements efficient data loading, batching, and preprocessing for training
language models with optimal memory usage and GPU utilization.
"""

import os
import logging
import math
import time
from typing import Iterator, List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for streaming dataset."""
    
    # File and corpus settings
    corpus_path: str
    vocab_size: int = 10000
    sequence_length: int = 128
    
    # Batch settings  
    batch_size: int = 32
    buffer_size: int = 10000  # For shuffling
    prefetch_buffer: int = tf.data.AUTOTUNE
    
    # Performance settings
    num_parallel_calls: int = tf.data.AUTOTUNE
    num_parallel_reads: int = 4
    interleave_cycle_length: int = 8
    
    # Memory optimization
    cache_dataset: bool = False
    cache_filename: Optional[str] = None
    map_num_parallel_calls: int = tf.data.AUTOTUNE
    
    # Training settings
    repeat_dataset: bool = True
    shuffle_seed: Optional[int] = None
    drop_remainder: bool = True  # For consistent batch sizes
    
    # Text processing
    lowercase: bool = True
    remove_punctuation: bool = False
    min_sequence_length: int = 10
    max_file_size_mb: int = 1000  # Skip very large files
    
    # Streaming settings
    chunk_size: int = 8192  # Bytes per chunk
    overlap_tokens: int = 64   # Token overlap between chunks
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.vocab_size > 0
        assert self.sequence_length > 0
        assert self.batch_size > 0
        assert self.buffer_size > 0
        assert self.min_sequence_length < self.sequence_length
        assert self.chunk_size > 0
        assert 0 <= self.overlap_tokens < self.sequence_length


class TextDataset:
    """High-performance text dataset with streaming capabilities."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.config.validate()
        
        self.tokenizer = None
        self.dataset = None
        self.file_paths = []
        self.total_files = 0
        self.total_size_mb = 0
        
        # Performance monitoring
        self.load_time = 0
        self.preprocessing_time = 0
        self.samples_processed = 0
        
        logger.info(f"TextDataset initialized with sequence_length={config.sequence_length}")
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for text processing."""
        self.tokenizer = tokenizer
        
        # Update vocab_size from tokenizer if available
        if hasattr(tokenizer, 'vocab_size'):
            self.config.vocab_size = tokenizer.vocab_size
        
        logger.info(f"Tokenizer set with vocab_size={self.config.vocab_size}")
    
    def discover_files(self, corpus_path: str) -> List[str]:
        """
        Discover text files in corpus directory.
        
        Args:
            corpus_path: Path to corpus directory or file
            
        Returns:
            List of discovered file paths
        """
        start_time = time.time()
        
        corpus_path = Path(corpus_path)
        file_paths = []
        
        if corpus_path.is_file():
            # Single file
            if self._is_text_file(corpus_path):
                file_paths.append(str(corpus_path))
        else:
            # Directory - find all text files
            text_extensions = {'.txt', '.text', '.log', '.md'}
            
            for ext in text_extensions:
                pattern = f"**/*{ext}"
                found_files = list(corpus_path.glob(pattern))
                
                for file_path in found_files:
                    if self._is_valid_file(file_path):
                        file_paths.append(str(file_path))
        
        # Sort for consistent ordering
        file_paths.sort()
        
        # Calculate total size
        total_size = sum(Path(fp).stat().st_size for fp in file_paths)
        self.total_size_mb = total_size / (1024 * 1024)
        
        discovery_time = time.time() - start_time
        
        logger.info(f"Discovered {len(file_paths)} files ({self.total_size_mb:.1f}MB) in {discovery_time:.2f}s")
        
        self.file_paths = file_paths
        self.total_files = len(file_paths)
        
        return file_paths
    
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file."""
        try:
            # Check extension
            if file_path.suffix.lower() not in {'.txt', '.text', '.log', '.md'}:
                return False
            
            # Try to read first few bytes to detect encoding
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                
            # Simple heuristic - if mostly printable ASCII/UTF-8
            try:
                sample.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
                
        except Exception:
            return False
    
    def _is_valid_file(self, file_path: Path) -> bool:
        """Check if file is valid for processing."""
        try:
            stat = file_path.stat()
            
            # Check file size
            size_mb = stat.st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                logger.warning(f"Skipping large file: {file_path} ({size_mb:.1f}MB)")
                return False
            
            # Check if readable
            if not file_path.is_file():
                return False
                
            return self._is_text_file(file_path)
            
        except Exception as e:
            logger.debug(f"Invalid file {file_path}: {e}")
            return False
    
    def create_dataset(self) -> tf.data.Dataset:
        """
        Create optimized tf.data.Dataset for training.
        
        Returns:
            Configured tf.data.Dataset
        """
        start_time = time.time()
        
        if not self.file_paths:
            self.discover_files(self.config.corpus_path)
        
        if not self.file_paths:
            raise ValueError(f"No valid text files found in {self.config.corpus_path}")
        
        # Create dataset from file paths
        file_dataset = tf.data.Dataset.from_tensor_slices(self.file_paths)
        
        # Shuffle files if requested
        if self.config.shuffle_seed is not None:
            file_dataset = file_dataset.shuffle(
                buffer_size=len(self.file_paths),
                seed=self.config.shuffle_seed
            )
        
        # Interleave file reading for better I/O performance
        dataset = file_dataset.interleave(
            lambda filename: self._create_text_dataset_from_file(filename),
            cycle_length=self.config.interleave_cycle_length,
            num_parallel_calls=self.config.num_parallel_reads,
            deterministic=False  # Better performance
        )
        
        # Tokenize and create sequences
        dataset = dataset.map(
            self._process_text_chunk,
            num_parallel_calls=self.config.map_num_parallel_calls
        )
        
        # Flatten nested sequences  
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        
        # Filter sequences that are too short
        dataset = dataset.filter(
            lambda seq: tf.shape(seq)[0] >= self.config.min_sequence_length
        )
        
        # Cache if requested
        if self.config.cache_dataset:
            if self.config.cache_filename:
                dataset = dataset.cache(self.config.cache_filename)
            else:
                dataset = dataset.cache()
        
        # Shuffle sequences
        dataset = dataset.shuffle(
            buffer_size=self.config.buffer_size,
            seed=self.config.shuffle_seed
        )
        
        # Create input-target pairs for language modeling
        dataset = dataset.map(
            self._create_input_target_pairs,
            num_parallel_calls=self.config.map_num_parallel_calls
        )
        
        # Batch
        dataset = dataset.batch(
            self.config.batch_size,
            drop_remainder=self.config.drop_remainder
        )
        
        # Prefetch for performance
        dataset = dataset.prefetch(self.config.prefetch_buffer)
        
        # Repeat if requested
        if self.config.repeat_dataset:
            dataset = dataset.repeat()
        
        self.dataset = dataset
        self.load_time = time.time() - start_time
        
        logger.info(f"Dataset created in {self.load_time:.2f}s")
        logger.info(f"Expected samples per epoch: ~{self._estimate_samples_per_epoch()}")
        
        return dataset
    
    def _create_text_dataset_from_file(self, filename: tf.Tensor) -> tf.data.Dataset:
        """Create dataset from a single file with chunking."""
        
        def read_file_chunks():
            """Generator that yields text chunks from file."""
            filename_str = filename.numpy().decode('utf-8')
            
            try:
                with open(filename_str, 'r', encoding='utf-8', errors='ignore') as f:
                    chunk_overlap = ""
                    
                    while True:
                        chunk = f.read(self.config.chunk_size)
                        if not chunk:
                            # End of file
                            if chunk_overlap.strip():
                                yield chunk_overlap
                            break
                        
                        # Combine with previous overlap
                        full_chunk = chunk_overlap + chunk
                        
                        # Find good split point (word boundary)
                        if len(full_chunk) > self.config.chunk_size:
                            # Find last space before chunk_size
                            split_point = full_chunk.rfind(' ', 0, self.config.chunk_size)
                            if split_point == -1:
                                split_point = self.config.chunk_size
                            
                            yield_chunk = full_chunk[:split_point]
                            chunk_overlap = full_chunk[split_point:].lstrip()
                        else:
                            yield_chunk = full_chunk
                            chunk_overlap = ""
                        
                        if yield_chunk.strip():
                            yield yield_chunk
                            
            except Exception as e:
                logger.warning(f"Error reading {filename_str}: {e}")
        
        return tf.data.Dataset.from_generator(
            lambda: read_file_chunks(),
            output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
        )
    
    def _process_text_chunk(self, text_chunk: tf.Tensor) -> tf.Tensor:
        """Process text chunk into token sequences."""
        
        def py_process_text(text_bytes):
            """Python function to process text."""
            text = text_bytes.numpy().decode('utf-8')
            
            if not self.tokenizer:
                # Fallback - simple word tokenization
                if self.config.lowercase:
                    text = text.lower()
                
                # Simple word splitting
                words = text.split()
                
                # Create dummy token IDs (would need real tokenizer)
                tokens = list(range(min(len(words), self.config.sequence_length)))
            else:
                # Use proper tokenizer
                if hasattr(self.tokenizer, 'encode'):
                    tokens = self.tokenizer.encode(text)
                else:
                    # Fallback for simple tokenizers
                    if self.config.lowercase:
                        text = text.lower()
                    words = text.split()
                    tokens = [
                        self.tokenizer.word_to_index.get(word, 1) 
                        for word in words
                    ]
            
            # Create overlapping sequences
            sequences = []
            step_size = self.config.sequence_length - self.config.overlap_tokens
            
            for i in range(0, len(tokens) - self.config.sequence_length + 1, step_size):
                sequence = tokens[i:i + self.config.sequence_length]
                
                if len(sequence) == self.config.sequence_length:
                    sequences.append(sequence)
            
            return np.array(sequences, dtype=np.int32)
        
        sequences = tf.py_function(
            py_process_text,
            [text_chunk],
            tf.int32
        )
        
        # Set shape for better optimization
        sequences.set_shape([None, self.config.sequence_length])
        
        return sequences
    
    def _create_input_target_pairs(self, sequence: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create input-target pairs for language modeling."""
        # Input: tokens[:-1], Target: tokens[1:]
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        return input_seq, target_seq
    
    def _estimate_samples_per_epoch(self) -> int:
        """Estimate number of samples per epoch."""
        if not self.total_size_mb:
            return 0
        
        # Rough estimate based on file sizes
        # Assume average of 5 characters per token, overlap factor
        chars_per_token = 5
        bytes_per_sequence = self.config.sequence_length * chars_per_token
        overlap_factor = 1.0 - (self.config.overlap_tokens / self.config.sequence_length)
        
        total_bytes = self.total_size_mb * 1024 * 1024
        estimated_sequences = int((total_bytes / bytes_per_sequence) * overlap_factor)
        
        return estimated_sequences
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            'total_files': self.total_files,
            'total_size_mb': self.total_size_mb,
            'sequence_length': self.config.sequence_length,
            'batch_size': self.config.batch_size,
            'vocab_size': self.config.vocab_size,
            'estimated_samples_per_epoch': self._estimate_samples_per_epoch(),
            'load_time': self.load_time,
            'config': {
                'buffer_size': self.config.buffer_size,
                'prefetch_buffer': self.config.prefetch_buffer,
                'num_parallel_calls': self.config.num_parallel_calls,
                'cache_dataset': self.config.cache_dataset
            }
        }


class StreamingDataLoader:
    """High-level streaming data loader with advanced features."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset_cache = {}
        self.performance_stats = {}
        
    def create_train_dataset(
        self,
        corpus_path: str,
        tokenizer=None,
        validation_split: float = 0.1
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create training and validation datasets.
        
        Args:
            corpus_path: Path to text corpus
            tokenizer: Tokenizer for text processing
            validation_split: Fraction for validation set
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create base dataset
        text_dataset = TextDataset(self.config)
        text_dataset.set_tokenizer(tokenizer)
        
        # Discover files
        file_paths = text_dataset.discover_files(corpus_path)
        
        if validation_split > 0:
            # Split files for train/validation
            num_val_files = max(1, int(len(file_paths) * validation_split))
            val_files = file_paths[:num_val_files]
            train_files = file_paths[num_val_files:]
            
            logger.info(f"Split: {len(train_files)} train files, {len(val_files)} val files")
            
            # Create separate datasets
            train_config = DatasetConfig(**self.config.__dict__)
            train_config.corpus_path = corpus_path
            train_dataset = TextDataset(train_config)
            train_dataset.set_tokenizer(tokenizer)
            train_dataset.file_paths = train_files
            train_dataset.total_files = len(train_files)
            train_ds = train_dataset.create_dataset()
            
            val_config = DatasetConfig(**self.config.__dict__)
            val_config.repeat_dataset = False  # Don't repeat validation
            val_config.corpus_path = corpus_path
            val_dataset = TextDataset(val_config)
            val_dataset.set_tokenizer(tokenizer)
            val_dataset.file_paths = val_files
            val_dataset.total_files = len(val_files)
            val_ds = val_dataset.create_dataset()
            
            return train_ds, val_ds
        else:
            # Single dataset
            dataset = text_dataset.create_dataset()
            return dataset, None
    
    def benchmark_dataset(self, dataset: tf.data.Dataset, num_batches: int = 100) -> Dict[str, float]:
        """
        Benchmark dataset performance.
        
        Args:
            dataset: Dataset to benchmark
            num_batches: Number of batches to process
            
        Returns:
            Performance statistics
        """
        logger.info(f"Benchmarking dataset performance ({num_batches} batches)...")
        
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataset.take(num_batches)):
            batch_start = time.time()
            
            # Simulate some processing
            _ = tf.reduce_sum(batch[0])  # Input
            _ = tf.reduce_sum(batch[1])  # Target
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if (i + 1) % 20 == 0:
                avg_time = np.mean(batch_times[-20:])
                logger.info(f"Batch {i+1}/{num_batches}: {avg_time*1000:.2f}ms avg")
        
        total_time = time.time() - start_time
        
        stats = {
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times),
            'min_batch_time': np.min(batch_times),
            'max_batch_time': np.max(batch_times),
            'std_batch_time': np.std(batch_times),
            'batches_per_second': num_batches / total_time,
            'samples_per_second': (num_batches * self.config.batch_size) / total_time
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Batches/sec: {stats['batches_per_second']:.2f}")
        logger.info(f"  Samples/sec: {stats['samples_per_second']:.1f}")
        logger.info(f"  Avg batch time: {stats['avg_batch_time']*1000:.2f}ms")
        
        return stats
    
    def optimize_performance(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Apply performance optimizations to dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Optimized dataset
        """
        # Options for performance
        options = tf.data.Options()
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_threading.private_threadpool_size = 8
        
        return dataset.with_options(options)


def create_streaming_dataset(
    corpus_path: str,
    sequence_length: int = 128,
    batch_size: int = 32,
    vocab_size: int = 10000,
    tokenizer=None,
    validation_split: float = 0.1,
    **kwargs
) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]:
    """
    Factory function to create streaming dataset.
    
    Args:
        corpus_path: Path to text corpus
        sequence_length: Length of input sequences
        batch_size: Batch size for training
        vocab_size: Vocabulary size
        tokenizer: Optional tokenizer
        validation_split: Validation split ratio
        **kwargs: Additional configuration options
        
    Returns:
        Dataset or tuple of (train, validation) datasets
    """
    config = DatasetConfig(
        corpus_path=corpus_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        vocab_size=vocab_size,
        **kwargs
    )
    
    loader = StreamingDataLoader(config)
    
    if validation_split > 0:
        return loader.create_train_dataset(corpus_path, tokenizer, validation_split)
    else:
        train_ds, _ = loader.create_train_dataset(corpus_path, tokenizer, 0)
        return train_ds


def benchmark_streaming_performance():
    """Benchmark streaming performance with different configurations."""
    # Test configurations
    configs = [
        {'batch_size': 16, 'prefetch_buffer': 1, 'num_parallel_calls': 1},
        {'batch_size': 32, 'prefetch_buffer': 2, 'num_parallel_calls': 2}, 
        {'batch_size': 64, 'prefetch_buffer': tf.data.AUTOTUNE, 'num_parallel_calls': tf.data.AUTOTUNE},
    ]
    
    corpus_path = "test_corpus"  # Would need actual corpus
    results = {}
    
    for i, config_overrides in enumerate(configs):
        logger.info(f"Testing configuration {i+1}: {config_overrides}")
        
        try:
            config = DatasetConfig(
                corpus_path=corpus_path,
                sequence_length=64,
                **config_overrides
            )
            
            loader = StreamingDataLoader(config)
            # Would need actual benchmarking with real data
            
        except Exception as e:
            logger.error(f"Config {i+1} failed: {e}")
    
    return results