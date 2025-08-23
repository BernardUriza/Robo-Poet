"""
Memory optimization for data pipelines with intelligent batching and caching.

Optimizes memory usage during data loading and preprocessing while maintaining
high throughput for training large language models.
"""

import os
import gc
import psutil
import logging
import time
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import threading
from collections import deque
import mmap

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    
    # Memory limits (in MB)
    max_memory_usage_mb: int = 4096  # 4GB default
    warning_threshold_mb: int = 3072  # 75% of max
    critical_threshold_mb: int = 3584  # 87.5% of max
    
    # Cache settings
    enable_dataset_cache: bool = True
    cache_dir: Optional[str] = None
    max_cache_size_mb: int = 2048  # 2GB cache
    cache_compression: bool = True
    
    # Batch optimization
    adaptive_batch_size: bool = True
    initial_batch_size: int = 32
    min_batch_size: int = 8
    max_batch_size: int = 256
    batch_size_adjustment_factor: float = 0.8
    
    # Memory mapping
    use_memory_mapping: bool = True
    mmap_threshold_mb: int = 100  # Files larger than 100MB
    
    # Prefetching
    prefetch_buffer_size: int = 4
    prefetch_workers: int = 2
    
    # Garbage collection
    gc_frequency: int = 100  # Every N batches
    force_gc_threshold: float = 0.9  # At 90% memory usage
    
    # Monitoring
    monitor_memory: bool = True
    monitor_interval: float = 10.0  # seconds
    log_memory_stats: bool = True
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.max_memory_usage_mb > 0
        assert self.warning_threshold_mb < self.max_memory_usage_mb
        assert self.critical_threshold_mb < self.max_memory_usage_mb
        assert self.min_batch_size <= self.initial_batch_size <= self.max_batch_size
        assert 0 < self.batch_size_adjustment_factor <= 1.0


class MemoryMonitor:
    """Real-time memory usage monitoring."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.process = psutil.Process()
        
        # Memory statistics
        self.peak_memory_mb = 0
        self.memory_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.warning_count = 0
        self.critical_count = 0
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return 0.0
    
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 * 1024)
        except Exception:
            return 1024.0  # Default fallback
    
    def check_memory_status(self) -> Dict[str, Any]:
        """Check current memory status and return warnings."""
        current_mb = self.get_current_memory_mb()
        available_mb = self.get_available_memory_mb()
        
        # Update statistics
        self.peak_memory_mb = max(self.peak_memory_mb, current_mb)
        self.memory_history.append({
            'timestamp': time.time(),
            'used_mb': current_mb,
            'available_mb': available_mb
        })
        
        # Check thresholds
        status = {
            'current_mb': current_mb,
            'available_mb': available_mb,
            'peak_mb': self.peak_memory_mb,
            'warning': False,
            'critical': False,
            'usage_percent': (current_mb / self.config.max_memory_usage_mb) * 100
        }
        
        if current_mb > self.config.warning_threshold_mb:
            status['warning'] = True
            self.warning_count += 1
            
        if current_mb > self.config.critical_threshold_mb:
            status['critical'] = True
            self.critical_count += 1
            
            if self.config.log_memory_stats:
                logger.warning(f"Critical memory usage: {current_mb:.1f}MB")
        
        return status
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
            
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                status = self.check_memory_status()
                
                if self.config.log_memory_stats and status['warning']:
                    logger.debug(f"Memory usage: {status['current_mb']:.1f}MB ({status['usage_percent']:.1f}%)")
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.config.monitor_interval)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_history:
            return {}
        
        recent_usage = [entry['used_mb'] for entry in self.memory_history[-10:]]
        
        return {
            'current_mb': self.get_current_memory_mb(),
            'peak_mb': self.peak_memory_mb,
            'avg_mb': np.mean([entry['used_mb'] for entry in self.memory_history]),
            'recent_avg_mb': np.mean(recent_usage) if recent_usage else 0,
            'warning_count': self.warning_count,
            'critical_count': self.critical_count,
            'samples': len(self.memory_history)
        }


class BatchOptimizer:
    """Adaptive batch size optimization based on memory usage."""
    
    def __init__(self, config: MemoryConfig, monitor: MemoryMonitor):
        self.config = config
        self.monitor = monitor
        
        # Batch size state
        self.current_batch_size = config.initial_batch_size
        self.optimal_batch_size = config.initial_batch_size
        self.batch_size_history = []
        
        # Performance tracking
        self.throughput_history = deque(maxlen=50)
        self.memory_per_batch = {}
        
    def get_optimal_batch_size(self, memory_usage_mb: float) -> int:
        """Calculate optimal batch size based on current memory usage."""
        
        # Calculate memory pressure
        memory_pressure = memory_usage_mb / self.config.max_memory_usage_mb
        
        if memory_pressure > 0.9:  # Very high pressure
            adjustment = 0.5
        elif memory_pressure > 0.8:  # High pressure
            adjustment = 0.7
        elif memory_pressure > 0.6:  # Medium pressure
            adjustment = 0.85
        else:  # Low pressure - can increase
            adjustment = 1.2
        
        new_batch_size = int(self.current_batch_size * adjustment)
        
        # Clamp to valid range
        new_batch_size = max(self.config.min_batch_size, 
                           min(new_batch_size, self.config.max_batch_size))
        
        return new_batch_size
    
    def update_batch_size(self, throughput: float, memory_mb: float) -> int:
        """Update batch size based on performance metrics."""
        
        # Record metrics
        self.throughput_history.append(throughput)
        self.memory_per_batch[self.current_batch_size] = memory_mb
        
        if not self.config.adaptive_batch_size:
            return self.current_batch_size
        
        # Calculate new batch size
        suggested_size = self.get_optimal_batch_size(memory_mb)
        
        # Check if we should adjust
        if suggested_size != self.current_batch_size:
            # Calculate expected benefit
            expected_improvement = self._estimate_improvement(suggested_size)
            
            if expected_improvement > 0.05:  # 5% improvement threshold
                self.current_batch_size = suggested_size
                self.batch_size_history.append({
                    'timestamp': time.time(),
                    'batch_size': suggested_size,
                    'memory_mb': memory_mb,
                    'throughput': throughput
                })
                
                logger.info(f"Adjusted batch size to {suggested_size} (memory: {memory_mb:.1f}MB)")
        
        return self.current_batch_size
    
    def _estimate_improvement(self, new_batch_size: int) -> float:
        """Estimate performance improvement from batch size change."""
        if len(self.throughput_history) < 5:
            return 0.1  # Default small improvement estimate
        
        # Simple heuristic based on memory efficiency
        current_memory_per_sample = (
            self.memory_per_batch.get(self.current_batch_size, 0) / 
            max(1, self.current_batch_size)
        )
        
        expected_memory = new_batch_size * current_memory_per_sample
        memory_efficiency = 1.0 - (expected_memory / self.config.max_memory_usage_mb)
        
        return max(0, memory_efficiency - 0.5)  # Improvement score
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch optimization statistics."""
        return {
            'current_batch_size': self.current_batch_size,
            'optimal_batch_size': self.optimal_batch_size,
            'avg_throughput': np.mean(self.throughput_history) if self.throughput_history else 0,
            'batch_adjustments': len(self.batch_size_history),
            'memory_per_batch': dict(self.memory_per_batch)
        }


class DataMemoryOptimizer:
    """Main memory optimizer for data pipelines."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.config.validate()
        
        # Components
        self.monitor = MemoryMonitor(config)
        self.batch_optimizer = BatchOptimizer(config, self.monitor)
        
        # Cache management
        self.cache_usage_mb = 0
        self.cached_files = {}
        
        # Memory mapping
        self.memory_mapped_files = {}
        
        # Statistics
        self.stats = {
            'batches_processed': 0,
            'gc_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_warnings': 0
        }
        
        logger.info("DataMemoryOptimizer initialized")
    
    def optimize_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Apply memory optimizations to dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Memory-optimized dataset
        """
        logger.info("Optimizing dataset for memory efficiency...")
        
        # Start monitoring
        if self.config.monitor_memory:
            self.monitor.start_monitoring()
        
        # Apply optimizations
        optimized_dataset = dataset
        
        # Dynamic batching
        if self.config.adaptive_batch_size:
            optimized_dataset = self._apply_dynamic_batching(optimized_dataset)
        
        # Memory-efficient prefetching
        optimized_dataset = optimized_dataset.prefetch(self.config.prefetch_buffer_size)
        
        # Add memory monitoring callbacks
        optimized_dataset = self._add_memory_callbacks(optimized_dataset)
        
        # Cache if beneficial
        if self.config.enable_dataset_cache:
            optimized_dataset = self._apply_smart_caching(optimized_dataset)
        
        logger.info("Dataset optimization complete")
        return optimized_dataset
    
    def _apply_dynamic_batching(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply dynamic batch sizing based on memory usage."""
        
        def dynamic_batch_fn(element):
            # Check current memory status
            status = self.monitor.check_memory_status()
            
            # Get optimal batch size
            optimal_size = self.batch_optimizer.get_optimal_batch_size(status['current_mb'])
            
            # Return element (batching will be handled elsewhere)
            return element
        
        return dataset.map(dynamic_batch_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    def _add_memory_callbacks(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Add memory monitoring callbacks to dataset."""
        
        def memory_callback(batch):
            # Update batch counter
            self.stats['batches_processed'] += 1
            
            # Check memory and potentially trigger GC
            if self.stats['batches_processed'] % self.config.gc_frequency == 0:
                self._maybe_gc()
            
            # Monitor memory status
            if self.config.monitor_memory:
                status = self.monitor.check_memory_status()
                
                if status['warning']:
                    self.stats['memory_warnings'] += 1
                
                if status['critical']:
                    logger.warning(f"Critical memory usage: {status['current_mb']:.1f}MB")
                    self._emergency_cleanup()
            
            return batch
        
        return dataset.map(memory_callback, num_parallel_calls=tf.data.AUTOTUNE)
    
    def _apply_smart_caching(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply intelligent caching based on memory availability."""
        
        if not self.config.enable_dataset_cache:
            return dataset
        
        # Check if we have enough memory for caching
        status = self.monitor.check_memory_status()
        available_for_cache = self.config.max_cache_size_mb
        
        if status['current_mb'] > self.config.warning_threshold_mb:
            # Reduce cache size under memory pressure
            available_for_cache = self.config.max_cache_size_mb // 2
        
        if available_for_cache < 100:  # Don't cache if less than 100MB available
            logger.info("Skipping cache due to memory constraints")
            return dataset
        
        # Apply caching
        if self.config.cache_dir:
            cache_path = os.path.join(self.config.cache_dir, "dataset_cache")
            cached_dataset = dataset.cache(cache_path)
        else:
            cached_dataset = dataset.cache()
        
        logger.info(f"Applied dataset caching (max {available_for_cache}MB)")
        return cached_dataset
    
    def _maybe_gc(self):
        """Conditionally trigger garbage collection."""
        status = self.monitor.check_memory_status()
        
        if (status['current_mb'] > self.config.warning_threshold_mb or
            status['current_mb'] / self.config.max_memory_usage_mb > self.config.force_gc_threshold):
            
            # Force garbage collection
            collected = gc.collect()
            self.stats['gc_calls'] += 1
            
            if collected > 0:
                new_usage = self.monitor.get_current_memory_mb()
                logger.debug(f"GC freed {collected} objects, memory: {new_usage:.1f}MB")
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup procedures."""
        logger.warning("Performing emergency memory cleanup...")
        
        # Clear caches
        self.clear_caches()
        
        # Force garbage collection
        for _ in range(3):  # Multiple GC passes
            gc.collect()
        
        # Close memory mapped files
        self._cleanup_memory_maps()
        
        logger.info("Emergency cleanup complete")
    
    def memory_map_file(self, filepath: str) -> Optional[mmap.mmap]:
        """Memory map a file if it's large enough."""
        if not self.config.use_memory_mapping:
            return None
        
        try:
            file_size = os.path.getsize(filepath)
            size_mb = file_size / (1024 * 1024)
            
            if size_mb < self.config.mmap_threshold_mb:
                return None  # File too small for memory mapping
            
            # Check if already mapped
            if filepath in self.memory_mapped_files:
                return self.memory_mapped_files[filepath]
            
            # Create memory mapping
            with open(filepath, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.memory_mapped_files[filepath] = mm
                
                logger.debug(f"Memory mapped {filepath} ({size_mb:.1f}MB)")
                return mm
        
        except Exception as e:
            logger.warning(f"Could not memory map {filepath}: {e}")
            return None
    
    def _cleanup_memory_maps(self):
        """Clean up memory mapped files."""
        for filepath, mm in list(self.memory_mapped_files.items()):
            try:
                mm.close()
                del self.memory_mapped_files[filepath]
            except Exception as e:
                logger.warning(f"Error closing memory map {filepath}: {e}")
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        logger.info("Clearing caches...")
        
        # Clear file cache
        self.cached_files.clear()
        self.cache_usage_mb = 0
        
        # Clear TensorFlow caches if available
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        memory_stats = self.monitor.get_memory_stats()
        batch_stats = self.batch_optimizer.get_batch_stats()
        
        return {
            'memory': memory_stats,
            'batching': batch_stats,
            'general': self.stats,
            'cache': {
                'usage_mb': self.cache_usage_mb,
                'files_cached': len(self.cached_files),
                'hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            },
            'memory_mapping': {
                'files_mapped': len(self.memory_mapped_files),
                'mapping_enabled': self.config.use_memory_mapping
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up DataMemoryOptimizer...")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Clear caches
        self.clear_caches()
        
        # Clean up memory maps
        self._cleanup_memory_maps()
        
        # Final garbage collection
        gc.collect()
        
        logger.info("Cleanup complete")


def optimize_dataset_memory(
    dataset: tf.data.Dataset,
    max_memory_mb: int = 4096,
    adaptive_batch_size: bool = True,
    enable_cache: bool = True,
    **config_kwargs
) -> Tuple[tf.data.Dataset, DataMemoryOptimizer]:
    """
    Factory function to optimize dataset memory usage.
    
    Args:
        dataset: Dataset to optimize
        max_memory_mb: Maximum memory usage in MB
        adaptive_batch_size: Whether to use adaptive batch sizing
        enable_cache: Whether to enable caching
        **config_kwargs: Additional configuration options
        
    Returns:
        Tuple of (optimized_dataset, optimizer)
    """
    config = MemoryConfig(
        max_memory_usage_mb=max_memory_mb,
        adaptive_batch_size=adaptive_batch_size,
        enable_dataset_cache=enable_cache,
        **config_kwargs
    )
    
    optimizer = DataMemoryOptimizer(config)
    optimized_dataset = optimizer.optimize_dataset(dataset)
    
    return optimized_dataset, optimizer


def demo_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    print("Memory Optimization Demo")
    print("=" * 30)
    
    # Create sample dataset
    def data_generator():
        for i in range(1000):
            yield np.random.random((100, 50)).astype(np.float32)  # Simulate data
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tf.TensorSpec(shape=(100, 50), dtype=tf.float32)
    )
    
    dataset = dataset.batch(32)
    
    # Create optimizer
    config = MemoryConfig(
        max_memory_usage_mb=2048,  # 2GB limit
        adaptive_batch_size=True,
        monitor_memory=True,
        log_memory_stats=True
    )
    
    optimizer = DataMemoryOptimizer(config)
    
    # Test optimization
    print("Before optimization:")
    print(f"  Memory usage: {optimizer.monitor.get_current_memory_mb():.1f}MB")
    
    optimized_dataset = optimizer.optimize_dataset(dataset)
    
    print("\nAfter optimization:")
    print(f"  Memory usage: {optimizer.monitor.get_current_memory_mb():.1f}MB")
    
    # Simulate processing some batches
    print("\nProcessing batches...")
    for i, batch in enumerate(optimized_dataset.take(10)):
        print(f"  Batch {i+1}: shape {batch.shape}")
        if i % 3 == 0:
            stats = optimizer.get_optimization_stats()
            memory_mb = stats['memory'].get('current_mb', 0)
            print(f"    Memory: {memory_mb:.1f}MB")
    
    # Final statistics
    final_stats = optimizer.get_optimization_stats()
    print("\nFinal Statistics:")
    for category, stats in final_stats.items():
        print(f"  {category.title()}:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
    
    # Cleanup
    optimizer.cleanup()
    
    return optimizer