"""
K-fold cross validation and advanced validation strategies for text generation models.

Implements stratified sampling, temporal splits, and comprehensive validation
metrics for robust model evaluation.
"""

import numpy as np
import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass 
class ValidationConfig:
    """Configuration for cross validation."""
    
    # K-fold settings
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    stratify: bool = False
    
    # Validation strategy
    validation_strategy: str = "kfold"  # kfold, temporal, holdout, custom
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Temporal validation (for time-series data)
    temporal_split_ratio: float = 0.8
    temporal_gap_days: Optional[int] = None
    
    # Stratification settings (for stratified validation)
    stratify_by_length: bool = False
    length_bins: int = 5
    stratify_by_domain: bool = False
    
    # Performance settings
    cache_splits: bool = True
    cache_dir: Optional[str] = None
    parallel_validation: bool = True
    
    # Evaluation settings
    evaluate_on_each_fold: bool = True
    aggregate_metrics: bool = True
    save_fold_results: bool = True
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.n_splits >= 2
        assert 0 < self.test_size < 1
        assert 0 < self.validation_size < 1
        assert self.validation_strategy in ['kfold', 'temporal', 'holdout', 'custom']
        if self.stratify_by_length:
            assert self.length_bins >= 2


@dataclass
class ValidationResult:
    """Results from cross validation."""
    
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    aggregated_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_fold: int = -1
    worst_fold: int = -1
    
    # Timing information
    total_validation_time: float = 0
    avg_fold_time: float = 0
    
    # Split information
    split_info: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration used
    config: ValidationConfig = None
    
    def get_mean_metric(self, metric_name: str) -> float:
        """Get mean value of a metric across folds."""
        if metric_name in self.aggregated_metrics:
            return self.aggregated_metrics[metric_name].get('mean', 0.0)
        return 0.0
    
    def get_std_metric(self, metric_name: str) -> float:
        """Get standard deviation of a metric across folds."""
        if metric_name in self.aggregated_metrics:
            return self.aggregated_metrics[metric_name].get('std', 0.0)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'fold_results': self.fold_results,
            'aggregated_metrics': self.aggregated_metrics,
            'best_fold': self.best_fold,
            'worst_fold': self.worst_fold,
            'total_validation_time': self.total_validation_time,
            'avg_fold_time': self.avg_fold_time,
            'split_info': self.split_info
        }
    
    def save(self, filepath: str):
        """Save validation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class CrossValidator:
    """Advanced cross validation for text generation models."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.config.validate()
        
        self.splits_cache = {}
        self.validation_history = []
        
        logger.info(f"CrossValidator initialized: {config.validation_strategy} with {config.n_splits} splits")
    
    def create_splits(
        self, 
        data: Union[List[Any], tf.data.Dataset],
        labels: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create train/validation splits based on strategy.
        
        Args:
            data: Input data (list or dataset)
            labels: Optional labels for stratification
            metadata: Optional metadata for custom splits
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        logger.info(f"Creating {self.config.validation_strategy} splits...")
        
        if isinstance(data, tf.data.Dataset):
            # Convert dataset to list of indices
            data_size = self._estimate_dataset_size(data)
            indices = list(range(data_size))
        else:
            indices = list(range(len(data)))
        
        # Check cache
        cache_key = self._get_cache_key(indices, labels)
        if self.config.cache_splits and cache_key in self.splits_cache:
            logger.info("Using cached splits")
            return self.splits_cache[cache_key]
        
        # Create splits based on strategy
        if self.config.validation_strategy == "kfold":
            splits = self._create_kfold_splits(indices, labels)
        elif self.config.validation_strategy == "temporal":
            splits = self._create_temporal_splits(indices, metadata)
        elif self.config.validation_strategy == "holdout":
            splits = self._create_holdout_splits(indices, labels)
        else:
            raise ValueError(f"Unknown validation strategy: {self.config.validation_strategy}")
        
        # Cache splits
        if self.config.cache_splits:
            self.splits_cache[cache_key] = splits
        
        logger.info(f"Created {len(splits)} splits")
        return splits
    
    def _create_kfold_splits(
        self, 
        indices: List[int], 
        labels: Optional[List[Any]] = None
    ) -> List[Tuple[List[int], List[int]]]:
        """Create K-fold cross validation splits."""
        
        if self.config.stratify and labels is not None:
            # Stratified K-fold
            if self.config.stratify_by_length:
                # Stratify by sequence length
                stratify_labels = self._create_length_labels(labels)
            else:
                stratify_labels = labels
            
            kfold = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            
            splits = list(kfold.split(indices, stratify_labels))
        else:
            # Regular K-fold
            kfold = KFold(
                n_splits=self.config.n_splits,
                shuffle=self.config.shuffle,
                random_state=self.config.random_state
            )
            
            splits = list(kfold.split(indices))
        
        # Convert to list of indices
        return [(train_idx.tolist(), val_idx.tolist()) for train_idx, val_idx in splits]
    
    def _create_temporal_splits(
        self, 
        indices: List[int], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[List[int], List[int]]]:
        """Create temporal splits for time-series data."""
        
        if metadata and 'timestamps' in metadata:
            # Sort by timestamp
            timestamps = metadata['timestamps']
            sorted_indices = sorted(indices, key=lambda i: timestamps[i])
        else:
            # Assume data is already in chronological order
            sorted_indices = indices
        
        splits = []
        n_samples = len(sorted_indices)
        
        if self.config.n_splits == 1:
            # Single temporal split
            split_point = int(n_samples * self.config.temporal_split_ratio)
            train_indices = sorted_indices[:split_point]
            val_indices = sorted_indices[split_point:]
            splits.append((train_indices, val_indices))
        else:
            # Multiple temporal splits (walk-forward validation)
            window_size = n_samples // (self.config.n_splits + 1)
            
            for i in range(self.config.n_splits):
                train_end = window_size * (i + 2)
                val_start = train_end
                val_end = min(val_start + window_size, n_samples)
                
                if val_end <= val_start:
                    break
                
                train_indices = sorted_indices[:train_end]
                val_indices = sorted_indices[val_start:val_end]
                splits.append((train_indices, val_indices))
        
        return splits
    
    def _create_holdout_splits(
        self, 
        indices: List[int], 
        labels: Optional[List[Any]] = None
    ) -> List[Tuple[List[int], List[int]]]:
        """Create single holdout split."""
        
        if self.config.stratify and labels is not None:
            from sklearn.model_selection import train_test_split
            
            if self.config.stratify_by_length:
                stratify_labels = self._create_length_labels(labels)
            else:
                stratify_labels = labels
            
            train_indices, val_indices = train_test_split(
                indices,
                test_size=self.config.validation_size,
                stratify=stratify_labels,
                random_state=self.config.random_state
            )
        else:
            # Random split
            np.random.seed(self.config.random_state)
            shuffled_indices = np.random.permutation(indices)
            
            split_point = int(len(indices) * (1 - self.config.validation_size))
            train_indices = shuffled_indices[:split_point].tolist()
            val_indices = shuffled_indices[split_point:].tolist()
        
        return [(train_indices, val_indices)]
    
    def _create_length_labels(self, sequences: List[Any]) -> List[int]:
        """Create labels based on sequence length for stratification."""
        lengths = [len(seq) if hasattr(seq, '__len__') else 0 for seq in sequences]
        
        # Create bins based on length percentiles
        percentiles = np.linspace(0, 100, self.config.length_bins + 1)
        bin_edges = np.percentile(lengths, percentiles)
        
        # Assign each sequence to a bin
        labels = np.digitize(lengths, bin_edges) - 1
        labels = np.clip(labels, 0, self.config.length_bins - 1)
        
        return labels.tolist()
    
    def validate_model(
        self,
        model_fn: Callable,
        train_data: Union[List[Any], tf.data.Dataset],
        eval_fn: Callable,
        labels: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Perform cross validation on a model.
        
        Args:
            model_fn: Function that creates and trains a model
            train_data: Training data
            eval_fn: Function to evaluate model performance
            labels: Optional labels for stratification
            metadata: Optional metadata for custom splits
            
        Returns:
            ValidationResult with aggregated metrics
        """
        start_time = time.time()
        
        # Create splits
        splits = self.create_splits(train_data, labels, metadata)
        
        result = ValidationResult(config=self.config)
        fold_times = []
        
        logger.info(f"Starting cross validation with {len(splits)} folds")
        
        # Process each fold
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            fold_start_time = time.time()
            
            logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")
            logger.info(f"  Train samples: {len(train_indices)}")
            logger.info(f"  Validation samples: {len(val_indices)}")
            
            try:
                # Create train/val datasets for this fold
                train_fold_data = self._subset_data(train_data, train_indices)
                val_fold_data = self._subset_data(train_data, val_indices)
                
                # Train model on this fold
                model = model_fn(train_fold_data)
                
                # Evaluate model
                if self.config.evaluate_on_each_fold:
                    fold_metrics = eval_fn(model, val_fold_data)
                    
                    fold_result = {
                        'fold': fold_idx,
                        'train_size': len(train_indices),
                        'val_size': len(val_indices),
                        'metrics': fold_metrics,
                        'train_indices': train_indices,
                        'val_indices': val_indices
                    }
                    
                    result.fold_results.append(fold_result)
                
                fold_time = time.time() - fold_start_time
                fold_times.append(fold_time)
                
                logger.info(f"  Fold {fold_idx + 1} completed in {fold_time:.2f}s")
                
                if self.config.evaluate_on_each_fold and fold_metrics:
                    for metric, value in fold_metrics.items():
                        logger.info(f"    {metric}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx + 1}: {e}")
                fold_result = {
                    'fold': fold_idx,
                    'error': str(e),
                    'train_size': len(train_indices),
                    'val_size': len(val_indices)
                }
                result.fold_results.append(fold_result)
        
        # Calculate aggregated metrics
        if self.config.aggregate_metrics:
            result.aggregated_metrics = self._aggregate_metrics(result.fold_results)
        
        # Find best and worst folds
        if result.fold_results and 'metrics' in result.fold_results[0]:
            primary_metric = self._get_primary_metric(result.fold_results[0]['metrics'])
            
            fold_scores = [
                fr['metrics'].get(primary_metric, 0) 
                for fr in result.fold_results 
                if 'metrics' in fr
            ]
            
            if fold_scores:
                result.best_fold = int(np.argmax(fold_scores))
                result.worst_fold = int(np.argmin(fold_scores))
        
        # Set timing information
        result.total_validation_time = time.time() - start_time
        result.avg_fold_time = np.mean(fold_times) if fold_times else 0
        
        # Set split information
        result.split_info = {
            'strategy': self.config.validation_strategy,
            'n_splits': len(splits),
            'total_samples': len(train_indices) + len(val_indices) if splits else 0
        }
        
        logger.info(f"Cross validation completed in {result.total_validation_time:.2f}s")
        
        if result.aggregated_metrics:
            logger.info("Aggregated Results:")
            for metric, stats in result.aggregated_metrics.items():
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        return result
    
    def _subset_data(self, data: Union[List[Any], tf.data.Dataset], indices: List[int]):
        """Create subset of data using indices."""
        if isinstance(data, list):
            return [data[i] for i in indices]
        else:
            # For tf.data.Dataset, this is more complex
            # Would need to implement dataset indexing
            # For now, return a placeholder
            return data.take(len(indices))
    
    def _aggregate_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across folds."""
        aggregated = defaultdict(list)
        
        # Collect metrics from all folds
        for fold_result in fold_results:
            if 'metrics' in fold_result:
                for metric_name, metric_value in fold_result['metrics'].items():
                    if isinstance(metric_value, (int, float)):
                        aggregated[metric_name].append(metric_value)
        
        # Calculate statistics for each metric
        result = {}
        for metric_name, values in aggregated.items():
            if values:
                result[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
        
        return result
    
    def _get_primary_metric(self, metrics: Dict[str, Any]) -> str:
        """Determine primary metric for ranking folds."""
        # Priority order for primary metric
        primary_candidates = ['bleu', 'rouge', 'accuracy', 'f1', 'loss']
        
        for candidate in primary_candidates:
            if candidate in metrics:
                return candidate
        
        # If none found, use first available metric
        return list(metrics.keys())[0] if metrics else 'loss'
    
    def _estimate_dataset_size(self, dataset: tf.data.Dataset) -> int:
        """Estimate size of tf.data.Dataset."""
        # This is a simplified estimation
        # In practice, you might need dataset.cardinality() or iterate once
        try:
            return len(list(dataset.take(1000)))  # Sample-based estimation
        except:
            return 1000  # Default fallback
    
    def _get_cache_key(self, indices: List[int], labels: Optional[List[Any]] = None) -> str:
        """Generate cache key for splits."""
        key_data = {
            'indices': indices,
            'labels': labels,
            'config': {
                'n_splits': self.config.n_splits,
                'shuffle': self.config.shuffle,
                'random_state': self.config.random_state,
                'strategy': self.config.validation_strategy
            }
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()


def create_kfold_splits(
    data: Union[List[Any], tf.data.Dataset],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    stratify: bool = False,
    labels: Optional[List[Any]] = None
) -> List[Tuple[List[int], List[int]]]:
    """
    Factory function to create K-fold splits.
    
    Args:
        data: Input data
        n_splits: Number of folds
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
        stratify: Whether to use stratified sampling
        labels: Labels for stratification
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    config = ValidationConfig(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratify,
        validation_strategy='kfold'
    )
    
    validator = CrossValidator(config)
    return validator.create_splits(data, labels)


def validate_text_generation_model(
    model_factory: Callable,
    training_data: tf.data.Dataset,
    evaluation_function: Callable,
    n_splits: int = 5,
    validation_strategy: str = 'kfold',
    **config_kwargs
) -> ValidationResult:
    """
    Validate a text generation model using cross validation.
    
    Args:
        model_factory: Function that creates and trains a model
        training_data: Training dataset
        evaluation_function: Function to evaluate model
        n_splits: Number of validation splits
        validation_strategy: Validation strategy to use
        **config_kwargs: Additional configuration options
        
    Returns:
        ValidationResult with metrics and analysis
    """
    config = ValidationConfig(
        n_splits=n_splits,
        validation_strategy=validation_strategy,
        **config_kwargs
    )
    
    validator = CrossValidator(config)
    
    return validator.validate_model(
        model_factory,
        training_data,
        evaluation_function
    )


class AdvancedValidator:
    """Advanced validation with custom strategies and analysis."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.cross_validator = CrossValidator(config)
        
    def nested_cross_validation(
        self,
        model_factory: Callable,
        hyperparameter_grid: Dict[str, List[Any]],
        training_data: tf.data.Dataset,
        evaluation_function: Callable,
        inner_cv_folds: int = 3,
        outer_cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform nested cross validation for hyperparameter tuning.
        
        This provides unbiased estimates of model performance while
        simultaneously performing hyperparameter optimization.
        """
        logger.info("Starting nested cross validation")
        logger.info(f"Outer CV: {outer_cv_folds} folds, Inner CV: {inner_cv_folds} folds")
        
        # Create outer CV splits
        outer_config = ValidationConfig(**self.config.__dict__)
        outer_config.n_splits = outer_cv_folds
        outer_validator = CrossValidator(outer_config)
        
        outer_splits = outer_validator.create_splits(training_data)
        
        nested_results = {
            'outer_scores': [],
            'best_params_per_fold': [],
            'inner_cv_results': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
            logger.info(f"Outer fold {fold_idx + 1}/{outer_cv_folds}")
            
            # Get data for this outer fold
            outer_train_data = self.cross_validator._subset_data(training_data, train_idx)
            outer_test_data = self.cross_validator._subset_data(training_data, test_idx)
            
            # Inner CV for hyperparameter tuning
            best_params, inner_results = self._hyperparameter_search(
                model_factory,
                hyperparameter_grid,
                outer_train_data,
                evaluation_function,
                inner_cv_folds
            )
            
            # Train final model with best params on full outer training set
            final_model = model_factory(outer_train_data, **best_params)
            
            # Evaluate on outer test set
            outer_score = evaluation_function(final_model, outer_test_data)
            
            nested_results['outer_scores'].append(outer_score)
            nested_results['best_params_per_fold'].append(best_params)
            nested_results['inner_cv_results'].append(inner_results)
            
            logger.info(f"Outer fold {fold_idx + 1} score: {outer_score}")
            logger.info(f"Best params: {best_params}")
        
        # Calculate final statistics
        nested_results['mean_outer_score'] = np.mean(nested_results['outer_scores'])
        nested_results['std_outer_score'] = np.std(nested_results['outer_scores'])
        
        logger.info(f"Nested CV Results:")
        logger.info(f"  Mean Score: {nested_results['mean_outer_score']:.4f}")
        logger.info(f"  Std Score: {nested_results['std_outer_score']:.4f}")
        
        return nested_results
    
    def _hyperparameter_search(
        self,
        model_factory: Callable,
        param_grid: Dict[str, List[Any]],
        training_data: tf.data.Dataset,
        eval_function: Callable,
        cv_folds: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform grid search with cross validation."""
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_score = -np.inf
        best_params = None
        all_results = []
        
        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))
            
            # Cross validate this parameter combination
            config = ValidationConfig(n_splits=cv_folds, validation_strategy='kfold')
            validator = CrossValidator(config)
            
            cv_result = validator.validate_model(
                lambda data: model_factory(data, **params),
                training_data,
                eval_function
            )
            
            # Get primary metric score
            primary_metric = self.cross_validator._get_primary_metric(
                cv_result.fold_results[0]['metrics']
            )
            mean_score = cv_result.get_mean_metric(primary_metric)
            
            all_results.append({
                'params': params,
                'mean_score': mean_score,
                'cv_result': cv_result
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        return best_params, {
            'best_score': best_score,
            'all_results': all_results
        }


def demo_cross_validation():
    """Demonstrate cross validation capabilities."""
    # Create dummy data
    dummy_data = [[i] * 10 for i in range(100)]  # 100 sequences
    dummy_labels = [len(seq) for seq in dummy_data]  # Length-based labels
    
    config = ValidationConfig(
        n_splits=5,
        shuffle=True,
        stratify=True,
        stratify_by_length=True
    )
    
    validator = CrossValidator(config)
    
    # Create splits
    splits = validator.create_splits(dummy_data, dummy_labels)
    
    print("Cross Validation Demo")
    print("=" * 30)
    print(f"Data size: {len(dummy_data)}")
    print(f"Number of splits: {len(splits)}")
    print()
    
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"Fold {i+1}:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")
        print(f"  Train ratio: {len(train_idx)/len(dummy_data):.3f}")
    
    return validator