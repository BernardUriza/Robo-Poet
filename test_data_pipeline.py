#!/usr/bin/env python3
"""
Test comprehensivo para Strategy 6: Pipeline de Datos Profesional
Prueba streaming, augmentation, validation, preprocessing y memory optimization
"""

import os
import sys
import logging
import time
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def create_test_corpus(corpus_dir: str, num_files: int = 5, lines_per_file: int = 100):
    """Crea corpus de prueba con múltiples archivos."""
    os.makedirs(corpus_dir, exist_ok=True)
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process data.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of training data.",
        "Text generation has become increasingly sophisticated.",
        "Artificial intelligence is revolutionizing many industries.",
        "Data preprocessing is crucial for model performance.",
        "Advanced algorithms can learn patterns from text.",
        "Language models are becoming more capable every year.",
        "Text augmentation helps improve model robustness."
    ]
    
    total_lines = 0
    for i in range(num_files):
        filepath = os.path.join(corpus_dir, f"text_file_{i+1}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for j in range(lines_per_file):
                # Vary the text length and content
                base_text = sample_texts[j % len(sample_texts)]
                if j % 3 == 0:
                    text = base_text.upper()
                elif j % 5 == 0:
                    text = base_text + f" Line {j+1} in file {i+1}."
                else:
                    text = base_text
                
                f.write(text + "\n")
                total_lines += 1
    
    logger.info(f"Created test corpus: {num_files} files, {total_lines} lines total")
    return corpus_dir


def test_streaming_pipeline():
    """Prueba pipeline de streaming de datos."""
    print("🔍 PRUEBA 1: Streaming Data Pipeline")
    print("=" * 50)
    
    try:
        from data.streaming import (
            TextDataset, StreamingDataLoader, DatasetConfig, create_streaming_dataset
        )
        
        # Crear corpus temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = create_test_corpus(temp_dir, num_files=3, lines_per_file=50)
            
            # Configuración básica
            config = DatasetConfig(
                corpus_path=corpus_dir,
                vocab_size=1000,
                sequence_length=20,
                batch_size=8,
                buffer_size=100,
                chunk_size=512,
                overlap_tokens=5
            )
            
            print(f"Configuración:")
            print(f"  Corpus: {corpus_dir}")
            print(f"  Vocab size: {config.vocab_size}")
            print(f"  Sequence length: {config.sequence_length}")
            print(f"  Batch size: {config.batch_size}")
            
            # Crear dataset
            text_dataset = TextDataset(config)
            
            # Descubrir archivos
            files = text_dataset.discover_files(corpus_dir)
            print(f"  Archivos descubiertos: {len(files)}")
            print(f"  Tamaño total: {text_dataset.total_size_mb:.3f}MB")
            
            # Crear tokenizer simple para prueba
            class SimpleTokenizer:
                def __init__(self):
                    self.word_to_index = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
                    self.vocab_size = 1000
                    self.next_id = 3
                
                def encode(self, text):
                    words = text.lower().split()
                    ids = []
                    for word in words:
                        if word not in self.word_to_index:
                            if self.next_id < self.vocab_size:
                                self.word_to_index[word] = self.next_id
                                self.next_id += 1
                            word_id = self.word_to_index.get(word, 1)
                        else:
                            word_id = self.word_to_index[word]
                        ids.append(word_id)
                    return ids[:config.sequence_length]  # Truncar
            
            tokenizer = SimpleTokenizer()
            text_dataset.set_tokenizer(tokenizer)
            
            # Crear dataset TensorFlow
            tf_dataset = text_dataset.create_dataset()
            
            print(f"✅ Dataset creado exitosamente")
            
            # Probar iteración
            sample_count = 0
            for batch in tf_dataset.take(3):
                inputs, targets = batch
                print(f"  Batch {sample_count + 1}: input_shape={inputs.shape}, target_shape={targets.shape}")
                sample_count += 1
            
            # Información del dataset
            info = text_dataset.get_dataset_info()
            print(f"Información del dataset:")
            for key, value in info.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
            
            # Probar StreamingDataLoader
            print(f"\nProbando StreamingDataLoader...")
            loader = StreamingDataLoader(config)
            train_ds, val_ds = loader.create_train_dataset(corpus_dir, tokenizer, validation_split=0.2)
            
            print(f"  Train dataset creado: {'✅' if train_ds else '❌'}")
            print(f"  Validation dataset creado: {'✅' if val_ds else '❌'}")
            
            if train_ds:
                # Benchmark performance
                print(f"  Ejecutando benchmark...")
                stats = loader.benchmark_dataset(train_ds, num_batches=10)
                print(f"    Batches/sec: {stats['batches_per_second']:.2f}")
                print(f"    Samples/sec: {stats['samples_per_second']:.1f}")
                print(f"    Avg batch time: {stats['avg_batch_time']*1000:.2f}ms")
        
        print("✅ Streaming pipeline probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en streaming: {e}")
        return False


def test_text_augmentation():
    """Prueba augmentation de texto."""
    print("\n🔍 PRUEBA 2: Text Augmentation")
    print("=" * 50)
    
    try:
        from data.augmentation import (
            TextAugmenter, AugmentationConfig, AugmentationType, create_augmented_dataset
        )
        
        # Configuración
        config = AugmentationConfig(
            augmentation_probability=0.5,
            synonym_prob=0.3,
            insertion_prob=0.2,
            swap_prob=0.2,
            deletion_prob=0.1,
            mask_prob=0.2
        )
        
        print(f"Configuración de augmentation:")
        print(f"  Probability: {config.augmentation_probability}")
        print(f"  Synonym replacement: {config.synonym_prob}")
        print(f"  Random insertion: {config.insertion_prob}")
        print(f"  Token swapping: {config.swap_prob}")
        print(f"  Token masking: {config.mask_prob}")
        
        # Crear augmenter
        augmenter = TextAugmenter(config, vocab_size=1000)
        
        # Probar con secuencias de tokens de ejemplo
        test_sequences = [
            [10, 25, 67, 123, 89, 45, 156, 78, 234],
            [5, 15, 35, 55, 75, 95, 115, 135],
            [100, 200, 300, 400, 500, 600]
        ]
        
        print(f"\nProbando augmentation en {len(test_sequences)} secuencias:")
        
        for i, tokens in enumerate(test_sequences):
            print(f"\n  Secuencia {i+1} original: {tokens}")
            
            # Generar versiones aumentadas
            augmented = augmenter.augment_sequence(tokens, num_augmentations=3)
            
            for j, aug_seq in enumerate(augmented[1:], 1):  # Skip original
                print(f"    Aumentada {j}: {aug_seq}")
        
        # Probar augmentation por lotes
        print(f"\nProbando augmentation por lotes...")
        batch_augmented = augmenter.augment_batch(test_sequences, augmentation_probability=0.8)
        
        for i, (orig, aug) in enumerate(zip(test_sequences, batch_augmented)):
            changed = "✅ Modificada" if orig != aug else "⭕ Sin cambios"
            print(f"  Batch {i+1}: {changed}")
        
        # Probar con dataset TensorFlow
        print(f"\nCreando dataset aumentado...")
        
        # Crear dataset simple
        def data_generator():
            for seq in test_sequences * 10:  # Repetir para tener más datos
                yield (np.array(seq[:-1]), np.array(seq[1:]))  # Input-target pairs
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        
        # Aplicar augmentation
        augmented_dataset = create_augmented_dataset(
            dataset.take(5), 
            vocab_size=1000,
            augmentation_factor=2
        )
        
        print(f"  Dataset aumentado creado: ✅")
        
        # Verificar que funcione
        sample_count = 0
        for batch in augmented_dataset.take(3):
            inputs, targets = batch
            print(f"    Muestra {sample_count + 1}: input_len={len(inputs)}, target_len={len(targets)}")
            sample_count += 1
        
        # Estadísticas
        stats = augmenter.get_augmentation_stats()
        print(f"\nEstadísticas de augmentation:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print("✅ Text augmentation probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en augmentation: {e}")
        return False


def test_cross_validation():
    """Prueba cross validation."""
    print("\n🔍 PRUEBA 3: Cross Validation")
    print("=" * 50)
    
    try:
        from data.validation import (
            CrossValidator, ValidationConfig, ValidationResult, create_kfold_splits
        )
        
        # Configuración
        config = ValidationConfig(
            n_splits=5,
            shuffle=True,
            random_state=42,
            validation_strategy='kfold',
            stratify=True,
            stratify_by_length=True
        )
        
        print(f"Configuración de validación:")
        print(f"  Estrategia: {config.validation_strategy}")
        print(f"  Splits: {config.n_splits}")
        print(f"  Shuffle: {config.shuffle}")
        print(f"  Stratify: {config.stratify}")
        
        # Crear datos de prueba
        dummy_data = []
        dummy_labels = []
        
        for i in range(100):
            # Crear secuencias de diferentes longitudes
            length = np.random.randint(10, 50)
            sequence = list(range(length))
            dummy_data.append(sequence)
            dummy_labels.append(length)  # Label basado en longitud
        
        print(f"  Datos de prueba: {len(dummy_data)} secuencias")
        print(f"  Longitud promedio: {np.mean(dummy_labels):.1f}")
        
        # Crear validator
        validator = CrossValidator(config)
        
        # Crear splits
        splits = validator.create_splits(dummy_data, dummy_labels)
        
        print(f"\nSplits creados: {len(splits)}")
        
        for i, (train_idx, val_idx) in enumerate(splits):
            train_size = len(train_idx)
            val_size = len(val_idx)
            train_ratio = train_size / (train_size + val_size)
            
            print(f"  Fold {i+1}: {train_size} train, {val_size} val (ratio: {train_ratio:.3f})")
        
        # Probar validación completa
        print(f"\nEjecutando validación cruzada...")
        
        # Función mock para crear modelo
        def mock_model_factory(train_data):
            """Mock model factory para pruebas."""
            return {'trained_on': len(train_data)}
        
        # Función mock para evaluación
        def mock_eval_function(model, val_data):
            """Mock evaluation function."""
            # Simular métricas
            return {
                'accuracy': np.random.uniform(0.7, 0.95),
                'loss': np.random.uniform(0.1, 0.5),
                'bleu': np.random.uniform(0.3, 0.8)
            }
        
        # Ejecutar validación
        result = validator.validate_model(
            mock_model_factory,
            dummy_data,
            mock_eval_function,
            labels=dummy_labels
        )
        
        print(f"Validación completada:")
        print(f"  Tiempo total: {result.total_validation_time:.2f}s")
        print(f"  Tiempo promedio por fold: {result.avg_fold_time:.2f}s")
        print(f"  Mejor fold: {result.best_fold + 1}")
        print(f"  Peor fold: {result.worst_fold + 1}")
        
        # Métricas agregadas
        if result.aggregated_metrics:
            print(f"\nMétricas agregadas:")
            for metric, stats in result.aggregated_metrics.items():
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Probar factory function
        print(f"\nProbando factory function...")
        simple_splits = create_kfold_splits(dummy_data, n_splits=3, stratify=True, labels=dummy_labels)
        
        print(f"  Factory splits creados: {len(simple_splits)}")
        
        print("✅ Cross validation probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en validation: {e}")
        return False


def test_preprocessing_pipeline():
    """Prueba pipeline de preprocessing."""
    print("\n🔍 PRUEBA 4: Preprocessing Pipeline")
    print("=" * 50)
    
    try:
        from data.preprocessing import (
            AdvancedPreprocessor, PreprocessingConfig, TokenizationStrategy, 
            create_preprocessing_pipeline
        )
        
        # Textos de prueba
        test_texts = [
            "Hello, World! This is a SAMPLE text with Numbers 123 and émojis 😊",
            "Another example with punctuation... and special chars: @#$%",
            "   Multiple    spaces   and\ttabs\nand newlines   ",
            "Very long text " + "word " * 100,  # Texto muy largo
            "Short",
            ""  # Texto vacío
        ]
        
        # Probar diferentes configuraciones
        configs = [
            {
                'tokenization_strategy': 'word_based',
                'lowercase': True,
                'remove_punctuation': False,
                'vocab_size': 100
            },
            {
                'tokenization_strategy': 'character_based',
                'lowercase': True,
                'remove_punctuation': True,
                'vocab_size': 50
            },
            {
                'tokenization_strategy': 'subword_bpe',
                'bpe_merges': 10,
                'vocab_size': 80
            }
        ]
        
        for i, config_params in enumerate(configs):
            print(f"\n--- Configuración {i+1}: {config_params['tokenization_strategy']} ---")
            
            # Crear preprocessor
            preprocessor = create_preprocessing_pipeline(**config_params)
            
            # Entrenar tokenizer
            print("Entrenando tokenizer...")
            stats = preprocessor.fit_tokenizer(test_texts)
            
            print(f"  Líneas procesadas: {stats['lines_processed']}")
            print(f"  Líneas filtradas: {stats['lines_filtered']}")
            print(f"  Tamaño de vocabulario: {stats['vocab_size']}")
            print(f"  Tiempo de procesamiento: {stats['processing_time']:.3f}s")
            
            # Probar encoding/decoding
            test_text = test_texts[0]
            print(f"\nProbando encoding/decoding:")
            print(f"  Original: '{test_text}'")
            
            # Limpiar texto
            cleaned = preprocessor.clean_text(test_text)
            print(f"  Limpieza: '{cleaned}'")
            
            # Tokenizar
            tokens = preprocessor.tokenize_text(cleaned)
            print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            
            # Encodificar
            encoded = preprocessor.encode(test_text)
            print(f"  Encoded: {encoded}")
            
            # Decodificar
            decoded = preprocessor.decode(encoded)
            print(f"  Decoded: '{decoded}'")
            
            # Batch processing
            print(f"\nProbando batch processing...")
            batch_encoded = preprocessor.batch_encode(test_texts[:3], pad_to_max_length=True)
            
            for j, seq in enumerate(batch_encoded):
                print(f"    Sequence {j+1}: length={len(seq)}")
            
            # Información del vocabulario
            vocab_info = preprocessor.get_vocab_info()
            print(f"\nInformación del vocabulario:")
            print(f"  Tamaño: {vocab_info['vocab_size']}")
            print(f"  Tokens especiales: {vocab_info['special_tokens']}")
            print(f"  Tokens más frecuentes: {vocab_info['most_frequent_tokens'][:5]}")
        
        # Probar guardado y carga
        print(f"\nProbando guardado y carga...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "preprocessor.pkl")
            
            # Guardar
            preprocessor.save(save_path)
            print(f"  Guardado en: {save_path}")
            
            # Cargar
            loaded_preprocessor = AdvancedPreprocessor.load(save_path)
            print(f"  Cargado exitosamente: ✅")
            
            # Verificar que funcione igual
            original_encoded = preprocessor.encode(test_texts[0])
            loaded_encoded = loaded_preprocessor.encode(test_texts[0])
            
            if original_encoded == loaded_encoded:
                print(f"  Encoding idéntico después de cargar: ✅")
            else:
                print(f"  Encoding diferente después de cargar: ❌")
        
        print("✅ Preprocessing pipeline probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en preprocessing: {e}")
        return False


def test_memory_optimization():
    """Prueba optimización de memoria."""
    print("\n🔍 PRUEBA 5: Memory Optimization")
    print("=" * 50)
    
    try:
        from data.memory_optimizer import (
            DataMemoryOptimizer, MemoryConfig, MemoryMonitor, BatchOptimizer,
            optimize_dataset_memory
        )
        
        # Configuración
        config = MemoryConfig(
            max_memory_usage_mb=1024,  # 1GB para pruebas
            adaptive_batch_size=True,
            enable_dataset_cache=False,  # Deshabilitar para pruebas
            monitor_memory=True,
            gc_frequency=5
        )
        
        print(f"Configuración de memoria:")
        print(f"  Límite máximo: {config.max_memory_usage_mb}MB")
        print(f"  Batch size adaptativo: {config.adaptive_batch_size}")
        print(f"  Monitoreo: {config.monitor_memory}")
        print(f"  GC frequency: {config.gc_frequency}")
        
        # Probar monitor de memoria
        print(f"\nProbando monitor de memoria...")
        monitor = MemoryMonitor(config)
        
        initial_memory = monitor.get_current_memory_mb()
        available_memory = monitor.get_available_memory_mb()
        
        print(f"  Memoria actual: {initial_memory:.1f}MB")
        print(f"  Memoria disponible: {available_memory:.1f}MB")
        
        # Probar estado de memoria
        status = monitor.check_memory_status()
        print(f"  Estado de memoria:")
        for key, value in status.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")
        
        # Crear dataset de prueba que consuma memoria
        def memory_intensive_generator():
            for i in range(100):
                # Crear arrays grandes para simular uso de memoria
                data = np.random.random((50, 100)).astype(np.float32)
                target = np.random.random((50, 100)).astype(np.float32)
                yield data, target
        
        dataset = tf.data.Dataset.from_generator(
            memory_intensive_generator,
            output_signature=(
                tf.TensorSpec(shape=(50, 100), dtype=tf.float32),
                tf.TensorSpec(shape=(50, 100), dtype=tf.float32)
            )
        )
        
        dataset = dataset.batch(8)
        
        print(f"\nCreando optimizador de memoria...")
        optimizer = DataMemoryOptimizer(config)
        
        # Optimizar dataset
        print(f"Optimizando dataset...")
        optimized_dataset = optimizer.optimize_dataset(dataset)
        
        print(f"  Dataset optimizado: ✅")
        
        # Procesar algunos batches para probar optimización
        print(f"\nProcesando batches optimizados...")
        batch_count = 0
        
        for batch in optimized_dataset.take(10):
            inputs, targets = batch
            print(f"  Batch {batch_count + 1}: input_shape={inputs.shape}")
            
            # Verificar memoria cada pocos batches
            if batch_count % 3 == 0:
                current_memory = monitor.get_current_memory_mb()
                print(f"    Memoria actual: {current_memory:.1f}MB")
            
            batch_count += 1
            
            # Simular algo de procesamiento
            _ = tf.reduce_mean(inputs)
        
        # Obtener estadísticas de optimización
        print(f"\nEstadísticas de optimización:")
        stats = optimizer.get_optimization_stats()
        
        for category, category_stats in stats.items():
            print(f"  {category.title()}:")
            for key, value in category_stats.items():
                if isinstance(value, (int, float)):
                    if 'mb' in key.lower():
                        print(f"    {key}: {value:.1f}")
                    elif 'rate' in key.lower():
                        print(f"    {key}: {value:.3f}")
                    else:
                        print(f"    {key}: {value}")
                else:
                    print(f"    {key}: {value}")
        
        # Probar factory function
        print(f"\nProbando factory function...")
        optimized_ds, opt = optimize_dataset_memory(
            dataset.take(5),
            max_memory_mb=512,
            adaptive_batch_size=True
        )
        
        print(f"  Factory optimization: ✅")
        
        # Limpiar
        optimizer.cleanup()
        print(f"  Cleanup completado: ✅")
        
        print("✅ Memory optimization probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en memory optimization: {e}")
        return False


def test_integrated_pipeline():
    """Prueba integración completa del pipeline."""
    print("\n🔍 PRUEBA 6: Integrated Data Pipeline")
    print("=" * 50)
    
    try:
        # Importar todos los componentes
        from data.streaming import create_streaming_dataset
        from data.augmentation import create_augmented_dataset
        from data.preprocessing import create_preprocessing_pipeline
        from data.memory_optimizer import optimize_dataset_memory
        
        print("Creando pipeline integrado completo...")
        
        # 1. Crear corpus temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = create_test_corpus(temp_dir, num_files=2, lines_per_file=20)
            print(f"  ✅ Corpus creado: {corpus_dir}")
            
            # 2. Preprocessing
            preprocessor = create_preprocessing_pipeline(
                tokenization_strategy='word_based',
                vocab_size=200,
                max_sequence_length=30,
                lowercase=True
            )
            
            # Entrenar preprocessor con archivos del corpus
            texts = []
            for file_path in Path(corpus_dir).glob("*.txt"):
                with open(file_path, 'r') as f:
                    texts.extend(f.readlines())
            
            preprocessor.fit_tokenizer(texts)
            print(f"  ✅ Preprocessor entrenado: vocab_size={len(preprocessor.vocabulary)}")
            
            # 3. Streaming dataset
            dataset = create_streaming_dataset(
                corpus_path=corpus_dir,
                sequence_length=25,
                batch_size=4,
                vocab_size=200,
                tokenizer=preprocessor,
                validation_split=0.0  # Sin split para simplificar
            )
            print(f"  ✅ Streaming dataset creado")
            
            # 4. Data augmentation
            augmented_dataset = create_augmented_dataset(
                dataset.take(10),
                vocab_size=200,
                augmentation_probability=0.3,
                augmentation_factor=2
            )
            print(f"  ✅ Augmented dataset creado")
            
            # 5. Memory optimization
            optimized_dataset, optimizer = optimize_dataset_memory(
                augmented_dataset,
                max_memory_mb=512,
                adaptive_batch_size=True,
                enable_cache=False
            )
            print(f"  ✅ Memory-optimized dataset creado")
            
            # 6. Probar pipeline completo
            print(f"\nProbando pipeline integrado...")
            sample_count = 0
            
            for batch in optimized_dataset.take(5):
                inputs, targets = batch
                print(f"  Batch {sample_count + 1}:")
                print(f"    Input shape: {inputs.shape}")
                print(f"    Target shape: {targets.shape}")
                print(f"    Input sample: {inputs[0].numpy()[:10]}...")
                print(f"    Target sample: {targets[0].numpy()[:10]}...")
                sample_count += 1
            
            # 7. Verificar que cada componente funcionó
            print(f"\nVerificación de componentes:")
            
            # Verificar preprocessing
            test_text = "This is a test sentence for preprocessing."
            encoded = preprocessor.encode(test_text)
            decoded = preprocessor.decode(encoded)
            print(f"  Preprocessing: '{test_text}' -> {len(encoded)} tokens -> '{decoded}'")
            
            # Verificar stats del optimizer
            opt_stats = optimizer.get_optimization_stats()
            memory_mb = opt_stats['memory'].get('current_mb', 0)
            print(f"  Memory optimization: {memory_mb:.1f}MB current usage")
            
            # Cleanup
            optimizer.cleanup()
            print(f"  ✅ Pipeline cleanup completado")
        
        print("✅ Integrated pipeline probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en integrated pipeline: {e}")
        return False


def main():
    """Función principal de testing."""
    print("🚀 TESTING STRATEGY 6: PIPELINE DE DATOS PROFESIONAL")
    print("=" * 70)
    print("Componentes: Streaming, Augmentation, Validation, Preprocessing, Memory Optimization")
    print("=" * 70)
    
    tests = [
        ("Streaming Pipeline", test_streaming_pipeline),
        ("Text Augmentation", test_text_augmentation),
        ("Cross Validation", test_cross_validation),
        ("Preprocessing Pipeline", test_preprocessing_pipeline),
        ("Memory Optimization", test_memory_optimization),
        ("Integrated Pipeline", test_integrated_pipeline)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print(f"\n" + "="*70)
            success = test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name}: EXITOSO")
            else:
                print(f"❌ {test_name}: FALLÓ")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Resumen final
    total_time = time.time() - start_time
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "="*70)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 70)
    
    for test_name, success in results.items():
        status = "✅ EXITOSO" if success else "❌ FALLÓ"
        print(f"{test_name:<25}: {status}")
    
    print(f"\nTests exitosos: {successful_tests}/{total_tests}")
    print(f"Tiempo total: {total_time:.2f} segundos")
    
    if successful_tests == total_tests:
        print("\n🎉 TODOS LOS TESTS EXITOSOS")
        print("Strategy 6: Pipeline de Datos Profesional - COMPLETADA")
        return True
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests fallaron")
        print("Revisar implementación de componentes")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)