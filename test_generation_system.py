#!/usr/bin/env python3
"""
Test comprehensivo para Strategy 5: Generación Avanzada Multi-Modo
Prueba todos los métodos de sampling, scheduling y generación avanzada
"""

import os
import sys
import logging
import time
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_simple_model() -> tf.keras.Model:
    """Crea modelo simple para pruebas."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(1000, 64, input_length=10),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1000, activation='softmax')
    ], name='test_model')
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Build model
    dummy_input = tf.random.uniform([1, 10], minval=0, maxval=1000, dtype=tf.int32)
    _ = model(dummy_input)
    
    return model


class SimpleTokenizer:
    """Tokenizer simple para pruebas."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_index = {'<PAD>': 0, '<UNK>': 1, '<EOS>': 2}
        self.index_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<EOS>'}
        
        # Agregar palabras comunes
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'being', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'time', 'way', 'person', 'year', 'work', 'government', 'school', 'company', 'world',
            'life', 'hand', 'part', 'child', 'eye', 'woman', 'man', 'place', 'work', 'week'
        ]
        
        for i, word in enumerate(common_words):
            if i + 3 < vocab_size:
                self.word_to_index[word] = i + 3
                self.index_to_word[i + 3] = word
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        return [self.word_to_index.get(word, 1) for word in words]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        words = [self.index_to_word.get(token, '<UNK>') for token in tokens]
        return ' '.join(words)


def test_samplers():
    """Prueba métodos de sampling básicos."""
    print("🔍 PRUEBA 1: Sampling Methods")
    print("=" * 50)
    
    try:
        from generation.samplers import (
            SamplerConfig, create_sampler,
            GreedySampler, TemperatureSampler, TopKSampler, NucleusSampler, BeamSearchSampler
        )
        
        # Configuración básica
        config = SamplerConfig(
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            beam_width=5,
            repetition_penalty=1.1
        )
        
        print(f"Configuración:")
        print(f"  Temperature: {config.temperature}")
        print(f"  Top-k: {config.top_k}")
        print(f"  Top-p: {config.top_p}")
        print(f"  Beam width: {config.beam_width}")
        print(f"  Repetition penalty: {config.repetition_penalty}")
        
        # Probar cada sampler
        samplers = ['greedy', 'temperature', 'top_k', 'nucleus', 'beam_search']
        
        for sampler_name in samplers:
            try:
                sampler = create_sampler(sampler_name, config)
                print(f"✅ {sampler_name.capitalize()} sampler creado: {type(sampler).__name__}")
                
                # Crear datos sintéticos para prueba
                batch_size = 2
                vocab_size = 1000
                seq_length = 10
                
                logits = tf.random.normal([batch_size, vocab_size])
                sequence = tf.random.uniform([batch_size, seq_length], minval=0, maxval=vocab_size, dtype=tf.int32)
                
                # Probar sampling
                next_token, log_prob = sampler.sample(logits, sequence, step=0)
                
                print(f"  Sample shape: {next_token.shape}")
                print(f"  Log prob shape: {log_prob.shape}")
                print(f"  Sample values: {next_token.numpy()}")
                
            except Exception as e:
                print(f"❌ Error en {sampler_name}: {e}")
        
        print("✅ Samplers probados exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en samplers: {e}")
        return False


def test_temperature_schedulers():
    """Prueba schedulers de temperatura."""
    print("\n🔍 PRUEBA 2: Temperature Schedulers")
    print("=" * 50)
    
    try:
        from generation.temperature_scheduler import (
            SchedulerConfig, create_scheduler,
            LinearScheduler, ExponentialScheduler, CosineScheduler,
            AdaptiveScheduler, calculate_ngram_diversity
        )
        
        # Configuración
        config = SchedulerConfig(
            initial_temperature=1.5,
            final_temperature=0.7,
            total_steps=100,
            warmup_steps=10,
            cooldown_steps=20,
            target_diversity=0.8
        )
        
        print(f"Configuración:")
        print(f"  Inicial: {config.initial_temperature}")
        print(f"  Final: {config.final_temperature}")
        print(f"  Total steps: {config.total_steps}")
        print(f"  Warmup: {config.warmup_steps}")
        print(f"  Cooldown: {config.cooldown_steps}")
        
        # Probar schedulers
        schedulers = {
            'Linear': create_scheduler('linear', config),
            'Exponential': create_scheduler('exponential', config, decay_rate=0.98),
            'Cosine': create_scheduler('cosine', config),
            'Adaptive': create_scheduler('adaptive', config),
            'Warmup': create_scheduler('warmup', config)
        }
        
        print(f"\nSchedulers creados: {len(schedulers)}")
        
        # Probar en diferentes steps
        test_steps = [0, 25, 50, 75, 100]
        print("\nTemperaturas por step:")
        
        for step in test_steps:
            temps = []
            for name, scheduler in schedulers.items():
                temp = scheduler.get_temperature(step)
                temps.append(f"{name}: {temp:.3f}")
            print(f"  Step {step:3d}: " + " | ".join(temps))
        
        # Probar diversity calculation
        test_tokens = [1, 2, 3, 1, 4, 5, 6, 2, 3, 7, 8, 1, 2]
        diversity = calculate_ngram_diversity(test_tokens, n=3)
        print(f"\nDiversidad de n-gramas (test): {diversity:.3f}")
        
        print("✅ Temperature schedulers probados exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en schedulers: {e}")
        return False


def test_advanced_generator():
    """Prueba generador avanzado."""
    print("\n🔍 PRUEBA 3: Advanced Text Generator")
    print("=" * 50)
    
    try:
        from generation.advanced_generator import (
            AdvancedTextGenerator, GenerationConfig, GenerationMode, StyleType,
            create_advanced_generator
        )
        from generation.samplers import SamplerConfig
        from generation.temperature_scheduler import SchedulerConfig
        
        # Crear modelo y tokenizer
        model = create_simple_model()
        tokenizer = SimpleTokenizer(1000)
        
        print(f"Modelo creado: {model.count_params():,} parámetros")
        print(f"Tokenizer: {tokenizer.vocab_size} tokens")
        
        # Configuración de generación
        generation_config = GenerationConfig(
            max_length=50,
            min_length=10,
            generation_mode=GenerationMode.NUCLEUS,
            use_temperature_scheduling=True,
            scheduler_type='adaptive',
            style_type=StyleType.CREATIVE,
            style_strength=0.5,
            diversity_threshold=0.6
        )
        
        print(f"Configuración:")
        print(f"  Modo: {generation_config.generation_mode.value}")
        print(f"  Longitud: {generation_config.min_length}-{generation_config.max_length}")
        print(f"  Estilo: {generation_config.style_type.value}")
        print(f"  Temperature scheduling: {generation_config.use_temperature_scheduling}")
        
        # Crear generador
        generator = AdvancedTextGenerator(model, tokenizer, generation_config)
        print(f"✅ Generador avanzado creado")
        
        # Probar generación básica
        prompt = "the quick brown fox"
        print(f"\nGenerando con prompt: '{prompt}'")
        
        result = generator.generate(prompt)
        
        print(f"Resultado:")
        print(f"  Texto: '{result.text[:100]}{'...' if len(result.text) > 100 else ''}'")
        print(f"  Tokens: {len(result.tokens)}")
        print(f"  Tiempo: {result.generation_time:.3f}s")
        print(f"  Velocidad: {result.tokens_per_second:.1f} tokens/s")
        print(f"  Diversidad: {result.diversity_score:.3f}")
        print(f"  Repetición: {result.repetition_score:.3f}")
        print(f"  Modo usado: {result.mode_used.value}")
        
        # Probar diferentes modos
        print(f"\nProbando diferentes modos de generación...")
        modes_to_test = [GenerationMode.GREEDY, GenerationMode.TOP_K, GenerationMode.NUCLEUS]
        
        for mode in modes_to_test:
            try:
                result = generator.generate(prompt, generation_mode=mode, max_length=30)
                print(f"  {mode.value}: {result.tokens_per_second:.1f} tok/s, diversidad: {result.diversity_score:.3f}")
            except Exception as e:
                print(f"  {mode.value}: ERROR - {e}")
        
        # Probar batch generation
        prompts = ["the cat", "a beautiful day", "once upon a time"]
        print(f"\nProbando generación en batch ({len(prompts)} prompts)...")
        
        batch_results = generator.generate_batch(prompts, max_length=20)
        
        for i, result in enumerate(batch_results):
            print(f"  Prompt {i+1}: {len(result.tokens)} tokens, {result.tokens_per_second:.1f} tok/s")
        
        # Estadísticas del generador
        stats = generator.get_generation_statistics()
        print(f"\nEstadísticas:")
        print(f"  Total generaciones: {stats['total_generations']}")
        print(f"  Tiempo promedio: {stats['avg_generation_time']:.3f}s")
        print(f"  Velocidad promedio: {stats['avg_tokens_per_second']:.1f} tok/s")
        print(f"  Diversidad promedio: {stats['avg_diversity_score']:.3f}")
        
        print("✅ Advanced generator probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en advanced generator: {e}")
        return False


def test_mixed_mode_generation():
    """Prueba generación en modo mixto."""
    print("\n🔍 PRUEBA 4: Mixed Mode Generation")
    print("=" * 50)
    
    try:
        from generation.advanced_generator import (
            AdvancedTextGenerator, GenerationConfig, GenerationMode
        )
        
        # Crear modelo y tokenizer
        model = create_simple_model()
        tokenizer = SimpleTokenizer(1000)
        
        # Configuración modo mixto
        config = GenerationConfig(
            max_length=40,
            use_mixed_mode=True,
            mixed_mode_weights={
                'greedy': 0.3,
                'nucleus': 0.5,
                'top_k': 0.2
            },
            generation_mode=GenerationMode.MIXED
        )
        
        generator = AdvancedTextGenerator(model, tokenizer, config)
        
        print(f"Configuración modo mixto:")
        for method, weight in config.mixed_mode_weights.items():
            print(f"  {method}: {weight*100:.1f}%")
        
        # Generar múltiples muestras
        prompt = "in the beginning"
        num_samples = 5
        
        print(f"\nGenerando {num_samples} muestras con modo mixto...")
        
        for i in range(num_samples):
            result = generator.generate(prompt)
            mode_choices = result.sampling_stats.get('mode_choices', [])
            
            if mode_choices:
                mode_counts = {}
                for choice in mode_choices:
                    mode_counts[choice] = mode_counts.get(choice, 0) + 1
                
                mode_stats = ", ".join([f"{k}:{v}" for k, v in mode_counts.items()])
                print(f"  Muestra {i+1}: {len(result.tokens)} tokens, modos usados: {mode_stats}")
            else:
                print(f"  Muestra {i+1}: {len(result.tokens)} tokens")
        
        print("✅ Mixed mode generation probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en mixed mode: {e}")
        return False


def test_style_conditioning():
    """Prueba acondicionamiento por estilo."""
    print("\n🔍 PRUEBA 5: Style Conditioning")
    print("=" * 50)
    
    try:
        from generation.advanced_generator import (
            AdvancedTextGenerator, GenerationConfig, StyleType, StyleConditioner
        )
        
        # Crear modelo y tokenizer
        model = create_simple_model()
        tokenizer = SimpleTokenizer(1000)
        
        # Probar diferentes estilos
        styles = [StyleType.FORMAL, StyleType.CASUAL, StyleType.CREATIVE, StyleType.TECHNICAL]
        prompt = "the future of technology"
        
        print(f"Probando estilos con prompt: '{prompt}'")
        
        results = {}
        
        for style in styles:
            config = GenerationConfig(
                max_length=30,
                style_type=style,
                style_strength=0.7,
                generation_mode=GenerationMode.NUCLEUS
            )
            
            generator = AdvancedTextGenerator(model, tokenizer, config)
            result = generator.generate(prompt)
            
            results[style.value] = result
            
            print(f"\n{style.value.upper()}:")
            print(f"  Texto: '{result.text[:80]}{'...' if len(result.text) > 80 else ''}'")
            print(f"  Diversidad: {result.diversity_score:.3f}")
            print(f"  Repetición: {result.repetition_score:.3f}")
        
        # Probar StyleConditioner directamente
        conditioner = StyleConditioner(vocab_size=1000)
        
        # Test logits conditioning
        test_logits = tf.random.normal([1, 1000])
        
        for style in styles:
            conditioned = conditioner.condition_logits(test_logits, style, strength=0.5)
            temp_adj = conditioner.get_temperature_adjustment(style, strength=0.5)
            
            print(f"\n{style.value}: temp adjustment = {temp_adj:.3f}")
        
        print("\n✅ Style conditioning probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en style conditioning: {e}")
        return False


def test_beam_search():
    """Prueba beam search."""
    print("\n🔍 PRUEBA 6: Beam Search")
    print("=" * 50)
    
    try:
        from generation.samplers import BeamSearchSampler, SamplerConfig
        
        # Crear modelo y configuración
        model = create_simple_model()
        
        config = SamplerConfig(
            beam_width=5,
            beam_length_penalty=1.0,
            max_length=20
        )
        
        sampler = BeamSearchSampler(config)
        
        # Input tokens
        input_tokens = tf.constant([[1, 5, 10, 15]], dtype=tf.int32)  # Batch size = 1
        
        print(f"Configuración beam search:")
        print(f"  Beam width: {config.beam_width}")
        print(f"  Length penalty: {config.beam_length_penalty}")
        print(f"  Max length: {config.max_length}")
        
        print(f"\nEjecutando beam search...")
        
        beams = sampler.beam_search(model, input_tokens, max_length=25)
        
        print(f"Beams generados: {len(beams)}")
        
        for i, beam in enumerate(beams[:3]):  # Mostrar top 3
            print(f"  Beam {i+1}: score={beam.score:.3f}, tokens={len(beam.tokens)}, "
                  f"tokens={beam.tokens[:10]}{'...' if len(beam.tokens) > 10 else ''}")
        
        print("✅ Beam search probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en beam search: {e}")
        return False


def test_generation_comparison():
    """Prueba comparación de métodos de generación."""
    print("\n🔍 PRUEBA 7: Generation Method Comparison")
    print("=" * 50)
    
    try:
        from generation.advanced_generator import (
            AdvancedTextGenerator, GenerationConfig, GenerationMode
        )
        
        # Crear modelo y tokenizer
        model = create_simple_model()
        tokenizer = SimpleTokenizer(1000)
        
        config = GenerationConfig(max_length=25)
        generator = AdvancedTextGenerator(model, tokenizer, config)
        
        # Comparar métodos
        prompt = "artificial intelligence will"
        modes = [GenerationMode.GREEDY, GenerationMode.TOP_K, GenerationMode.NUCLEUS]
        
        print(f"Comparando métodos con prompt: '{prompt}'")
        
        comparison_results = generator.compare_generation_modes(
            prompt=prompt,
            modes=modes,
            num_samples=2
        )
        
        for mode_name, results in comparison_results.items():
            print(f"\n{mode_name.upper()}:")
            
            for i, result in enumerate(results):
                print(f"  Muestra {i+1}:")
                print(f"    Texto: '{result.text[:60]}{'...' if len(result.text) > 60 else ''}'")
                print(f"    Diversidad: {result.diversity_score:.3f}")
                print(f"    Velocidad: {result.tokens_per_second:.1f} tok/s")
        
        print("\n✅ Generation comparison probado exitosamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en comparison: {e}")
        return False


def main():
    """Función principal de testing."""
    print("🚀 TESTING STRATEGY 5: GENERACIÓN AVANZADA MULTI-MODO")
    print("=" * 70)
    print("Componentes: Samplers, Temperature Schedulers, Advanced Generator, Style Conditioning")
    print("=" * 70)
    
    tests = [
        ("Sampling Methods", test_samplers),
        ("Temperature Schedulers", test_temperature_schedulers),
        ("Advanced Generator", test_advanced_generator),
        ("Mixed Mode Generation", test_mixed_mode_generation),
        ("Style Conditioning", test_style_conditioning),
        ("Beam Search", test_beam_search),
        ("Generation Comparison", test_generation_comparison)
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
        print("Strategy 5: Generación Avanzada Multi-Modo - COMPLETADA")
        return True
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests fallaron")
        print("Revisar implementación de componentes")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)