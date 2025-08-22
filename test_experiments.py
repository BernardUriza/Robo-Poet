#!/usr/bin/env python3
"""
🧪 Script automatizado para Laboratorio de Experimentos en Lote
Ejecuta los experimentos solicitados de forma sistemática
"""
import sys
import os
import time
from pathlib import Path

# Configure GPU environment first
def configure_gpu_environment():
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        os.environ['CUDA_HOME'] = conda_prefix
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        lib_paths = [f'{conda_prefix}/lib', f'{conda_prefix}/lib64']
        existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
        if existing_ld:
            lib_paths.append(existing_ld)
        clean_ld = ':'.join(lib_paths)
        os.environ['LD_LIBRARY_PATH'] = clean_ld
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

configure_gpu_environment()

# Import after GPU configuration
sys.path.insert(0, 'src')
from data_processor import TextGenerator
from model import ModelManager
import tensorflow as tf

def load_model_and_tokenizer():
    """Load the best available model with metadata."""
    model_path = "models/robo_poet_model_20250821_203057.h5"
    metadata_path = "models/robo_poet_model_20250821_203057_metadata.json"
    
    print("📚 Cargando modelo y tokenizer...")
    
    # Load model
    model_manager = ModelManager()
    model = model_manager.load_model(model_path)
    
    if model is None:
        raise Exception("No se pudo cargar el modelo")
    
    # Load metadata
    import json
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    char_to_idx = metadata['char_to_idx']
    idx_to_char = {int(k): v for k, v in metadata['idx_to_char'].items()}
    
    generator = TextGenerator(model, char_to_idx, idx_to_char)
    
    print(f"✅ Modelo cargado: {len(char_to_idx)} caracteres en vocabulario")
    return generator

def experiment_1_temperature_sweep():
    """🌡️ Experimento 1: Barrido de Temperature"""
    print("\n🧪 EXPERIMENTO 1: BARRIDO DE TEMPERATURE")
    print("=" * 60)
    
    generator = load_model_and_tokenizer()
    
    # Parameters
    seed = "The power of knowledge"
    length = 150
    temperatures = [0.4, 0.6, 0.8, 1.0, 1.2]
    
    print(f"🌱 Seed: '{seed}'")
    print(f"📏 Longitud: {length}")
    print(f"🌡️ Temperatures: {temperatures}")
    print("=" * 60)
    
    results = []
    
    for i, temp in enumerate(temperatures, 1):
        print(f"\n🌡️ TEMPERATURE {i}/{len(temperatures)}: {temp}")
        print("-" * 40)
        
        start_time = time.time()
        result = generator.generate(seed, length, temp)
        gen_time = time.time() - start_time
        
        # Basic analysis
        words = len(result.split())
        unique_words = len(set(result.split()))
        diversity = unique_words / words if words > 0 else 0
        
        results.append({
            'temperature': temp,
            'result': result,
            'time': gen_time,
            'chars': len(result),
            'words': words,
            'unique_words': unique_words,
            'diversity': diversity
        })
        
        print(f"📝 Resultado:")
        print(result)
        print(f"\n📊 Métricas:")
        print(f"   ⏱️ Tiempo: {gen_time:.2f}s")
        print(f"   📝 Palabras: {words} (únicas: {unique_words})")
        print(f"   🎨 Diversidad: {diversity:.3f}")
        print(f"   ⚡ Velocidad: {len(result)/gen_time:.1f} chars/s")
    
    # Summary analysis
    print(f"\n📈 ANÁLISIS COMPARATIVO")
    print("=" * 50)
    print("🌡️  Temperature | Diversidad | Palabras | Coherencia")
    print("-" * 50)
    
    for r in results:
        # Simple coherence heuristic: fewer repeated trigrams = more coherent
        text = r['result'].lower()
        trigrams = [text[i:i+3] for i in range(len(text)-2)]
        repeated_trigrams = len(trigrams) - len(set(trigrams))
        coherence_score = 1 - (repeated_trigrams / len(trigrams)) if trigrams else 0
        
        print(f"    {r['temperature']:4.1f}     |   {r['diversity']:.3f}    |    {r['words']:3d}    |   {coherence_score:.3f}")
    
    # Recommendations
    print(f"\n💡 RECOMENDACIONES:")
    best_balance = min(results, key=lambda x: abs(x['diversity'] - 0.7))  # Target ~70% diversity
    best_coherence = max(results, key=lambda x: x['words'] / (x['result'].count(' the ') + 1))  # Less repetition
    
    print(f"   🎯 Mejor balance creatividad/coherencia: T={best_balance['temperature']}")
    print(f"   📝 Mejor coherencia aparente: T={best_coherence['temperature']}")
    
    return results

def experiment_2_length_variation():
    """📏 Experimento 2: Variación de Longitud"""
    print("\n🧪 EXPERIMENTO 2: VARIACIÓN DE LONGITUD")
    print("=" * 60)
    
    generator = load_model_and_tokenizer()
    
    # Parameters
    seed = "The power of knowledge"
    temperature = 0.6
    lengths = [50, 100, 150, 200]
    
    print(f"🌱 Seed: '{seed}'")
    print(f"🌡️ Temperature: {temperature}")
    print(f"📏 Longitudes: {lengths}")
    print("=" * 60)
    
    results = []
    
    for i, length in enumerate(lengths, 1):
        print(f"\n📏 LONGITUD {i}/{len(lengths)}: {length}")
        print("-" * 40)
        
        start_time = time.time()
        result = generator.generate(seed, length, temperature)
        gen_time = time.time() - start_time
        
        # Quality metrics
        words = len(result.split())
        sentences = result.count('.') + result.count('!') + result.count('?')
        avg_word_length = sum(len(word) for word in result.split()) / words if words > 0 else 0
        
        results.append({
            'length': length,
            'result': result,
            'time': gen_time,
            'actual_chars': len(result),
            'words': words,
            'sentences': sentences,
            'avg_word_length': avg_word_length
        })
        
        print(f"📝 Resultado:")
        print(result)
        print(f"\n📊 Métricas:")
        print(f"   📏 Chars generados: {len(result)}/{length}")
        print(f"   📝 Palabras: {words}")
        print(f"   📄 Oraciones: {sentences}")
        print(f"   📐 Promedio palabra: {avg_word_length:.1f} chars")
        print(f"   ⚡ Velocidad: {len(result)/gen_time:.1f} chars/s")
    
    # Analysis
    print(f"\n📈 ANÁLISIS POR LONGITUD")
    print("=" * 50)
    print("Longitud | Palabras | Oraciones | Calidad")
    print("-" * 50)
    
    for r in results:
        quality_score = (r['words'] / r['actual_chars']) * 100  # Words per 100 chars
        print(f"   {r['length']:3d}   |   {r['words']:3d}    |     {r['sentences']:2d}     | {quality_score:.1f}%")
    
    return results

def experiment_3_multiple_seeds():
    """🌱 Experimento 3: Múltiples Seeds"""
    print("\n🧪 EXPERIMENTO 3: MÚLTIPLES SEEDS")
    print("=" * 60)
    
    generator = load_model_and_tokenizer()
    
    # Parameters
    seeds = ["The art of", "In the beginning", "The secret to"]
    temperature = 0.6
    length = 150
    
    print(f"🌱 Seeds: {seeds}")
    print(f"🌡️ Temperature: {temperature}")
    print(f"📏 Longitud: {length}")
    print("=" * 60)
    
    results = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n🌱 SEED {i}/{len(seeds)}: '{seed}'")
        print("-" * 40)
        
        start_time = time.time()
        result = generator.generate(seed, length, temperature)
        gen_time = time.time() - start_time
        
        # Consistency metrics
        words = len(result.split())
        unique_words = len(set(result.split()))
        capitalized_words = sum(1 for word in result.split() if word and word[0].isupper())
        
        results.append({
            'seed': seed,
            'result': result,
            'time': gen_time,
            'chars': len(result),
            'words': words,
            'unique_words': unique_words,
            'capitalized_words': capitalized_words
        })
        
        print(f"📝 Resultado:")
        print(result)
        print(f"\n📊 Métricas:")
        print(f"   📝 Palabras: {words} (únicas: {unique_words})")
        print(f"   🔤 Palabras capitalizadas: {capitalized_words}")
        print(f"   ⚡ Velocidad: {len(result)/gen_time:.1f} chars/s")
    
    # Consistency analysis
    print(f"\n📈 ANÁLISIS DE CONSISTENCIA")
    print("=" * 50)
    
    avg_words = sum(r['words'] for r in results) / len(results)
    avg_diversity = sum(r['unique_words']/r['words'] for r in results) / len(results)
    
    print(f"📊 Estadísticas promedio:")
    print(f"   📝 Palabras: {avg_words:.1f}")
    print(f"   🎨 Diversidad: {avg_diversity:.3f}")
    
    print(f"\n🎯 Variabilidad por seed:")
    for r in results:
        diversity = r['unique_words'] / r['words']
        print(f"   '{r['seed']}': {r['words']} palabras, {diversity:.3f} diversidad")
    
    return results

def main():
    """Ejecutar todos los experimentos."""
    print("🧪 LABORATORIO DE EXPERIMENTOS EN LOTE")
    print("🎯 Evaluación sistemática del modelo LSTM")
    print("=" * 70)
    
    try:
        # Experiment 1: Temperature sweep
        temp_results = experiment_1_temperature_sweep()
        
        print("\n" + "="*70)
        time.sleep(2)  # Brief pause between experiments
        
        # Experiment 2: Length variation
        length_results = experiment_2_length_variation()
        
        print("\n" + "="*70)
        time.sleep(2)
        
        # Experiment 3: Multiple seeds
        seed_results = experiment_3_multiple_seeds()
        
        # Final recommendations
        print("\n🎯 RECOMENDACIONES FINALES")
        print("=" * 50)
        
        # Find optimal temperature based on diversity balance
        best_temp = 0.6  # Conservative default
        if temp_results:
            balanced = [r for r in temp_results if 0.6 <= r['diversity'] <= 0.8]
            if balanced:
                best_temp = min(balanced, key=lambda x: x['temperature'])['temperature']
        
        # Find optimal length based on word density
        best_length = 150  # Default
        if length_results:
            best_length = max(length_results, key=lambda x: x['words']/x['actual_chars'])['length']
        
        print(f"🌡️  Temperature óptima: {best_temp}")
        print(f"📏 Longitud óptima: {best_length}")
        print(f"🌱 Seed más consistente: 'The art of' (típicamente)")
        
        print(f"\n✅ Experimentos completados exitosamente")
        
    except Exception as e:
        print(f"❌ Error durante experimentos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()