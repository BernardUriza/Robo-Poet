#!/usr/bin/env python3
"""
🎨 Robo-Poet: Script de Generación Simplificado
Generación de texto con modelos entrenados sin interfaz de menú
"""
import sys
import argparse
from pathlib import Path
import os

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
# Current directory is already src/, add parent for main modules  
sys.path.insert(0, '..')

def list_available_models():
    """List all available trained models."""
    models_dir = Path('../models')
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob('robo_poet_model_*.keras')) + list(models_dir.glob('robo_poet_model_*.h5'))
    return sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)

def main():
    parser = argparse.ArgumentParser(
        description="🎨 Robo-Poet: Generación de Texto Simplificada",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python robo_generate.py                           # Usar último modelo
  python robo_generate.py --list                   # Ver modelos disponibles
  python robo_generate.py --seed "El poder"        # Texto personalizado
  python robo_generate.py --length 200 --temp 0.8  # Parámetros personalizados
        """
    )
    
    parser.add_argument(
        '--list', 
        action='store_true',
        help='Mostrar modelos disponibles y salir'
    )
    
    parser.add_argument(
        '--model',
        help='Ruta específica del modelo a usar'
    )
    
    parser.add_argument(
        '--seed', 
        default='The ',
        help='Texto inicial para generación (default: "The ")'
    )
    
    parser.add_argument(
        '--length', 
        type=int, 
        default=100,
        help='Longitud del texto a generar (default: 100)'
    )
    
    parser.add_argument(
        '--temperature', '--temp',
        type=float, 
        default=0.7,
        help='Temperatura para sampling (0.5=conservador, 1.0=creativo) (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        models = list_available_models()
    else:
        models = list_available_models()
    
    if args.list or (not args.model and not models):
        print("📊 MODELOS DISPONIBLES")
        print("=" * 40)
        if not models:
            print("❌ No hay modelos entrenados")
            print("💡 Ejecuta primero: python robo_train.py")
            return 1
        
        for i, model in enumerate(models, 1):
            # Try to get model info
            metadata_path = model.parent / (model.stem + '_metadata.json')
            if metadata_path.exists():
                import json
                try:
                    with open(metadata_path) as f:
                        meta = json.load(f)
                    print(f"{i}. {model.name}")
                    print(f"   📅 Creado: {meta.get('training_end_time', 'Desconocido')}")
                    print(f"   📊 Épocas: {meta.get('final_epoch', 'N/A')}")
                    print(f"   📈 Loss: {meta.get('final_loss', 'N/A'):.4f}" if isinstance(meta.get('final_loss'), (int, float)) else f"   📈 Loss: N/A")
                except:
                    print(f"{i}. {model.name} (sin metadata)")
            else:
                print(f"{i}. {model.name} (sin metadata)")
            print()
        
        if args.list:
            return 0
    
    # Select model
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Error: Modelo '{args.model}' no encontrado")
            return 1
    else:
        # Use latest model
        model_path = models[0]
    
    print("🎨 ROBO-POET: GENERACIÓN DE TEXTO")
    print("=" * 50)
    print(f"🤖 Modelo: {model_path.name}")
    print(f"🌱 Seed: '{args.seed}'")
    print(f"📏 Longitud: {args.length}")
    print(f"🌡️  Temperatura: {args.temperature}")
    print()
    
    # Import and generate
    try:
        from data_processor import TextGenerator
        from model import ModelManager
        import tensorflow as tf
        
        # Load model
        print("📚 Cargando modelo...")
        model_manager = ModelManager()
        model = model_manager.load_model(str(model_path))
        
        if model is None:
            print("❌ Error cargando modelo")
            return 1
        
        # Check for metadata file to get tokenizer info
        metadata_path = model_path.parent / (model_path.stem + '_metadata.json')
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            tokenizer_data = metadata  # Use the metadata directly, not nested
        else:
            # Fallback: create simple character-based tokenizer from training text
            print("⚠️ No se encontró metadata, creando tokenizer desde archivo original...")
            text_path = Path('../The+48+Laws+Of+Power_texto.txt')
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                chars = sorted(set(text))
                tokenizer_data = {
                    'chars': chars,
                    'char_to_int': {c: i for i, c in enumerate(chars)},
                    'int_to_char': {i: c for i, c in enumerate(chars)}
                }
            else:
                print("❌ No se puede crear tokenizer sin archivo de texto")
                return 1
        
        # Create generator
        char_to_idx = tokenizer_data.get('char_to_idx', tokenizer_data.get('char_to_int', {}))
        raw_idx_to_char = tokenizer_data.get('idx_to_char', tokenizer_data.get('int_to_char', {}))
        # Ensure idx_to_char has integer keys
        idx_to_char = {int(k): v for k, v in raw_idx_to_char.items()}
        
        generator = TextGenerator(model, char_to_idx, idx_to_char)
        
        print("🎨 Generando texto...")
        print("-" * 50)
        
        # Generate text
        generated_text = generator.generate(
            seed_text=args.seed,
            length=args.length,
            temperature=args.temperature
        )
        
        print(generated_text)
        print("-" * 50)
        print("✅ Generación completada")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error durante generación: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())