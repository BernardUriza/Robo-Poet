#!/usr/bin/env python3
"""
[GRAD] Robo-Poet: Script de Entrenamiento Simplificado
Entrenamiento directo sin interfaz de menú para evitar problemas de terminal interactivo
"""
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="[GRAD] Robo-Poet: Entrenamiento LSTM Simplificado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python robo_train.py                                   # Entrenamiento básico (10 épocas)
  python robo_train.py --epochs 50                      # Entrenamiento intensivo
  python robo_train.py --text mi_archivo.txt --epochs 25 # Personalizado
  
Archivos soportados:
  - The+48+Laws+Of+Power_texto.txt (incluido)
  - Cualquier archivo .txt en UTF-8
        """
    )
    
    parser.add_argument(
        '--text', 
        default='The+48+Laws+Of+Power_texto.txt',
        help='Archivo de texto para entrenar (default: The+48+Laws+Of+Power_texto.txt)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='Número de épocas de entrenamiento (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Check if text file exists (look in parent directory)
    text_path = Path(f'../{args.text}') if not Path(args.text).exists() else Path(args.text)
    if not text_path.exists():
        print(f"[X] Error: No se encontró el archivo '{args.text}'")
        print("\n Archivos disponibles:")
        for txt_file in Path('..').glob('*.txt'):
            print(f"   - {txt_file.name}")
        return 1
    
    # Use the absolute path for training
    args.text = str(text_path)
    
    print("[GRAD] ROBO-POET: ENTRENAMIENTO ACADÉMICO")
    print("=" * 50)
    print(f" Archivo: {args.text}")
    print(f"[TARGET] Épocas: {args.epochs}")
    print(f"[TIME]  Tiempo estimado: {args.epochs * 2} minutos")
    print()
    
    # Import and run robo_poet training
    try:
        # Add parent directory to path for robo_poet imports
        sys.path.insert(0, '..')
        import robo_poet
        success = robo_poet.run_direct_training(args.text, args.epochs)
        
        if success:
            print("\n ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
            print("\n[ART] Próximo paso - Generación de texto:")
            print("   python robo_generate.py")
            return 0
        else:
            print("\n[X] Entrenamiento falló")
            return 1
            
    except Exception as e:
        print(f"[X] Error durante entrenamiento: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())