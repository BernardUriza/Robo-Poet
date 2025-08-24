#!/usr/bin/env python3
"""
Demo del Sistema Multi-Corpus
Creado por Bernard Orozco

Demostración del nuevo sistema que automáticamente usa todos los archivos .txt 
de la carpeta corpus/ para entrenar modelos más ricos y diversos.
"""

from pathlib import Path
import sys

def demo_multi_corpus_system():
    """Demostrar las capacidades del sistema multi-corpus."""
    
    print("🎯 DEMO: SISTEMA MULTI-CORPUS ROBO-POET")
    print("=" * 60)
    print("🚀 ¡El sistema ahora entrena con MÚLTIPLES textos automáticamente!")
    print()
    
    # Verificar estructura
    corpus_path = Path("corpus")
    
    print("📁 ESTRUCTURA DEL SISTEMA:")
    print("-" * 40)
    
    if corpus_path.exists():
        txt_files = list(corpus_path.glob("*.txt"))
        
        if txt_files:
            print(f"✅ Carpeta corpus/: {len(txt_files)} archivos encontrados")
            
            total_size = 0
            for txt_file in sorted(txt_files):
                size = txt_file.stat().st_size
                total_size += size
                print(f"   📖 {txt_file.name}: {size:,} bytes")
            
            print(f"\n📊 CORPUS COMBINADO:")
            print(f"   Total archivos: {len(txt_files)}")
            print(f"   Tamaño total: {total_size:,} bytes ({total_size/1024:.1f} KB)")
            
        else:
            print("📭 Carpeta corpus/ existe pero está vacía")
    else:
        print("❌ Carpeta corpus/ no encontrada")
    
    print(f"\n🎓 CÓMO USAR EL NUEVO SISTEMA:")
    print("-" * 40)
    print("1️⃣ Agrega tus archivos .txt a la carpeta corpus/:")
    print("   cp mi_libro.txt corpus/")
    print("   cp otra_novela.txt corpus/")
    print("   cp poemas.txt corpus/")
    print()
    
    print("2️⃣ Entrena un modelo con TODO el corpus automáticamente:")
    print("   python robo_poet.py --model mi_modelo_completo --epochs 20")
    print()
    
    print("3️⃣ El sistema combinará TODOS los textos para:")
    print("   ✨ Vocabulario más rico y diverso")
    print("   ✨ Estilos múltiples en un solo modelo")
    print("   ✨ Mayor capacidad de generalización")
    print("   ✨ Textos generados más interesantes")
    print()
    
    print("🔬 COMANDOS DISPONIBLES:")
    print("-" * 40)
    print("# Entrenamiento multi-corpus:")
    print("python robo_poet.py --model shakespeare_alice --epochs 15")
    print()
    
    print("# Análisis con datos combinados:")
    print("python robo_poet.py --analyze modelo.keras")
    print("python robo_poet.py --minima modelo.keras")
    print()
    
    print("# Tests usando corpus completo:")
    print("python robo_poet.py --test quick")
    print()
    
    print("⚡ VENTAJAS DEL MULTI-CORPUS:")
    print("-" * 40)
    print("📚 Ya no necesitas elegir UN solo texto")
    print("🎭 Combina diferentes géneros y estilos")
    print("🧠 Modelos más inteligentes y versátiles") 
    print("🎯 Generación más rica e impredecible")
    print("📈 Mayor diversidad de vocabulario")
    print()
    
    if corpus_path.exists() and list(corpus_path.glob("*.txt")):
        print("✅ Tu sistema está LISTO para entrenamiento multi-corpus!")
    else:
        print("💡 Solución rápida:")
        print("   mkdir corpus")
        print("   cp *.txt corpus/  # Si tienes archivos txt en el directorio actual")
    print()
    
    print("🦁 ¡Sistema multi-corpus listo por Aslan!")
    print("🧉 Todo preparado como un buen mate compartido entre múltiples literaturas")

if __name__ == "__main__":
    demo_multi_corpus_system()