#!/usr/bin/env python3
"""
Script para limpiar textos de Project Gutenberg y prepararlos para entrenamiento.
Remueve headers, footers, y metadatos, dejando solo el contenido principal.
"""

import re
import sys

def clean_gutenberg_text(filename, output_filename, title_marker):
    """
    Limpia un archivo de Project Gutenberg.
    
    Args:
        filename: Archivo de entrada
        output_filename: Archivo de salida limpio
        title_marker: Marcador que indica el inicio del contenido principal
    """
    
    print(f"📖 Procesando: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📝 Tamaño original: {len(content):,} caracteres")
    
    # Encontrar el inicio del contenido principal
    start_pos = 0
    if title_marker in content:
        start_pos = content.find(title_marker)
        print(f"✅ Encontrado marcador de inicio: '{title_marker[:30]}...'")
    else:
        # Buscar patrones alternativos
        patterns = [
            r"CHAPTER I\.",
            r"ACT I\.",
            r"SCENE I\.",
            r"\*\*\* START OF .*\*\*\*",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                start_pos = match.start()
                print(f"✅ Encontrado patrón alternativo: '{pattern}'")
                break
    
    # Encontrar el final del contenido (antes de footer de Gutenberg)
    end_patterns = [
        r"\*\*\* END OF .*\*\*\*",
        r"End of Project Gutenberg",
        r"End of the Project Gutenberg",
        r"FINIS\.",
        r"THE END\.",
    ]
    
    end_pos = len(content)
    for pattern in end_patterns:
        match = re.search(pattern, content[start_pos:])
        if match:
            end_pos = start_pos + match.start()
            print(f"✅ Encontrado marcador de final: '{pattern}'")
            break
    
    # Extraer contenido principal
    main_content = content[start_pos:end_pos]
    
    # Limpiezas adicionales
    main_content = clean_text_formatting(main_content)
    
    print(f"✨ Tamaño limpio: {len(main_content):,} caracteres")
    print(f"📉 Reducción: {len(content) - len(main_content):,} caracteres")
    
    # Guardar archivo limpio
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(main_content)
    
    print(f"💾 Guardado como: {output_filename}")
    
    # Mostrar muestra
    lines = main_content.split('\n')[:5]
    print(f"\n📋 Primeras líneas:")
    for i, line in enumerate(lines, 1):
        if line.strip():
            print(f"{i:2d}: {line.strip()[:80]}{'...' if len(line.strip()) > 80 else ''}")

def clean_text_formatting(text):
    """Aplica limpiezas de formato al texto."""
    
    # Remover headers de capítulos excesivos
    text = re.sub(r'^.*Produced by.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*Updated:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*Language:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*Release Date:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^.*Character set encoding:.*$', '', text, flags=re.MULTILINE)
    
    # Remover líneas de separadores excesivas
    text = re.sub(r'^\*+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^=+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-+$', '', text, flags=re.MULTILINE)
    
    # Limpiar espacios múltiples pero preservar estructura
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 líneas vacías
    text = re.sub(r' +', ' ', text)  # Espacios múltiples a uno
    text = text.strip()
    
    return text

def main():
    files_to_clean = [
        {
            'input': 'alice_raw.txt',
            'output': 'alice_wonderland.txt',
            'marker': "CHAPTER I."
        },
        {
            'input': 'hamlet_raw.txt', 
            'output': 'hamlet_shakespeare.txt',
            'marker': "ACT I."
        }
    ]
    
    print("🧹 LIMPIANDO TEXTOS DE PROJECT GUTENBERG")
    print("=" * 50)
    
    for file_info in files_to_clean:
        try:
            clean_gutenberg_text(
                file_info['input'], 
                file_info['output'], 
                file_info['marker']
            )
            print()
        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {file_info['input']}")
        except Exception as e:
            print(f"❌ Error procesando {file_info['input']}: {e}")
    
    print("✅ Limpieza completada!")
    print("\n📊 ARCHIVOS LISTOS PARA ENTRENAMIENTO:")
    print("   - alice_wonderland.txt")
    print("   - hamlet_shakespeare.txt")

if __name__ == "__main__":
    main()