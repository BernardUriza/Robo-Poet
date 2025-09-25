#!/usr/bin/env python3
"""
Setup script for Claude AI Integration in Robo-Poet

Installs dependencies and guides through Claude API key configuration.

Author: Bernard Uriza Orozco
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[X] Python 3.8+ requerido para Claude AI integration")
        print(f"   Version actual: {version.major}.{version.minor}")
        return False
    print(f"[OK] Python {version.major}.{version.minor} compatible")
    return True

def install_claude_dependencies():
    """Install Claude AI dependencies."""
    print("\n[CYCLE] Instalando dependencias de Claude AI...")

    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_claude.txt"
        ], check=True)
        print("[OK] Dependencias instaladas exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] Error instalando dependencias: {e}")
        return False
    except FileNotFoundError:
        print("[X] requirements_claude.txt no encontrado")
        return False

def test_claude_import():
    """Test if Claude can be imported."""
    try:
        import anthropic
        print("[OK] Librería Anthropic importada exitosamente")
        return True
    except ImportError as e:
        print(f"[X] Error importando Anthropic: {e}")
        return False

def setup_api_key():
    """Guide user through API key setup."""
    print("\n CONFIGURACIÓN DE API KEY")
    print("=" * 40)

    print("1. Ve a https://console.anthropic.com/")
    print("2. Crea una cuenta o inicia sesión")
    print("3. Ve a 'API Keys' y crea una nueva key")
    print("4. Copia tu API key")

    api_key = input("\n Pega tu Claude API key aquí: ").strip()

    if not api_key:
        print("[X] API key no puede estar vacía")
        return False

    if not api_key.startswith('sk-'):
        print("WARNING: La API key debería comenzar con 'sk-'")
        confirm = input("¿Continuar de todas formas? (y/N): ").lower()
        if confirm not in ('y', 'yes'):
            return False

    # Set environment variable for current session
    os.environ['CLAUDE_API_KEY'] = api_key

    # Try to create .env file
    try:
        env_file = Path('.env')
        if env_file.exists():
            # Read existing .env
            content = env_file.read_text()
            if 'CLAUDE_API_KEY=' in content:
                # Update existing key
                lines = content.split('\n')
                new_lines = []
                for line in lines:
                    if line.startswith('CLAUDE_API_KEY='):
                        new_lines.append(f'CLAUDE_API_KEY={api_key}')
                    else:
                        new_lines.append(line)
                content = '\n'.join(new_lines)
            else:
                # Add new key
                content += f'\nCLAUDE_API_KEY={api_key}\n'
        else:
            # Create new .env from example
            example_file = Path('.env.example')
            if example_file.exists():
                content = example_file.read_text()
                content = content.replace('CLAUDE_API_KEY=your_claude_api_key_here',
                                        f'CLAUDE_API_KEY={api_key}')
            else:
                content = f'CLAUDE_API_KEY={api_key}\n'

        env_file.write_text(content)
        print("[OK] API key guardada en .env")

    except Exception as e:
        print(f"WARNING: No se pudo guardar en .env: {e}")
        print(f"   Puedes configurar manualmente: export CLAUDE_API_KEY={api_key}")

    return True

def test_claude_connection():
    """Test connection to Claude API."""
    print("\n[CYCLE] Probando conexión con Claude API...")

    try:
        from src.intelligence.claude_integration import test_claude_integration

        if test_claude_integration():
            print("[OK] Conexión con Claude API exitosa")
            return True
        else:
            print("[X] Error en conexión con Claude API")
            return False

    except ImportError:
        print("WARNING: Módulo de integración Claude no encontrado")
        print("   Asegúrate de que el proyecto esté correctamente configurado")
        return False
    except Exception as e:
        print(f"[X] Error probando conexión: {e}")
        return False

def main():
    """Main setup function."""
    print("[AI] CONFIGURACIÓN DE CLAUDE AI PARA ROBO-POET")
    print("=" * 50)

    # Step 1: Check Python version
    if not check_python_version():
        return 1

    # Step 2: Install dependencies
    if not install_claude_dependencies():
        return 1

    # Step 3: Test import
    if not test_claude_import():
        return 1

    # Step 4: Setup API key
    if not setup_api_key():
        return 1

    # Step 5: Test connection
    if not test_claude_connection():
        print("\nWARNING: La configuración básica está completa pero hay problemas de conexión")
        print("   Verifica tu API key y conexión a internet")
        return 0

    # Success
    print("\n CONFIGURACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 50)
    print("[OK] Claude AI está listo para usar")
    print("[OK] Puedes usar la Opción 3 - Ciclo Inteligente")
    print("[OK] Para empezar: python robo_poet.py")

    return 0

if __name__ == "__main__":
    sys.exit(main())