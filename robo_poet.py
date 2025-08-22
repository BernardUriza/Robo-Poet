#!/usr/bin/env python3
"""
Robo-Poet Academic Framework - Modular Entry Point

Clean, modular entry point for the academic text generation framework.
This replaces the monolithic robo_poet.py with a modern, maintainable architecture.

Author: ML Academic Framework
Version: 2.1 - Modular Architecture
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Import and run orchestrator
from orchestrator import main

if __name__ == "__main__":
    sys.exit(main())