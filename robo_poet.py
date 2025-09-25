#!/usr/bin/env python3
"""
Robo-Poet: Academic Text Generation Framework
Enterprise-grade transformer model training and generation system

Single entry point for all CLI operations and UI interactions.
Architecture: Clean Architecture with Domain-Driven Design
Framework: PyTorch-based with CUDA acceleration support
"""

import sys
from pathlib import Path

def main():
    """Main entry point for Robo-Poet application."""
    # Add src directory to Python path
    src_path = Path(__file__).parent / 'src'
    sys.path.insert(0, str(src_path))

    try:
        # Import orchestrator after path setup
        from orchestrator import main as orchestrator_main
        return orchestrator_main()
    except ImportError as e:
        print(f"Error: Failed to import orchestrator module: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())