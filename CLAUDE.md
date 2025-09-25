# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Robo-Poet** is an academic text generation framework that has undergone migration from TensorFlow to PyTorch. It's a modular system for training GPT-style transformer models on literary texts and generating coherent poetry/prose.

### Key Technologies
- **PyTorch** (primary deep learning framework)
- **Python 3.12+**
- **CUDA support** for GPU acceleration (NVIDIA RTX 2000 Ada target)
- **WSL2 + Windows 11** development environment

## Architecture Overview

### Dual-Framework Structure
The project contains two main implementations:

1. **Legacy Modular System** (`src/` directory):
   - Domain-driven design with clean architecture patterns
   - Complex enterprise-style structure with repositories, services, entities
   - Orchestrator-based workflow (`src/orchestrator.py`)

2. **PyTorch Implementation** (`robo-poet-pytorch/` directory):
   - Simplified, research-focused structure
   - Direct CLI interface via `main.py`
   - MinGPT-style transformer architecture

### Core Components

#### Main Entry Points
- `robo_poet.py` - Legacy system orchestrator entry point
- `robo-poet-pytorch/main.py` - PyTorch CLI interface

#### Model Architecture (`src/models/gpt_pytorch.py`)
- MinGPT-inspired transformer with 6 layers, 8 heads, 256 embedding dimensions
- Target: <10M parameters, validation loss <5.0
- Optimized for small literary datasets (Shakespeare, Alice in Wonderland)

#### Configuration System (`src/core/unified_config.py`)
- Hierarchical configuration with environment variable support
- GPU backend configuration (CUDA/ROCM/CPU/AUTO)
- Training hyperparameters and system settings

#### Data Processing
- `src/data/` - Dataset preprocessing and vocabulary building
- `robo-poet-pytorch/src/data/shakespeare_dataset.py` - PyTorch data loading
- Supports multiple text corpora with unified processing

#### Intelligence Layer (NEW)
- `src/intelligence/claude_integration.py` - Claude AI integration for smart training
- `src/interface/phase3_intelligent_cycle.py` - Intelligent training cycle interface
- Real-time dataset optimization using Claude API feedback
- Adaptive training with AI-guided improvements

## Common Development Commands

### Training
```bash
# Legacy system training
python robo_poet.py

# PyTorch direct training
cd robo-poet-pytorch
python main.py train --epochs 25 --batch_size 32

# Quick test training
python robo_poet.py --test quick

# NEW: Intelligent training cycle with Claude AI
python robo_poet.py
# Select option 3: 🧠 FASE 3: Ciclo Inteligente con Claude AI
```

### Claude AI Integration Setup
```bash
# Install Claude AI dependencies
python setup_claude_integration.py

# Manual installation
pip install -r requirements_claude.txt

# Configure API key
export CLAUDE_API_KEY=your_api_key_here
```

### Text Generation
```bash
# PyTorch generation
cd robo-poet-pytorch
python main.py generate --checkpoint checkpoints/best.pth --prompt "To be or not to be"

# Interactive generation
python main.py generate --checkpoint checkpoints/best.pth --interactive
```

### Testing and Validation
```bash
# Module 2 comprehensive test suite
python src/testing/module2_test_suite.py

# Legacy test runner (deprecated)
python src/utils/run_module2_tests.py --quick
```

### Vocabulary Management
```bash
# Create vocabulary from text corpus
cd robo-poet-pytorch
python main.py vocab --text_path data/processed/unified_corpus.txt

# Legacy vocabulary builder
python src/vocabulary_builder.py
```

## Development Guidelines

### Model Training Targets
- **Parameters**: Target <10M for academic demonstrations
- **Loss**: Validation loss <5.0 for coherent generation
- **Context**: 128-token context window (configurable)
- **Generation**: Coherent 200+ token outputs

### GPU Optimization
- CUDA memory optimization with `max_split_size_mb:512`
- Mixed precision training support
- RTX 2000 Ada specific optimizations
- Fallback to CPU if GPU unavailable

### Data Processing
- Text preprocessing in `src/data/dataset_preprocessor.py`
- Character and word-level vocabularies supported
- Multi-corpus processing with metadata tracking
- Processed data stored in `data/processed/`

### Testing Strategy
- Academic validation through `Module2TestSuite`
- Quick tests (~5 minutes) vs full validation (~20 minutes)
- Gradient flow analysis and loss landscape evaluation
- Automated reporting with timestamped outputs

### Intelligent Training (NEW)
- **Claude AI Integration**: Real-time dataset optimization during training
- **Adaptive Cycles**: Train → Evaluate → Consult AI → Improve → Repeat
- **Smart Suggestions**: AI-guided dataset improvements based on model performance
- **Cost Effective**: Uses Claude Haiku model (~$0.25 per full cycle)
- **Fallback Support**: Works without AI when API unavailable

## Project Structure Context

### Legacy System (`src/`)
```
src/
├── core/           # Configuration and exceptions
├── domain/         # Business logic and entities
├── application/    # Commands, queries, services
├── infrastructure/ # Persistence and external services
├── interface/      # UI and workflow phases
├── models/         # Neural network implementations
├── data/           # Data processing pipelines
└── utils/          # Helper utilities
```

### PyTorch System (`robo-poet-pytorch/`)
```
robo-poet-pytorch/
├── src/
│   ├── models/     # GPT model implementation
│   ├── training/   # Training loops and optimization
│   ├── generation/ # Text generation utilities
│   ├── data/       # Dataset handling
│   └── utils/      # Vocabulary and preprocessing
├── data/           # Training data and vocabularies
├── checkpoints/    # Model weights and training states
└── logs/           # TensorBoard and training logs
```

## Important Configuration

### Environment Setup
- Python 3.12+ required
- PyTorch with CUDA support recommended
- WSL2 environment optimizations included
- Environment variables in `.env.example`

### Line Endings
**CRITICAL**: All files must use CRLF line endings (Windows)
- Never convert to LF - breaks code consistency
- Verify `git config core.autocrlf true` and `git config core.eol crlf`

### Git Authorship
- All commits authored by "Bernard Uriza Orozco"
- No AI attribution in commit messages
- Conventional commit format: `feat:`, `fix:`, `refactor:`
- No emojis in commit messages

## Performance and Monitoring

### Key Metrics
- Training loss curves and validation tracking
- Gradient flow analysis for training stability
- Memory usage optimization for RTX 2000 Ada
- Generation quality through perplexity and human evaluation

### Debugging Tools
- Structured logging with timestamped outputs
- Hospital/surgery system for model debugging (`src/hospital/`)
- Gradient analysis utilities
- Loss landscape visualization

## Notes for Development

- The system supports both character-level and word-level tokenization
- Model checkpointing includes optimizer states for resume capability
- TensorBoard logging for training visualization
- Extensive error handling with academic-focused reporting
- Multi-phase training interface (Phase 1: Training, Phase 2: Generation)

When working with this codebase, prefer the PyTorch implementation (`robo-poet-pytorch/`) for new features, as it represents the current direction of the project. The legacy system (`src/`) is maintained for compatibility but is more complex.