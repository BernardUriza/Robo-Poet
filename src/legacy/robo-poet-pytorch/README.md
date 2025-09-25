# 🚀 RoboPoet PyTorch Edition

**Creado por Bernard Orozco - TensorFlow to PyTorch Migration**

## Overview
PyTorch implementation of RoboPoet text generation framework, migrated from TensorFlow LSTM to modern GPT architecture.

## Architecture
- **Model**: MinGPT/NanoGPT-style Transformer
- **Parameters**: <10M (optimized for small dataset)
- **Context Length**: 128 tokens
- **Vocabulary**: 6,725 tokens (Shakespeare + Alice corpus)

## Target Performance
- **Validation Loss**: <5.0 (vs TensorFlow LSTM baseline: 6.5)
- **Training Speed**: >1000 tokens/sec on RTX 2000 Ada
- **Memory Usage**: <6GB VRAM
- **Generation Length**: 200+ coherent tokens

## Quick Start

### Environment Setup
```bash
conda activate robo-poet-pytorch
python src/train.py --config configs/gpt_small.yaml
```

### Generation
```bash
python src/generate.py --model checkpoints/best.pth --prompt "To be or not to be"
```

## Project Structure
```
robo-poet-pytorch/
├── src/
│   ├── models/          # GPT architecture
│   ├── data/           # Dataset classes
│   ├── training/       # Training loops
│   ├── generation/     # Text generation
│   └── utils/          # Utilities
├── data/
│   ├── processed/      # Unified Shakespeare+Alice corpus
│   └── corpus/         # Raw text files
├── configs/            # YAML configurations
├── checkpoints/        # Model checkpoints
├── logs/              # Tensorboard logs
└── tests/             # Unit tests
```

## Hardware Requirements - 🎓 ACADEMIC PERFORMANCE: GPU MANDATORY

⚠️ **IMPORTANT**: This implementation enforces **mandatory GPU usage** for academic performance standards.

- **GPU**: NVIDIA RTX 2000 Ada (8GB VRAM) - **REQUIRED**
- **CUDA**: 11.8+ - **REQUIRED** 
- **PyTorch**: 2.1.0+cu118 (CUDA version) - **REQUIRED**

### Academic Justification:
- **Performance**: >10x faster training than CPU
- **Mixed Precision**: FP16 training requires tensor cores
- **Batch Processing**: Large batches need GPU parallelization
- **Research Standards**: Academic benchmarks require GPU timing

## Migration Status - ✅ COMPLETED
- ✅ Environment setup
- ✅ Data pipeline (ShakespeareDataset with full features)
- ✅ GPT model architecture (9.8M parameters, MinGPT-style)
- ✅ Training loop (mixed precision, early stopping, checkpointing)
- ✅ Generation pipeline (interactive, multiple sampling strategies)
- ✅ CLI interface (train, generate, vocab commands)
- ✅ Ready for production use

---
*This is the PyTorch evolution of the TensorFlow RoboPoet framework, optimized for modern transformer architectures while maintaining the educational focus and enterprise structure.*