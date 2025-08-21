#!/usr/bin/env python3
"""
Robo-Poet: Academic Neural Text Generation Framework

A comprehensive implementation of LSTM-based character-level text generation,
optimized for educational purposes and GPU acceleration.

Author: Student ML Researcher
Version: 1.0.0
Hardware: Optimized for NVIDIA RTX 2000 Ada
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append('src')

from config import get_config, GPUConfigurator
from data_processor import TextProcessor, TextGenerator
from model import LSTMTextGenerator, ModelTrainer, ModelManager

class RoboPoet:
    """Main orchestrator class for the text generation pipeline."""
    
    def __init__(self):
        """Initialize Robo-Poet with configuration."""
        self.model_config, self.system_config = get_config()
        self.text_processor = TextProcessor(
            sequence_length=self.model_config.sequence_length,
            step_size=self.model_config.step_size
        )
        self.model = None
        self.trainer = None
        
    def setup_environment(self) -> str:
        """
        Configure hardware and environment.
        
        Returns:
            Device string for computation
        """
        print("🚀 Initializing Robo-Poet Environment")
        print("=" * 60)
        
        # Configure GPU
        gpu_available = GPUConfigurator.setup_gpu()
        device = GPUConfigurator.get_device_strategy()
        
        print(f"💻 Computation device: {device}")
        
        if gpu_available:
            print("🎉 RTX 2000 Ada detected and configured!")
        else:
            print("📚 CPU mode - perfect for learning ML concepts")
        
        return device
    
    def prepare_data(self, text_file: str) -> tuple:
        """
        Load and prepare training data.
        
        Args:
            text_file: Path to input text file
            
        Returns:
            Tuple of prepared training data (X, y)
        """
        print("\n📚 Data Preparation Phase")
        print("=" * 60)
        
        # Prepare data
        X, y = self.text_processor.prepare_data(
            text_file, 
            max_length=self.model_config.max_text_length
        )
        
        return X, y
    
    def build_model(self) -> LSTMTextGenerator:
        """
        Construct and compile the neural network model.
        
        Returns:
            Built LSTM text generator
        """
        print("\n🧠 Model Architecture Phase")
        print("=" * 60)
        
        # Create model
        lstm_generator = LSTMTextGenerator(
            vocab_size=self.text_processor.vocab_size,
            sequence_length=self.model_config.sequence_length,
            lstm_units=self.model_config.lstm_units,
            dropout_rate=self.model_config.dropout_rate
        )
        
        # Build architecture
        model = lstm_generator.build_model()
        
        # Print summary
        print("\n📋 Model Architecture Summary:")
        print(lstm_generator.get_model_summary())
        
        return lstm_generator
    
    def train_model(self, lstm_generator: LSTMTextGenerator, X, y, device: str):
        """
        Train the model with prepared data.
        
        Args:
            lstm_generator: LSTM model generator
            X: Input training data
            y: Target training data
            device: Computation device
        """
        print("\n⚡ Training Phase")
        print("=" * 60)
        
        # Initialize trainer
        self.trainer = ModelTrainer(lstm_generator.model, device)
        
        # Train model
        history = self.trainer.train(
            X, y,
            batch_size=self.model_config.batch_size,
            epochs=self.model_config.epochs,
            validation_split=self.model_config.validation_split
        )
        
        return history
    
    def generate_samples(self, model, seeds: list = None) -> None:
        """
        Generate text samples using the trained model.
        
        Args:
            model: Trained model
            seeds: List of seed texts for generation
        """
        print("\n🎨 Text Generation Phase")
        print("=" * 60)
        
        # Default seeds if none provided
        if seeds is None:
            seeds = ["power is", "the law", "never", "always"]
        
        # Initialize generator
        generator = TextGenerator(
            model,
            self.text_processor.char_to_idx,
            self.text_processor.idx_to_char
        )
        
        # Generate samples
        for i, seed in enumerate(seeds, 1):
            print(f"\n🌱 Sample {i} - Seed: '{seed}'")
            print("-" * 40)
            
            generated = generator.generate(
                seed_text=seed,
                length=150,
                temperature=0.8
            )
            
            print(f"📝 Generated text:")
            print(f"{generated}")
            print()
    
    def run_complete_pipeline(self, text_file: str, seeds: list = None):
        """
        Execute the complete text generation pipeline.
        
        Args:
            text_file: Path to training text file
            seeds: Optional list of generation seeds
        """
        try:
            # Setup environment
            device = self.setup_environment()
            
            # Prepare data
            X, y = self.prepare_data(text_file)
            
            # Build model
            lstm_generator = self.build_model()
            
            # Train model
            self.train_model(lstm_generator, X, y, device)
            
            # Generate samples
            self.generate_samples(lstm_generator.model, seeds)
            
            print("\n🎉 Pipeline completed successfully!")
            print("   Your first neural text generator is ready!")
            
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robo-Poet: Neural Text Generation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robo_poet.py --text "The+48+Laws+Of+Power_texto.txt"
  python robo_poet.py --text corpus.txt --seeds "power" "law" "strategy"
        """
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        default="The+48+Laws+Of+Power_texto.txt",
        help='Path to training text file (default: The+48+Laws+Of+Power_texto.txt)'
    )
    
    parser.add_argument(
        '--seeds', '-s',
        nargs='*',
        default=None,
        help='Seed texts for generation (default: ["power is", "the law", "never", "always"])'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=128,
        help='Training batch size (default: 128)'
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # ASCII Art Header
    print("""
╭─────────────────────────────────────────────────╮
│   🤖 ROBO-POET: Neural Text Generation v1.0    │
│   Academic LSTM Implementation                  │
│   Optimized for RTX 2000 Ada                   │
╰─────────────────────────────────────────────────╯
    """)
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize framework
    robo_poet = RoboPoet()
    
    # Override config with CLI args
    robo_poet.model_config.epochs = args.epochs
    robo_poet.model_config.batch_size = args.batch_size
    
    # Run pipeline
    robo_poet.run_complete_pipeline(
        text_file=args.text,
        seeds=args.seeds
    )

if __name__ == "__main__":
    main()