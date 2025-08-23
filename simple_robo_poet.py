#!/usr/bin/env python3
"""
Simplified Robo-Poet Entry Point for Educational Use

This provides a streamlined interface that bypasses the enterprise architecture
complexity while still leveraging the advanced features underneath.

Perfect for students and beginners who want to focus on learning text generation
concepts without getting overwhelmed by DDD patterns.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRoboPoet:
    """
    Simplified interface to Robo-Poet functionality.
    
    This class provides an easy-to-understand API that wraps the complex
    enterprise architecture underneath, making it accessible for educational use.
    """
    
    def __init__(self, gpu_enabled: bool = True, debug: bool = True):
        """
        Initialize Simple Robo-Poet.
        
        Args:
            gpu_enabled: Whether to use GPU acceleration
            debug: Enable debug logging for learning
        """
        self.gpu_enabled = gpu_enabled
        self.debug = debug
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("🤖 Welcome to Simple Robo-Poet!")
            logger.info("📚 Educational mode enabled - extra logging for learning")
        
        # Initialize components lazily
        self._model = None
        self._tokenizer = None
        self._generator = None
        self._gpu_manager = None
        
        # Configuration
        self.config = self._create_simple_config()
        
        # Setup GPU if requested
        if gpu_enabled:
            self._setup_gpu()
    
    def _create_simple_config(self) -> Dict[str, Any]:
        """Create simplified configuration for educational use."""
        return {
            "model": {
                "vocab_size": 5000,
                "embedding_dim": 128,
                "lstm_units": [256, 256],
                "dropout": 0.3,
                "sequence_length": 100,
                "batch_size": 32,
                "epochs": 10
            },
            "generation": {
                "max_length": 100,
                "temperature": 1.0,
                "method": "nucleus",
                "top_p": 0.9,
                "top_k": 50
            },
            "training": {
                "learning_rate": 0.001,
                "patience": 3,
                "save_best": True
            }
        }
    
    def _setup_gpu(self):
        """Setup GPU with educational explanations."""
        logger.info("🔧 Setting up GPU for text generation...")
        
        try:
            import tensorflow as tf
            
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger.warning("❌ No GPU found! Falling back to CPU")
                logger.info("💡 Learning tip: GPU training is 10-50x faster than CPU")
                self.gpu_enabled = False
                return
            
            logger.info(f"✅ Found {len(gpus)} GPU(s)")
            
            # Setup memory growth (prevents VRAM allocation issues)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logger.info("💾 Configured dynamic memory growth")
            logger.info("💡 Learning tip: This prevents TensorFlow from allocating all VRAM at once")
            
            # Enable mixed precision for RTX GPUs (automatic speedup)
            if any("RTX" in str(gpu) or "GeForce" in str(gpu) for gpu in gpus):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("⚡ Enabled mixed precision training (FP16)")
                logger.info("💡 Learning tip: Mixed precision can double training speed on RTX GPUs")
            
            # Import GPU optimization modules
            from gpu_optimization.memory_manager import GPUMemoryManager
            from gpu_optimization.mixed_precision import MixedPrecisionManager
            
            self._gpu_manager = GPUMemoryManager()
            logger.info("🚀 GPU setup complete!")
            
        except ImportError as e:
            logger.error(f"❌ Could not import GPU modules: {e}")
            logger.info("💡 Make sure you've installed the GPU optimization components")
            self.gpu_enabled = False
        except Exception as e:
            logger.error(f"❌ GPU setup failed: {e}")
            logger.info("💡 Continuing with CPU - training will be slower but still works!")
            self.gpu_enabled = False
    
    def train(self, text_file: str, model_name: str = "my_model") -> Dict[str, Any]:
        """
        Train a text generation model on your text file.
        
        Args:
            text_file: Path to your text file (.txt)
            model_name: Name for your trained model
            
        Returns:
            Training results and metrics
        """
        logger.info(f"📖 Starting training on: {text_file}")
        
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")
        
        # Step 1: Load and preprocess data
        logger.info("🔄 Step 1: Loading and preprocessing text...")
        training_data, vocab_size = self._prepare_data(text_file)
        self.config["model"]["vocab_size"] = vocab_size
        
        # Step 2: Create model
        logger.info("🧠 Step 2: Creating LSTM model...")
        model = self._create_model()
        
        # Step 3: Train model
        logger.info("🏋️ Step 3: Training model...")
        logger.info("💡 Learning tip: This might take a while depending on your data size")
        
        history = self._train_model(model, training_data)
        
        # Step 4: Save model
        logger.info("💾 Step 4: Saving trained model...")
        model_path = f"models/{model_name}"
        os.makedirs("models", exist_ok=True)
        model.save(f"{model_path}.h5")
        
        # Save tokenizer too
        import pickle
        with open(f"{model_path}_tokenizer.pkl", "wb") as f:
            pickle.dump(self._tokenizer, f)
        
        logger.info(f"✅ Model saved as: {model_path}")
        
        return {
            "model_path": model_path,
            "vocab_size": vocab_size,
            "training_loss": history.history["loss"][-1],
            "epochs_trained": len(history.history["loss"]),
            "performance_tips": [
                "Try different temperatures for generation (0.5-1.5)",
                "Use nucleus sampling (top_p) for better quality",
                "Train longer for better results (more epochs)",
                "Use larger vocab_size for richer vocabulary"
            ]
        }
    
    def generate(
        self, 
        prompt: str, 
        model_name: str = "my_model",
        max_length: int = 100,
        temperature: float = 1.0,
        method: str = "nucleus"
    ) -> str:
        """
        Generate text using your trained model.
        
        Args:
            prompt: Starting text to continue
            model_name: Name of your trained model
            max_length: Maximum length to generate
            temperature: Creativity level (0.5=conservative, 1.5=creative)
            method: Generation method (nucleus, top_k, temperature, greedy)
            
        Returns:
            Generated text
        """
        logger.info(f"✨ Generating text from prompt: '{prompt[:50]}...'")
        
        # Load model if not already loaded
        if self._model is None or self._generator is None:
            logger.info("📂 Loading trained model...")
            self._load_model(model_name)
        
        # Generate text using advanced generation system
        try:
            from generation.advanced_generator import AdvancedTextGenerator, GenerationConfig
            
            if self._generator is None:
                gen_config = GenerationConfig(
                    max_length=max_length,
                    generation_mode=method,
                    temperature=temperature
                )
                self._generator = AdvancedTextGenerator(self._model, self._tokenizer, gen_config)
            
            result = self._generator.generate(prompt)
            
            logger.info("✅ Text generated successfully!")
            logger.info(f"📊 Generation stats:")
            logger.info(f"   - Length: {len(result.tokens)} tokens")
            logger.info(f"   - Speed: {result.tokens_per_second:.1f} tokens/sec")
            logger.info(f"   - Diversity: {result.diversity_score:.3f}")
            
            return result.text
            
        except ImportError:
            logger.info("💡 Using simplified generation (advanced features not available)")
            return self._simple_generate(prompt, max_length, temperature)
    
    def _prepare_data(self, text_file: str):
        """Prepare training data from text file."""
        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"📄 Text loaded: {len(text):,} characters")
        
        # Simple tokenization for educational purposes
        # In a real implementation, you'd use the advanced preprocessing pipeline
        try:
            from data.preprocessing import create_preprocessing_pipeline
            
            logger.info("🔧 Using advanced preprocessing pipeline")
            preprocessor = create_preprocessing_pipeline(
                vocab_size=self.config["model"]["vocab_size"],
                max_sequence_length=self.config["model"]["sequence_length"]
            )
            
            # Fit tokenizer
            preprocessor.fit_tokenizer([text])
            self._tokenizer = preprocessor
            
            # Create sequences
            sequences = []
            encoded = preprocessor.encode(text)
            seq_len = self.config["model"]["sequence_length"]
            
            for i in range(0, len(encoded) - seq_len, seq_len // 2):
                sequences.append(encoded[i:i + seq_len + 1])
            
            logger.info(f"🔢 Created {len(sequences)} training sequences")
            return sequences, len(preprocessor.vocabulary)
            
        except ImportError:
            logger.info("💡 Using simplified preprocessing (advanced features not available)")
            return self._simple_tokenize(text)
    
    def _simple_tokenize(self, text: str):
        """Simple tokenization fallback."""
        # Basic word tokenization
        words = text.lower().split()
        
        # Build vocabulary
        from collections import Counter
        word_counts = Counter(words)
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [
            word for word, count in word_counts.most_common(self.config["model"]["vocab_size"] - 4)
        ]
        
        word_to_id = {word: i for i, word in enumerate(vocab)}
        self._tokenizer = {"vocab": vocab, "word_to_id": word_to_id, "id_to_word": {i: word for word, i in word_to_id.items()}}
        
        # Create sequences
        sequences = []
        seq_len = self.config["model"]["sequence_length"]
        
        # Convert to IDs
        ids = [word_to_id.get(word, 1) for word in words]  # 1 = <UNK>
        
        for i in range(0, len(ids) - seq_len, seq_len // 2):
            sequences.append(ids[i:i + seq_len + 1])
        
        logger.info(f"🔢 Vocabulary size: {len(vocab)}")
        logger.info(f"🔢 Training sequences: {len(sequences)}")
        
        return sequences, len(vocab)
    
    def _create_model(self):
        """Create LSTM model."""
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        config = self.config["model"]
        
        logger.info("🏗️ Building LSTM architecture:")
        logger.info(f"   - Vocabulary: {config['vocab_size']:,} words")
        logger.info(f"   - Embedding: {config['embedding_dim']} dimensions")
        logger.info(f"   - LSTM layers: {config['lstm_units']}")
        logger.info(f"   - Dropout: {config['dropout']}")
        
        model = models.Sequential([
            # Embedding layer - converts word IDs to dense vectors
            layers.Embedding(
                config["vocab_size"],
                config["embedding_dim"],
                input_length=config["sequence_length"],
                name="embedding"
            ),
            
            # First LSTM layer - learns sequential patterns
            layers.LSTM(
                config["lstm_units"][0],
                return_sequences=True,
                dropout=config["dropout"],
                name="lstm_1"
            ),
            
            # Second LSTM layer - learns higher-level patterns  
            layers.LSTM(
                config["lstm_units"][1],
                dropout=config["dropout"],
                name="lstm_2"
            ),
            
            # Output layer - predicts next word probabilities
            layers.Dense(config["vocab_size"], activation="softmax", name="output")
        ])
        
        # Compile model
        optimizer = "adam"
        if self.gpu_enabled:
            # Use mixed precision optimizer if GPU available
            try:
                from tensorflow.keras.mixed_precision import LossScaleOptimizer
                base_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["training"]["learning_rate"])
                optimizer = LossScaleOptimizer(base_optimizer)
                logger.info("⚡ Using mixed precision optimizer")
            except:
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["training"]["learning_rate"])
        
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Print model summary for educational purposes
        if self.debug:
            logger.info("📋 Model architecture:")
            model.summary(print_fn=logger.info)
        
        total_params = model.count_params()
        logger.info(f"🔢 Total parameters: {total_params:,}")
        
        # Estimate memory usage
        param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        logger.info(f"💾 Estimated model size: {param_size_mb:.1f} MB")
        
        return model
    
    def _train_model(self, model, training_data):
        """Train the model with educational logging."""
        import tensorflow as tf
        import numpy as np
        
        # Prepare training data
        X, y = [], []
        for sequence in training_data:
            X.append(sequence[:-1])  # Input: all but last token
            y.append(sequence[-1])   # Target: last token
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"📊 Training data shape: X={X.shape}, y={y.shape}")
        
        # Callbacks for better training
        callbacks = []
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=self.config["training"]["patience"],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpointing
        if self.config["training"]["save_best"]:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Custom callback for educational logging
        class LearningCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    logger.info(f"📈 Epoch {epoch+1}: loss={logs['loss']:.4f}, accuracy={logs.get('accuracy', 0):.4f}")
                    
                    # Educational tips based on training progress
                    if epoch == 0:
                        logger.info("💡 Learning tip: Loss should decrease over time")
                    elif epoch == 2:
                        logger.info("💡 Learning tip: If loss stops improving, try adjusting learning rate")
        
        callbacks.append(LearningCallback())
        
        # Train model
        logger.info("🏋️ Starting training...")
        logger.info("💡 Learning tip: This is where the magic happens - the model learns patterns in your text!")
        
        history = model.fit(
            X, y,
            batch_size=self.config["model"]["batch_size"],
            epochs=self.config["model"]["epochs"],
            callbacks=callbacks,
            verbose=1 if self.debug else 2
        )
        
        logger.info("✅ Training completed!")
        
        # Training summary
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history.get('accuracy', [0])[-1]
        
        logger.info(f"📊 Final training metrics:")
        logger.info(f"   - Loss: {final_loss:.4f}")
        logger.info(f"   - Accuracy: {final_accuracy:.4f}")
        
        # Educational interpretation
        if final_loss < 2.0:
            logger.info("🎉 Excellent! Your model learned the patterns well")
        elif final_loss < 4.0:
            logger.info("👍 Good training! Model should generate decent text")
        else:
            logger.info("🤔 Model might need more training or different parameters")
        
        self._model = model
        return history
    
    def _load_model(self, model_name: str):
        """Load a trained model."""
        import tensorflow as tf
        import pickle
        
        model_path = f"models/{model_name}.h5"
        tokenizer_path = f"models/{model_name}_tokenizer.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self._model = tf.keras.models.load_model(model_path)
        
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                self._tokenizer = pickle.load(f)
        
        logger.info(f"✅ Model loaded: {model_name}")
    
    def _simple_generate(self, prompt: str, max_length: int, temperature: float) -> str:
        """Simple generation fallback."""
        import numpy as np
        
        # Convert prompt to IDs
        if hasattr(self._tokenizer, 'encode'):
            input_ids = self._tokenizer.encode(prompt)
        else:
            words = prompt.lower().split()
            input_ids = [self._tokenizer["word_to_id"].get(word, 1) for word in words]
        
        # Generate tokens
        generated_ids = input_ids.copy()
        
        for _ in range(max_length):
            # Prepare input
            sequence = generated_ids[-self.config["model"]["sequence_length"]:]
            sequence = np.pad(sequence, (self.config["model"]["sequence_length"] - len(sequence), 0), constant_values=0)
            
            # Predict next token
            predictions = self._model.predict(np.array([sequence]), verbose=0)
            predictions = predictions[0] / temperature
            
            # Sample from probabilities
            probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
            next_token = np.random.choice(len(probabilities), p=probabilities)
            
            generated_ids.append(next_token)
            
            # Stop if end token
            if next_token == 3:  # <END>
                break
        
        # Convert back to text
        if hasattr(self._tokenizer, 'decode'):
            return self._tokenizer.decode(generated_ids)
        else:
            words = [self._tokenizer["id_to_word"].get(id, '<UNK>') for id in generated_ids]
            return ' '.join(words)
    
    def interactive_mode(self):
        """Interactive mode for experimentation."""
        logger.info("🎮 Entering interactive mode!")
        logger.info("💡 Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                command = input("\\nRobo-Poet> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                elif command.lower() == 'help':
                    self._show_help()
                elif command.startswith('train '):
                    text_file = command.split(' ', 1)[1]
                    self.train(text_file)
                elif command.startswith('generate '):
                    prompt = command.split(' ', 1)[1]
                    result = self.generate(prompt)
                    print(f"\\n📝 Generated text:\\n{result}\\n")
                elif command.startswith('config'):
                    self._show_config()
                else:
                    logger.info("❓ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                logger.info("\\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show interactive help."""
        help_text = """
🤖 Simple Robo-Poet Commands:

📚 TRAINING:
  train <file.txt>           Train a model on your text file
  
✨ GENERATION:  
  generate <prompt>          Generate text from a prompt
  
⚙️ CONFIGURATION:
  config                     Show current configuration
  
❓ HELP:
  help                       Show this help message
  quit                       Exit interactive mode

💡 EXAMPLES:
  train my_story.txt
  generate "Once upon a time"
  generate "The future of AI is"
        """
        print(help_text)
    
    def _show_config(self):
        """Show current configuration."""
        print("\\n⚙️ Current Configuration:")
        print("=" * 40)
        print(f"GPU Enabled: {self.gpu_enabled}")
        print(f"Debug Mode: {self.debug}")
        print("\\nModel Settings:")
        for key, value in self.config["model"].items():
            print(f"  {key}: {value}")
        print("\\nGeneration Settings:")
        for key, value in self.config["generation"].items():
            print(f"  {key}: {value}")
        print("=" * 40)


def main():
    """Main entry point for Simple Robo-Poet."""
    parser = argparse.ArgumentParser(
        description="Simple Robo-Poet - Learn Text Generation the Easy Way!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_robo_poet.py --train my_text.txt
  python simple_robo_poet.py --generate "Hello world" --model my_model
  python simple_robo_poet.py --interactive
  
Educational Features:
  - Detailed logging explains what's happening
  - GPU optimization happens automatically
  - Simple API hides complex architecture
  - Interactive mode for experimentation
        """
    )
    
    parser.add_argument("--train", help="Train model on text file")
    parser.add_argument("--generate", help="Generate text from prompt")
    parser.add_argument("--model", default="my_model", help="Model name to use")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation creativity (0.5-1.5)")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum length to generate")
    parser.add_argument("--method", default="nucleus", choices=["nucleus", "top_k", "temperature", "greedy"], 
                       help="Generation method")
    
    args = parser.parse_args()
    
    # Create Simple Robo-Poet
    robo_poet = SimpleRoboPoet(
        gpu_enabled=not args.no_gpu,
        debug=not args.quiet
    )
    
    try:
        if args.train:
            logger.info(f"🚀 Training mode: {args.train}")
            result = robo_poet.train(args.train, args.model)
            print("\\n✅ Training Complete!")
            print("=" * 50)
            for tip in result["performance_tips"]:
                print(f"💡 {tip}")
            
        elif args.generate:
            logger.info(f"✨ Generation mode: '{args.generate}'")
            generated_text = robo_poet.generate(
                args.generate,
                args.model,
                args.max_length,
                args.temperature,
                args.method
            )
            print("\\n📝 Generated Text:")
            print("=" * 50)
            print(generated_text)
            print("=" * 50)
            
        elif args.interactive:
            robo_poet.interactive_mode()
            
        else:
            parser.print_help()
            print("\\n💡 Try --interactive mode for experimentation!")
    
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        logger.info("💡 Make sure your file path is correct")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if robo_poet.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()