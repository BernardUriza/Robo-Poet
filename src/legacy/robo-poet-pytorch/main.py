"""
RoboPoet PyTorch - Main CLI Interface
Created by Bernard Orozco - TensorFlow to PyTorch Migration

Unified command-line interface for training, generation, and evaluation.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RoboPoet PyTorch - Text Generation with GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python main.py train --epochs 25 --batch_size 32
  
  # Generate text
  python main.py generate --checkpoint checkpoints/best.pth --prompt "To be or not to be"
  
  # Interactive generation
  python main.py generate --checkpoint checkpoints/best.pth --interactive
  
  # Create vocabulary
  python main.py vocab --text_path data/processed/unified_corpus.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train GPT model')
    train_parser.add_argument('--data_dir', type=str, default='data/processed',
                             help='Path to processed data directory')
    train_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                             help='Directory to save checkpoints')
    train_parser.add_argument('--log_dir', type=str, default='logs',
                             help='Directory for TensorBoard logs')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=32,
                             help='Batch size for training')
    train_parser.add_argument('--context_length', type=int, default=128,
                             help='Context length for sequences')
    train_parser.add_argument('--learning_rate', type=float, default=6e-4,
                             help='Learning rate')
    train_parser.add_argument('--n_layer', type=int, default=6,
                             help='Number of transformer layers')
    train_parser.add_argument('--n_head', type=int, default=8,
                             help='Number of attention heads')
    train_parser.add_argument('--n_embd', type=int, default=256,
                             help='Embedding dimensions')
    train_parser.add_argument('--dropout', type=float, default=0.1,
                             help='Dropout probability')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    train_parser.add_argument('--no_mixed_precision', action='store_true',
                             help='Disable mixed precision training')
    
    # Generation command  
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--checkpoint', '-c', type=str, required=True,
                           help='Path to model checkpoint')
    gen_parser.add_argument('--prompt', '-p', type=str, default='',
                           help='Text prompt for generation')
    gen_parser.add_argument('--style', '-s', type=str, default='shakespeare',
                           choices=['shakespeare', 'alice', 'neutral'],
                           help='Generation style')
    gen_parser.add_argument('--max_tokens', '-t', type=int, default=200,
                           help='Maximum tokens to generate')
    gen_parser.add_argument('--temperature', type=float, default=0.8,
                           help='Sampling temperature')
    gen_parser.add_argument('--top_k', type=int, default=40,
                           help='Top-k filtering')
    gen_parser.add_argument('--top_p', type=float, default=0.95,
                           help='Nucleus sampling threshold')
    gen_parser.add_argument('--repetition_penalty', type=float, default=1.1,
                           help='Repetition penalty')
    gen_parser.add_argument('--interactive', '-i', action='store_true',
                           help='Start interactive generation session')
    gen_parser.add_argument('--data_dir', type=str, default='data/processed',
                           help='Path to processed data directory')
    
    # Vocabulary command
    vocab_parser = subparsers.add_parser('vocab', help='Create character vocabulary')
    vocab_parser.add_argument('--text_path', type=str, required=True,
                             help='Path to text file')
    vocab_parser.add_argument('--output_path', type=str, 
                             help='Output path for vocabulary (default: same dir as text)')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--data_dir', type=str, default='data/processed',
                            help='Path to processed data directory')
    eval_parser.add_argument('--split', type=str, default='test',
                            choices=['train', 'validation', 'test'],
                            help='Dataset split to evaluate on')
    eval_parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'train':
        # Academic Performance Requirement: GPU Check
        try:
            import torch
            if not torch.cuda.is_available():
                print("[GRAD] ACADEMIC PERFORMANCE REQUIREMENT: GPU/CUDA not available!")
                print("   [BOOKS] Academic training standards require GPU for:")
                print("   • >10x faster training performance")
                print("   • Mixed precision training (FP16)")
                print("   • Large batch processing capabilities")
                print("   • Research-grade benchmarking compliance")
                print("   [FIX] Please install CUDA-enabled PyTorch")
                return 1
            print(f"[FIRE] Academic Performance Mode: Using {torch.cuda.get_device_name(0)}")
        except ImportError:
            print("[X] PyTorch not installed. Please install PyTorch with CUDA support.")
            return 1
            
        from training.train import main as train_main
        
        # Override config with command line args
        import training.train as train_module
        config = {
            'data_dir': args.data_dir,
            'batch_size': args.batch_size,
            'context_length': args.context_length,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'min_lr': args.learning_rate / 100,
            'weight_decay': 0.01,
            'mixed_precision': not args.no_mixed_precision,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'log_interval': 50,
            'num_workers': 4,
            'resume_from': args.resume,
            'n_layer': args.n_layer,
            'n_head': args.n_head,
            'n_embd': args.n_embd,
            'dropout': args.dropout,
            'checkpoint_dir': args.checkpoint_dir,
            'log_dir': args.log_dir
        }
        
        # Monkey patch config for train main
        train_module.CONFIG = config
        train_main()
    
    elif args.command == 'generate':
        from generation.generate import main as generate_main
        
        # Set sys.argv for argparse in generate main
        sys.argv = ['generate.py']
        sys.argv.extend(['--checkpoint', args.checkpoint])
        if args.prompt:
            sys.argv.extend(['--prompt', args.prompt])
        sys.argv.extend(['--style', args.style])
        sys.argv.extend(['--max_tokens', str(args.max_tokens)])
        sys.argv.extend(['--temperature', str(args.temperature)])
        sys.argv.extend(['--top_k', str(args.top_k)])
        sys.argv.extend(['--top_p', str(args.top_p)])
        sys.argv.extend(['--repetition_penalty', str(args.repetition_penalty)])
        sys.argv.extend(['--data_dir', args.data_dir])
        if args.interactive:
            sys.argv.append('--interactive')
            
        generate_main()
    
    elif args.command == 'vocab':
        from utils.create_vocab import create_character_vocabulary
        
        text_path = Path(args.text_path)
        if args.output_path:
            output_path = args.output_path
        else:
            output_path = text_path.parent / 'char_vocabulary.json'
        
        create_character_vocabulary(str(text_path), str(output_path))
        
    elif args.command == 'evaluate':
        print("[CALC] Evaluation functionality coming soon...")
        print(f"   Would evaluate: {args.checkpoint}")
        print(f"   On split: {args.split}")


if __name__ == "__main__":
    print("[LAUNCH] RoboPoet PyTorch - Text Generation Framework")
    print("Created by Bernard Orozco")
    print("=" * 50)
    main()