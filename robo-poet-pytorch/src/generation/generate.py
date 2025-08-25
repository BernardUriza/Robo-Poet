"""
Text generation script for RoboPoet PyTorch GPT model.
Created by Bernard Orozco - TensorFlow to PyTorch Migration

Provides interactive text generation with multiple sampling strategies.
Features temperature, top-k, and nucleus (top-p) sampling.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.gpt_model import GPT, GPTConfig, create_model
from data.shakespeare_dataset import ShakespeareDataset


class TextGenerator:
    """
    Text generator with multiple sampling strategies and style control.
    
    Features:
    - Temperature sampling
    - Top-k filtering  
    - Nucleus (top-p) sampling
    - Repetition penalty
    - Style prompting for Shakespeare/Alice
    - Interactive generation
    """
    
    def __init__(self, model: GPT, dataset: ShakespeareDataset, device: torch.device):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.model.eval()
        
        # Style prompts for different genres
        self.style_prompts = {
            'shakespeare': 'To be or not to be, that is the question:',
            'alice': 'Alice was beginning to get very tired of sitting by her sister',
            'neutral': 'Once upon a time',
            'custom': ''
        }
        
        print(f"🎭 Text Generator initialized:")
        print(f"   🤖 Model: {model.get_num_params():,} parameters")
        print(f"   🔤 Vocabulary: {dataset.get_vocab_size()} tokens")
        print(f"   📏 Context length: {model.config.block_size}")
        print(f"   🎮 Device: {device}")
    
    @torch.no_grad()
    def generate_text(
        self,
        prompt: str = "",
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """
        Generate text with advanced sampling strategies.
        
        Args:
            prompt: Initial text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (keep only k most likely tokens)
            top_p: Nucleus sampling (keep tokens with cumulative probability <= p)
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of stop sequences
            
        Returns:
            Generated text string
        """
        if not prompt:
            prompt = self.style_prompts['neutral']
        
        # Encode prompt
        input_ids = self.dataset.encode(prompt).unsqueeze(0).to(self.device)  # [1, seq_len]
        generated_ids = input_ids.clone()
        
        print(f"🎯 Generating with:")
        print(f"   📝 Prompt: '{prompt}'")
        print(f"   🌡️  Temperature: {temperature}")
        print(f"   🔝 Top-k: {top_k}")
        print(f"   🎯 Top-p: {top_p}")
        print(f"   🔄 Rep. penalty: {repetition_penalty}")
        print(f"   📏 Max tokens: {max_tokens}")
        
        # Generation loop
        for step in range(max_tokens):
            # Get model predictions
            logits, _ = self.model(generated_ids)
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
            
            # Check for stop conditions
            if stop_tokens:
                generated_text = self.dataset.decode(generated_ids[0])
                for stop_token in stop_tokens:
                    if stop_token in generated_text:
                        print(f"🛑 Stopped at token: '{stop_token}'")
                        return generated_text.split(stop_token)[0] + stop_token
            
            # Print progress
            if (step + 1) % 50 == 0:
                partial_text = self.dataset.decode(generated_ids[0])
                print(f"📊 Step {step + 1}/{max_tokens}: ...{partial_text[-50:]}")
        
        # Decode final text
        generated_text = self.dataset.decode(generated_ids[0])
        return generated_text
    
    def generate_with_style(self, style: str = 'shakespeare', **kwargs) -> str:
        """Generate text in a specific style."""
        if style in self.style_prompts:
            prompt = self.style_prompts[style]
        else:
            prompt = style  # Treat as custom prompt
        
        return self.generate_text(prompt=prompt, **kwargs)
    
    def interactive_generation(self):
        """Interactive text generation session."""
        print(f"\n🎭 Interactive Text Generation")
        print(f"Commands:")
        print(f"  - 'shakespeare' or 's': Generate in Shakespeare style")
        print(f"  - 'alice' or 'a': Generate in Alice in Wonderland style")
        print(f"  - 'custom <prompt>': Use custom prompt")
        print(f"  - 'settings': Show current settings")
        print(f"  - 'quit' or 'q': Exit")
        print(f"")
        
        # Default settings
        settings = {
            'max_tokens': 200,
            'temperature': 0.8,
            'top_k': 40,
            'top_p': 0.95,
            'repetition_penalty': 1.1
        }
        
        while True:
            try:
                user_input = input("🎯 Enter command: ").strip().lower()
                
                if user_input in ['quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input in ['shakespeare', 's']:
                    print("\n🎭 Generating Shakespeare-style text...")
                    text = self.generate_with_style('shakespeare', **settings)
                    print(f"\n📜 Generated text:\n{text}\n")
                
                elif user_input in ['alice', 'a']:
                    print("\n🐰 Generating Alice in Wonderland-style text...")
                    text = self.generate_with_style('alice', **settings)
                    print(f"\n📜 Generated text:\n{text}\n")
                
                elif user_input.startswith('custom '):
                    prompt = user_input[7:]  # Remove 'custom '
                    print(f"\n✨ Generating with custom prompt...")
                    text = self.generate_text(prompt=prompt, **settings)
                    print(f"\n📜 Generated text:\n{text}\n")
                
                elif user_input == 'settings':
                    print(f"\n⚙️ Current settings:")
                    for key, value in settings.items():
                        print(f"   {key}: {value}")
                    print()
                
                elif user_input.startswith('temp '):
                    try:
                        settings['temperature'] = float(user_input[5:])
                        print(f"🌡️ Temperature set to {settings['temperature']}")
                    except ValueError:
                        print("❌ Invalid temperature value")
                
                elif user_input.startswith('topk '):
                    try:
                        settings['top_k'] = int(user_input[5:])
                        print(f"🔝 Top-k set to {settings['top_k']}")
                    except ValueError:
                        print("❌ Invalid top-k value")
                
                elif user_input.startswith('topp '):
                    try:
                        settings['top_p'] = float(user_input[5:])
                        print(f"🎯 Top-p set to {settings['top_p']}")
                    except ValueError:
                        print("❌ Invalid top-p value")
                
                elif user_input.startswith('tokens '):
                    try:
                        settings['max_tokens'] = int(user_input[7:])
                        print(f"📏 Max tokens set to {settings['max_tokens']}")
                    except ValueError:
                        print("❌ Invalid max tokens value")
                
                else:
                    print("❌ Unknown command. Try 'shakespeare', 'alice', 'custom <prompt>', or 'quit'")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def load_model_and_dataset(checkpoint_path: str, data_dir: str = "data/processed") -> tuple:
    """Load trained model and dataset."""
    print(f"📂 Loading model from {checkpoint_path}")
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model from config
    model_config = GPTConfig(**checkpoint['model_config'])
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load dataset for encoding/decoding
    dataset = ShakespeareDataset(
        data_dir=data_dir,
        split="train",
        context_length=model_config.block_size
    )
    
    print(f"✅ Model and dataset loaded successfully")
    return model, dataset, device


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description="RoboPoet Text Generation")
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', '-p', type=str, default='',
                       help='Text prompt for generation')
    parser.add_argument('--style', '-s', type=str, default='shakespeare',
                       choices=['shakespeare', 'alice', 'neutral'],
                       help='Generation style')
    parser.add_argument('--max_tokens', '-t', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k filtering')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Nucleus sampling threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.1,
                       help='Repetition penalty')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Start interactive generation session')
    parser.add_argument('--data_dir', type=str, default='../../data/processed',
                       help='Path to processed data directory')
    
    args = parser.parse_args()
    
    try:
        # Load model and dataset
        model, dataset, device = load_model_and_dataset(args.checkpoint, args.data_dir)
        
        # Create generator
        generator = TextGenerator(model, dataset, device)
        
        if args.interactive:
            # Interactive mode
            generator.interactive_generation()
        else:
            # Single generation
            if args.prompt:
                text = generator.generate_text(
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
            else:
                text = generator.generate_with_style(
                    style=args.style,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
            
            print(f"\n📜 Generated text:")
            print(f"{'='*60}")
            print(text)
            print(f"{'='*60}")
    
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()