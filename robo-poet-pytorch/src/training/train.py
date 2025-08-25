"""
Training script for RoboPoet PyTorch GPT model.
Created by Bernard Orozco - TensorFlow to PyTorch Migration

Implements training loop with mixed precision, gradient clipping, and checkpointing.
Target: validation loss <5.0, >1000 tokens/sec on RTX 2000 Ada.
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from models.gpt_model import GPT, GPTConfig, create_model
from data.shakespeare_dataset import ShakespeareDataset, create_dataloaders


class GPTTrainer:
    """
    Trainer class for GPT model with advanced features.
    
    Features:
    - Mixed precision training (autocast + GradScaler)
    - Gradient accumulation for larger effective batch sizes
    - Learning rate scheduling with warmup and cosine decay
    - Gradient clipping
    - Automatic checkpointing
    - TensorBoard logging
    - Early stopping
    """
    
    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = model.configure_optimizers()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get('mixed_precision', True))
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config['epochs']
        warmup_steps = int(0.1 * total_steps)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps - warmup_steps,
            eta_min=config.get('min_lr', 1e-6)
        )
        self.warmup_steps = warmup_steps
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        print(f"🏋️ GPT Trainer initialized:")
        print(f"   🎯 Target val loss: <5.0")
        print(f"   📊 Train batches: {len(train_loader)}")
        print(f"   📊 Val batches: {len(val_loader)}")
        print(f"   🔥 Mixed precision: {config.get('mixed_precision', True)}")
        print(f"   📈 Learning rate: {config.get('learning_rate', 6e-4)}")
        print(f"   🎲 Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.config.get('mixed_precision', True)):
                logits, loss = self.model(inputs, targets)
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Learning rate scheduling
                if self.global_step < self.warmup_steps:
                    # Linear warmup
                    lr = self.config.get('learning_rate', 6e-4) * (self.global_step + 1) / self.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Accumulate metrics
            total_loss += loss.item() * gradient_accumulation_steps  # Unscale loss for logging
            total_tokens += inputs.numel()
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"   📈 Batch {batch_idx:4d}/{len(self.train_loader):4d} | "
                      f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Tokens/sec: {tokens_per_sec:.0f}")
                
                # TensorBoard logging
                self.writer.add_scalar('train/loss_step', loss.item() * gradient_accumulation_steps, self.global_step)
                self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)
                self.writer.add_scalar('train/tokens_per_sec', tokens_per_sec, self.global_step)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        elapsed_time = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed_time
        
        return {
            'train_loss': avg_loss,
            'tokens_per_sec': tokens_per_sec,
            'elapsed_time': elapsed_time
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.get('mixed_precision', True)):
                logits, loss = self.model(inputs, targets)
            
            total_loss += loss.item()
            total_tokens += inputs.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens
        }
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch_{self.epoch}.pth'
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_config': self.model.config.__dict__
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Best checkpoint saved: {best_path}")
        
        print(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"📂 Checkpoint loaded: {filepath}")
        print(f"   Resuming from epoch {self.epoch}, step {self.global_step}")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
    
    def train(self, epochs: int, save_every: int = 5, patience: int = 10):
        """
        Main training loop.
        
        Args:
            epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            patience: Early stopping patience (epochs without improvement)
        """
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"   💾 Saving every {save_every} epochs")
        print(f"   ⏰ Early stopping patience: {patience}")
        
        for epoch in range(epochs):
            self.epoch = epoch
            print(f"\n🎯 Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"✅ Training completed:")
            print(f"   📉 Loss: {train_metrics['train_loss']:.4f}")
            print(f"   ⚡ Speed: {train_metrics['tokens_per_sec']:.0f} tokens/sec")
            print(f"   ⏱️  Time: {train_metrics['elapsed_time']:.1f}s")
            
            # Validate
            val_metrics = self.validate()
            print(f"📊 Validation:")
            print(f"   📉 Loss: {val_metrics['val_loss']:.4f}")
            print(f"   📈 Perplexity: {val_metrics['perplexity']:.2f}")
            
            # TensorBoard logging
            self.writer.add_scalar('train/epoch_loss', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('val/perplexity', val_metrics['perplexity'], epoch)
            self.writer.add_scalar('train/epoch_tokens_per_sec', train_metrics['tokens_per_sec'], epoch)
            
            # Check for improvement
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                print(f"🎉 New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"⏳ No improvement for {self.patience_counter}/{patience} epochs")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                break
            
            # Check if target achieved
            if val_metrics['val_loss'] < 5.0:
                print(f"🎯 Target validation loss achieved: {val_metrics['val_loss']:.4f} < 5.0")
                self.save_checkpoint(is_best=True, filename='target_achieved.pth')
        
        print(f"\n🏁 Training completed!")
        print(f"   🏆 Best validation loss: {self.best_val_loss:.4f}")
        print(f"   📊 Total steps: {self.global_step}")
        
        self.writer.close()


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_dir': '/mnt/c/Users/Bernard.Orozco/Documents/Github/Robo-Poet/data/processed',
        'batch_size': 32,
        'context_length': 128,
        'epochs': 50,
        'learning_rate': 6e-4,
        'min_lr': 1e-6,
        'weight_decay': 0.01,
        'mixed_precision': True,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'log_interval': 50,
        'num_workers': 4,
        'resume_from': None  # Path to checkpoint to resume from
    }
    
    # Academic Performance Requirement: GPU Mandatory
    if not torch.cuda.is_available():
        raise RuntimeError(
            "🎓 ACADEMIC PERFORMANCE REQUIREMENT: GPU/CUDA not available!\n"
            "   📚 Academic training requires GPU for:\n"
            "   • >10x faster training performance\n"
            "   • Mixed precision training (FP16)\n"
            "   • Large batch processing\n"
            "   • Academic benchmarking standards\n"
            "   🔧 Please install CUDA-enabled PyTorch"
        )
    
    device = torch.device('cuda')
    print(f"🔥 Using GPU (Academic Performance Mode): {device}")
    print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"   🎓 Academic compliance: GPU mandatory for performance standards")
    
    # Create data loaders
    print(f"\n📚 Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        context_length=config['context_length'],
        num_workers=config['num_workers'],
        device=device
    )
    
    # Get vocabulary size from dataset
    vocab_size = train_loader.dataset.get_vocab_size()
    
    # Create model
    print(f"\n🤖 Creating GPT model...")
    model = create_model(
        vocab_size=vocab_size,
        n_layer=6,
        n_head=8,
        n_embd=256,
        block_size=config['context_length'],
        dropout=0.1
    )
    
    # Create trainer
    trainer = GPTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir='../../checkpoints',
        log_dir='../../logs'
    )
    
    # Resume from checkpoint if specified
    if config['resume_from']:
        trainer.load_checkpoint(config['resume_from'])
    
    # Start training
    trainer.train(
        epochs=config['epochs'],
        save_every=5,
        patience=10
    )


if __name__ == "__main__":
    main()