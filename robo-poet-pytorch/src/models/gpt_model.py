"""
MinGPT-style transformer model for RoboPoet text generation.
Created by Bernard Orozco - TensorFlow to PyTorch Migration

Based on MinGPT/NanoGPT architecture patterns optimized for small literary datasets.
Target: <10M parameters, validation loss <5.0, coherent 200+ token generation.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT model configuration."""
    # Model architecture
    n_layer: int = 6        # Number of transformer layers
    n_head: int = 8         # Number of attention heads  
    n_embd: int = 256       # Embedding dimensions
    vocab_size: int = 6725  # Vocabulary size (Shakespeare + Alice)
    block_size: int = 128   # Maximum context length
    
    # Regularization
    dropout: float = 0.1    # Dropout probability
    bias: bool = True       # Use bias in linear layers
    
    # Training
    weight_decay: float = 0.01
    learning_rate: float = 6e-4
    betas: Tuple[float, float] = (0.9, 0.95)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    
    Implements scaled dot-product attention with causal masking for autoregressive generation.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Key, query, value projections for all heads (batched)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection  
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask (lower triangular matrix)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            Output tensor [batch_size, seq_len, n_embd]
        """
        B, T, C = x.size()  # batch_size, seq_len, n_embd
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Causal self-attention: [B, n_head, T, head_dim] @ [B, n_head, head_dim, T] -> [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask (prevent attending to future tokens)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values: [B, n_head, T, T] @ [B, n_head, T, head_dim] -> [B, n_head, T, head_dim]
        y = att @ v
        
        # Reassemble all head outputs: [B, n_head, T, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            Output tensor [batch_size, seq_len, n_embd]
        """
        x = self.c_fc(x)      # [B, T, n_embd] -> [B, T, 4*n_embd]
        x = self.gelu(x)      # GELU activation
        x = self.c_proj(x)    # [B, T, 4*n_embd] -> [B, T, n_embd]
        x = self.dropout(x)   # Dropout
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm configuration.
    
    Architecture:
    - LayerNorm -> Self-Attention -> Residual Connection
    - LayerNorm -> MLP -> Residual Connection
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            Output tensor [batch_size, seq_len, n_embd]
        """
        # Pre-norm self-attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # Pre-norm MLP with residual connection  
        x = x + self.mlp(self.ln_2(x))
        
        return x


class GPT(nn.Module):
    """
    GPT model for autoregressive text generation.
    
    Architecture follows MinGPT/NanoGPT patterns optimized for small literary datasets.
    Target: <10M parameters, validation loss <5.0, coherent 200+ token generation.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Positional embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),  # Final layer norm
        ))
        
        # Language modeling head (tied with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: share weights between embedding and output layer
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections  
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"🤖 GPT model initialized:")
        print(f"   📊 Parameters: {self.get_num_params():,}")
        print(f"   📏 Context length: {config.block_size}")
        print(f"   🔤 Vocabulary: {config.vocab_size}")
        print(f"   🏗️  Layers: {config.n_layer}")
        print(f"   🔥 Attention heads: {config.n_head}")
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 initialization scheme."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: Exclude position embeddings (these don't count toward model capacity)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through GPT model.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target token indices for loss calculation [batch_size, seq_len]
            
        Returns:
            Tuple of (logits, loss)
            - logits: [batch_size, seq_len, vocab_size]
            - loss: Cross-entropy loss (if targets provided)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        
        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # [t]
        
        tok_emb = self.transformer.wte(idx)    # [b, t, n_embd]  
        pos_emb = self.transformer.wpe(pos)    # [t, n_embd]
        x = self.transformer.drop(tok_emb + pos_emb)  # [b, t, n_embd]
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer norm and language modeling head
        x = self.transformer.ln_f(x)  # [b, t, n_embd]
        logits = self.lm_head(x)      # [b, t, vocab_size]
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: [b*t, vocab_size] and [b*t]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, 
                 top_k: Optional[int] = None, top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            idx: Conditioning sequence [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k filtering (keep only k most likely tokens)
            top_p: Nucleus sampling (keep tokens with cumulative probability <= p)
            
        Returns:
            Generated sequence [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed (keep last block_size tokens)
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Focus on last time step and apply temperature
            logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # [batch_size, seq_len + 1]
        
        return idx
    
    def configure_optimizers(self):
        """
        Configure optimizer following GPT training best practices.
        
        Returns:
            Configured AdamW optimizer
        """
        # Separate parameters based on their names and properties
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # Rules for weight decay:
            # - Apply weight decay to all weight parameters in Linear layers
            # - Don't apply to biases, LayerNorm, or Embedding weights
            if (name.endswith('.weight') and 
                ('mlp.c_fc' in name or 'mlp.c_proj' in name or 
                 'attn.c_attn' in name or 'attn.c_proj' in name or
                 'wte' in name)):  # Include token embedding for weight decay
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        
        # Create optimizer groups
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        print(f"🔧 Optimizer configured:")
        print(f"   📉 Weight decay params: {len(decay_params)}")
        print(f"   🚫 No decay params: {len(no_decay_params)}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=self.config.betas)
        return optimizer


def create_model(vocab_size: int = 6725, force_gpu: bool = True, **kwargs) -> GPT:
    """
    Create GPT model with sensible defaults for RoboPoet.
    
    Args:
        vocab_size: Vocabulary size from dataset
        force_gpu: Require GPU for academic performance standards
        **kwargs: Override default config parameters
        
    Returns:
        Initialized GPT model
    """
    # Academic Performance Requirement: GPU Validation
    if force_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "🎓 ACADEMIC PERFORMANCE REQUIREMENT: GPU/CUDA not available!\n"
            "   📚 Transformer training requires GPU for:\n"
            "   • >10x faster training than CPU\n"
            "   • Mixed precision (FP16) optimization\n"
            "   • Academic benchmarking compliance\n"
            "   • Memory-efficient large batch processing\n"
            "   🔧 Install CUDA PyTorch or set force_gpu=False"
        )
    
    # Create config with overrides
    config_dict = {
        'vocab_size': vocab_size,
        'n_layer': 6,
        'n_head': 8, 
        'n_embd': 256,
        'block_size': 128,
        'dropout': 0.1
    }
    config_dict.update(kwargs)
    
    config = GPTConfig(**config_dict)
    model = GPT(config)
    
    # Move to GPU if available and required
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        print(f"🔥 Model moved to GPU: {torch.cuda.get_device_name(0)}")
    elif force_gpu:
        raise RuntimeError("GPU required but not available")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing GPT model...")
    
    # Create model
    model = create_model(vocab_size=6725)
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    x = torch.randint(0, 6725, (batch_size, seq_len))
    targets = torch.randint(0, 6725, (batch_size, seq_len))
    
    print(f"\n🔍 Testing forward pass:")
    print(f"   Input shape: {x.shape}")
    print(f"   Target shape: {targets.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits, loss = model(x, targets)
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Loss: {loss.item():.4f}")
    
    # Test generation
    print(f"\n🎯 Testing generation:")
    prompt = torch.randint(0, 6725, (1, 10))  # Single batch, 10 tokens
    print(f"   Prompt shape: {prompt.shape}")
    
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=40)
        print(f"   Generated shape: {generated.shape}")
        print(f"   Generated tokens: {generated[0].tolist()}")
    
    print(f"\n✅ GPT model test completed!")
    print(f"📊 Total parameters: {model.get_num_params():,}")