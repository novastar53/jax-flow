#!/usr/bin/env python3
"""
JiT (Just Image Transformer) implementation matching the official paper.
Uses AdaLN modulation, SwiGLU, QK norm, and proper training configuration.
"""

import sys
import os
import json
import threading
import queue
import copy
from datetime import datetime
from typing import List

# Force unbuffered output for real-time logging
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace')

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import math

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
from huggingface_hub import login, upload_folder, snapshot_download
import orbax.checkpoint as ocp

# ==========================================
# 1. Configuration (Matching JiT official)
# ==========================================
CONFIG = {
    "img_size": 64,
    "patch_size": 4,
    "dim_raw": 4 * 4 * 3,
    "channels": 3,
    "dim_bottleneck": 128,
    "dim_model": 256,
    "depth": 6,
    "heads": 8,
    "mlp_ratio": 4.0,
    "batch_size": 64,
    "lr": 1e-4,
    "min_lr": 1e-6,
    "epochs": 50,
    "warmup_epochs": 5,
    "lr_schedule": "cosine",
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "seed": 42,
    "P_mean": -0.8,  # Logit-normal params
    "P_std": 0.8,
    "t_eps": 1e-3,
    "noise_scale": 1.0,
    "ema_decay1": 0.9999,
    "ema_decay2": 0.999943,
}

# ==========================================
# 2. RMSNorm and Utilities
# ==========================================

class RMSNorm(nnx.Module):
    """RMSNorm as used in JiT and Llama models."""
    def __init__(self, hidden_size: int, eps: float = 1e-6, rngs: nnx.Rngs = None):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((hidden_size,)))

    def __call__(self, x):
        input_dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        # Use rsqrt like PyTorch reference
        x = x * jax.lax.rsqrt(variance + self.eps)
        x = x.astype(input_dtype)
        return x * self.weight.value


def rotate_half(x):
    """Rotate half the hidden dims of the input (Llama-style RoPE).
    x shape: [..., D] where D is even
    Returns: [-x_{D/2:}, x_{:D/2}]
    """
    n = x.shape[-1] // 2
    return jnp.concatenate((-x[..., n:], x[..., :n]), axis=-1)


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings.
    x: [B, H, N, D]
    cos, sin: [N, D]
    """
    # Reshape cos/sin for broadcasting: [1, 1, N, D]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    scale = jnp.expand_dims(scale, axis=1)
    shift = jnp.expand_dims(shift, axis=1)
    return x * (1.0 + scale) + shift


# ==========================================
# 3. Patch Embedding
# ==========================================

class LinearBottleneckPatchEmbed(nnx.Module):
    """Bottleneck patch embedding like JiT."""
    def __init__(self, patch_size: int, dim_bottleneck: int, dim_model: int, rngs: nnx.Rngs):
        self.patch_size = patch_size

        # Two-stage bottleneck
        self.compress = nnx.Linear(
            patch_size * patch_size * 3,
            dim_bottleneck,
            use_bias=False,
            rngs=rngs
        )
        self.expand = nnx.Linear(
            dim_bottleneck,
            dim_model,
            use_bias=True,
            rngs=rngs
        )

    def __call__(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        p = self.patch_size

        # Patchify
        x = x.reshape(B, H // p, p, W // p, p, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, -1, p * p * C)

        # Bottleneck
        x = self.compress(x)
        x = self.expand(x)
        return x


# ==========================================
# 4. Vision Rotary Embeddings and Attention
# ==========================================

class VisionRotaryEmbeddingFast(nnx.Module):
    """Fast 2D rotary embeddings for images (matching JiT reference)."""
    def __init__(self, dim: int, grid_size: int, theta: float = 10000.0):
        """
        dim: head dimension (head_dim)
        grid_size: number of patches along one dimension (img_size // patch_size)
        """
        # Generate frequencies for the dimension (like jaxpt's calc_rope_omega_llama)
        # Each dimension gets half the frequencies for H and half for W
        half_dim = dim // 2
        freqs = 1.0 / (theta ** (jnp.arange(0, half_dim, 2).astype(jnp.float32) / half_dim))
        # Duplicate like jaxpt: concat([freqs, freqs])
        freqs = jnp.concatenate([freqs, freqs], axis=0)  # [half_dim]

        # Position indices for H and W
        t = jnp.arange(grid_size, dtype=jnp.float32)

        # Outer product: [grid_size, half_dim]
        freqs_h = jnp.outer(t, freqs)  # [grid_size, half_dim]
        freqs_w = jnp.outer(t, freqs)

        # Broadcast to 2D grid - need to broadcast to same shape before concatenating
        freqs_h = freqs_h[:, None, :]  # [grid_size, 1, half_dim]
        freqs_w = freqs_w[None, :, :]  # [1, grid_size, half_dim]

        # Broadcast both to [grid_size, grid_size, half_dim] then concatenate
        freqs_h = jnp.broadcast_to(freqs_h, (grid_size, grid_size, half_dim))
        freqs_w = jnp.broadcast_to(freqs_w, (grid_size, grid_size, half_dim))

        # Concatenate h and w frequencies: [grid_size, grid_size, dim]
        freqs = jnp.concatenate([freqs_h, freqs_w], axis=-1)

        # Flatten to [N_patches, dim] where N = grid_size * grid_size
        freqs = freqs.reshape(-1, dim)

        self.freqs_cos = nnx.Variable(jnp.cos(freqs))
        self.freqs_sin = nnx.Variable(jnp.sin(freqs))

    def __call__(self, x):
        """Apply rotary embeddings.
        x: [B, H, N, D] - batch, heads, seq_len, head_dim
        """
        # Expand cos/sin for batch and head dims: [1, 1, N, D]
        cos = self.freqs_cos.value[None, None, :, :]
        sin = self.freqs_sin.value[None, None, :, :]
        return x * cos + rotate_half(x) * sin



def apply_rope(x, omega):
    """Apply rotary position embeddings."""
    # x: [B, H, N, D]
    head_dim = x.shape[-1]
    seq_len = x.shape[2]

    # Create position indices
    pos = jnp.arange(seq_len)

    # Get freqs
    freqs = omega.value  # [N, D//2, 2]

    # Reshape x for rotation
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # [B, H, N, D//2, 2]

    # Apply rotation
    x_out = jnp.stack([
        x_reshaped[..., 0] * freqs[:seq_len, :, 0] - x_reshaped[..., 1] * freqs[:seq_len, :, 1],
        x_reshaped[..., 0] * freqs[:seq_len, :, 1] + x_reshaped[..., 1] * freqs[:seq_len, :, 0]
    ], axis=-1)

    return x_out.reshape(x.shape)


class Attention(nnx.Module):
    """Multi-head attention with QK norm and RoPE (matching JiT reference)."""
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

        # QK Norm (applied per head)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, rngs=rngs)

    def __call__(self, x, rope=None):
        """
        x: [B, N, C]
        rope: VisionRotaryEmbeddingFast instance
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        # [B, N, 3*C] -> [B, N, 3, H, D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each [B, N, H, D]

        # Transpose to [B, H, N, D] for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # QK norm per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        if rope is not None:
            q = rope(q)
            k = rope(k)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)

        # Reshape back: [B, H, N, D] -> [B, N, C]
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


# ==========================================
# 5. SwiGLU FFN
# ==========================================

class SwiGLU(nnx.Module):
    """SwiGLU FFN as used in JiT."""
    def __init__(self, dim: int, hidden_dim: int, rngs: nnx.Rngs):
        # JiT uses hidden_dim = int(mlp_ratio * dim * 2/3)
        self.hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nnx.Linear(dim, 2 * self.hidden_dim, use_bias=True, rngs=rngs)
        self.w3 = nnx.Linear(self.hidden_dim, dim, use_bias=True, rngs=rngs)

    def __call__(self, x):
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nnx.silu(x1) * x2
        return self.w3(hidden)


# ==========================================
# 6. JiT Block with AdaLN
# ==========================================

class JiTBlock(nnx.Module):
    """JiT transformer block with AdaLN modulation."""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, rngs: nnx.Rngs = None):
        self.norm1 = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)
        self.attn = Attention(hidden_size, num_heads, rngs=rngs)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(hidden_size, mlp_hidden_dim, rngs=rngs)

        # AdaLN modulation: outputs 6 values
        self.adaLN_modulation = nnx.Linear(hidden_size, 6 * hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x, c, rope=None):
        # Get AdaLN params
        params = self.adaLN_modulation(c)  # [B, 6*C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(params, 6, axis=-1)

        # Attention block
        x = x + jnp.expand_dims(gate_msa, 1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope
        )

        # MLP block
        x = x + jnp.expand_dims(gate_mlp, 1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


# ==========================================
# 7. Final Layer with AdaLN
# ==========================================

class FinalLayer(nnx.Module):
    """Final layer with AdaLN modulation."""
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, rngs: nnx.Rngs):
        self.norm_final = RMSNorm(hidden_size, eps=1e-6, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size, patch_size * patch_size * out_channels, use_bias=True, rngs=rngs
        )
        self.adaLN_modulation = nnx.Linear(hidden_size, 2 * hidden_size, use_bias=True, rngs=rngs)

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ==========================================
# 8. JiT Denoising Transformer
# ==========================================

class TimestepEmbedder(nnx.Module):
    """Timestep embedder with sinusoidal base + MLP (matching JiT reference)."""
    def __init__(self, hidden_size: int, freq_embed_size: int = 256, rngs: nnx.Rngs = None):
        self.freq_embed_size = freq_embed_size
        self.mlp = nnx.Sequential(
            nnx.Linear(freq_embed_size, hidden_size, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        )

    def timestep_embedding(self, t: jnp.ndarray, dim: int, max_period: float = 10000.0) -> jnp.ndarray:
        """
        Create sinusoidal timestep embeddings.
        t: [B, 1] - timesteps
        """
        half = dim // 2
        # Create frequency bands
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
        # Outer product: [B, 1] * [half] -> [B, half]
        args = t.astype(jnp.float32) * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2 == 1:
            # Pad if odd dimension
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t):
        """
        t: [B, 1]
        Returns: [B, hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.freq_embed_size)
        return self.mlp(t_freq)


class JiTDenoisingTransformer(nnx.Module):
    """JiT-style denoising transformer."""
    def __init__(self, config: dict, rngs: nnx.Rngs):
        self.config = config
        self.patch_embed = LinearBottleneckPatchEmbed(
            config['patch_size'],
            config['dim_bottleneck'],
            config['dim_model'],
            rngs
        )

        # Time embedding with sinusoidal base (matching JiT reference)
        self.t_embedder = TimestepEmbedder(config['dim_model'], freq_embed_size=256, rngs=rngs)

        # Fixed position embeddings (will be initialized)
        num_patches = (config['img_size'] // config['patch_size']) ** 2
        self.pos_embed = nnx.Param(jnp.zeros((1, num_patches, config['dim_model'])))

        # RoPE for image patches (matching JiT reference)
        grid_size = config['img_size'] // config['patch_size']
        head_dim = config['dim_model'] // config['heads']
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=head_dim,
            grid_size=grid_size,
            theta=10000.0
        )

        # JiT blocks
        self.blocks = nnx.List([
            JiTBlock(
                hidden_size=config['dim_model'],
                num_heads=config['heads'],
                mlp_ratio=config['mlp_ratio'],
                rngs=rngs
            )
            for _ in range(config['depth'])
        ])

        # Final layer
        self.final_layer = FinalLayer(
            config['dim_model'],
            config['patch_size'],
            config['channels'],
            rngs
        )

        # Initialize pos embed and weights
        self._init_pos_embed()
        self._init_weights()

    def _init_pos_embed(self):
        """Initialize 2D sin-cos position embeddings."""
        grid_size = int(self.config['img_size'] // self.config['patch_size'])
        embed_dim = self.config['dim_model']

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)

        # Get 2D sin-cos embeddings
        emb_h = self._get_1d_sincos(embed_dim // 2, grid[0].reshape(-1))
        emb_w = self._get_1d_sincos(embed_dim // 2, grid[1].reshape(-1))
        pos_embed = np.concatenate([emb_h, emb_w], axis=1)

        self.pos_embed.value = jnp.array(pos_embed[None, ...])

    def _init_weights(self):
        """Initialize weights matching JiT reference.
        Critical: Zero out adaLN modulation and final layer for training stability.
        """
        # Note: Linear layers in flax.nnx use default init which is similar to PyTorch
        # We need to zero out the adaLN modulation weights and biases
        for block in self.blocks:
            # Zero out the last layer of adaLN_modulation (the linear layer)
            ada_ln = block.adaLN_modulation
            # Get the last linear layer in the Sequential
            if hasattr(ada_ln, 'layers') and len(ada_ln.layers) >= 2:
                # It's a Sequential with [SiLU, Linear] - access the Linear layer
                linear_layer = ada_ln.layers[-1]
                if isinstance(linear_layer, nnx.Linear):
                    # Zero out weights and biases
                    linear_layer.kernel.value = jnp.zeros_like(linear_layer.kernel.value)
                    if linear_layer.bias is not None:
                        linear_layer.bias.value = jnp.zeros_like(linear_layer.bias.value)

        # Zero out final layer adaLN and linear
        final_ada_ln = self.final_layer.adaLN_modulation
        if hasattr(final_ada_ln, 'layers') and len(final_ada_ln.layers) >= 2:
            linear_layer = final_ada_ln.layers[-1]
            if isinstance(linear_layer, nnx.Linear):
                linear_layer.kernel.value = jnp.zeros_like(linear_layer.kernel.value)
                if linear_layer.bias is not None:
                    linear_layer.bias.value = jnp.zeros_like(linear_layer.bias.value)

        # Zero out final linear layer
        final_linear = self.final_layer.linear
        final_linear.kernel.value = jnp.zeros_like(final_linear.kernel.value)
        if final_linear.bias is not None:
            final_linear.bias.value = jnp.zeros_like(final_linear.bias.value)

    def _get_1d_sincos(self, embed_dim, pos):
        """Generate 1D sin-cos position embeddings."""
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega

        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        return np.concatenate([emb_sin, emb_cos], axis=1)

    def __call__(self, x_noisy, t):
        # x_noisy: [B, H, W, C]
        # t: [B, 1]

        # Patch embed
        x = self.patch_embed(x_noisy)  # [B, N, D]

        # Add positional embeddings
        x = x + self.pos_embed.value

        # Get time embedding
        c = self.t_embedder(t)  # [B, D]

        # Pass through blocks with RoPE
        for block in self.blocks:
            x = block(x, c, rope=self.feat_rope)

        # Final layer
        x = self.final_layer(x, c)

        # Unpatchify
        return self._unpatchify(x)

    def _unpatchify(self, x_patches):
        B, N, D = x_patches.shape
        p = self.config['patch_size']
        h = w = int(math.sqrt(N))
        C = self.config['channels']

        x = x_patches.reshape(B, h, w, p, p, C)
        x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))  # [B, C, h, p, w, p]
        x = x.reshape(B, C, h * p, w * p)
        x = jnp.transpose(x, (0, 2, 3, 1))  # [B, H, W, C]
        return x


# ==========================================
# 9. Data Loading
# ==========================================

def celeba_generator_hf(split="train", batch_size=32, img_size=64):
    """HuggingFace-based CelebA dataloader."""
    print(f"Loading CelebA dataset from HuggingFace (split={split})...", flush=True)
    ds = load_dataset("flwrlabs/celeba", split=split, streaming=True)
    print(f"Dataset loaded successfully! Starting data stream...", flush=True)

    ds_iterator = iter(ds)

    while True:
        batch_imgs = []
        for _ in range(batch_size):
            try:
                sample = next(ds_iterator)
                img = sample["image"]
                img = img.resize((img_size, img_size))
                img_array = np.array(img).astype(np.float32) / 255.0

                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]

                batch_imgs.append(img_array)
            except StopIteration:
                ds_iterator = iter(ds)
                break

        if len(batch_imgs) == batch_size:
            batch = np.stack(batch_imgs, axis=0)
            batch = (batch * 2.0) - 1.0  # Normalize to [-1, 1]
            yield batch


class PrefetchGenerator:
    """Prefetch batches in a background thread to overlap data loading with GPU compute."""
    def __init__(self, generator, buffer_size=4):
        self.generator = generator
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def _prefetch_loop(self):
        """Background thread that fills the prefetch queue."""
        try:
            for item in self.generator:
                self.queue.put(item)
        except StopIteration:
            pass
        # Signal end with None
        for _ in range(self.buffer_size):
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item


# ==========================================
# 10. Training
# ==========================================

def create_model_and_optimizer(rng_key, config, mesh, total_steps: int = 10000):
    """Create model and optimizer with proper sharding and weight decay mask."""
    with mesh:
        rngs = nnx.Rngs(rng_key)
        model = JiTDenoisingTransformer(config, rngs)

        # Initialize with dummy data
        dummy_x = jnp.ones((1, config['img_size'], config['img_size'], config['channels']))
        dummy_t = jnp.ones((1, 1))
        _ = model(dummy_x, dummy_t)

        # Learning rate schedule with warmup and cosine decay
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config['lr'],
            warmup_steps=config['warmup_epochs'] * (total_steps // config['epochs']),
            decay_steps=total_steps,
            end_value=config['min_lr'],
        )

        # Use AdamW with weight decay applied to all parameters
        # (weight decay masking would require additional complexity with nnx)
        tx = optax.adamw(
            learning_rate=schedule,
            weight_decay=config['weight_decay'],
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    return model, optimizer


class EMATracker:
    """EMA tracker for model parameters (matching JiT reference)."""
    def __init__(self, model, decay1: float = 0.9999, decay2: float = 0.999943):
        self.decay1 = decay1
        self.decay2 = decay2

        # Get all parameters from the model
        graphdef, params, _ = nnx.split(model, nnx.Param, nnx.Variable)

        # Initialize EMA parameters as copies (params are already JAX arrays from nnx.split)
        self.ema_params1 = jax.tree.map(lambda x: x.copy(), params)
        self.ema_params2 = jax.tree.map(lambda x: x.copy(), params)

    def update(self, model):
        """Update EMA parameters."""
        _, params, _ = nnx.split(model, nnx.Param, nnx.Variable)

        # Update EMA1 (params are already JAX arrays)
        self.ema_params1 = jax.tree.map(
            lambda ema, p: self.decay1 * ema + (1 - self.decay1) * p,
            self.ema_params1,
            params
        )

        # Update EMA2
        self.ema_params2 = jax.tree.map(
            lambda ema, p: self.decay2 * ema + (1 - self.decay2) * p,
            self.ema_params2,
            params
        )

    def apply_to_model(self, model):
        """Apply EMA parameters to a model (for inference)."""
        graphdef, _, rest = nnx.split(model, nnx.Param, nnx.Variable)
        # Merge with EMA params
        ema_model = nnx.merge(graphdef, self.ema_params1, rest)
        return ema_model


def get_lr(step, total_steps, config):
    """Get learning rate with warmup and cosine decay."""
    epoch = step / (total_steps / config['epochs'])

    if epoch < config['warmup_epochs']:
        # Linear warmup
        return config['lr'] * epoch / config['warmup_epochs']
    else:
        # Cosine decay
        if config['lr_schedule'] == 'cosine':
            progress = (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])
            return config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return config['lr']


def train_step_fn(model, batch, rng, P_mean, P_std, noise_scale, t_eps):
    """Training step logic (not JIT'd - JIT applied in trainer loop)."""
    x_clean = batch  # [B, H, W, C]
    B = x_clean.shape[0]

    rng, t_rng, n_rng = jax.random.split(rng, 3)

    # Sample t from logit-normal
    z = jax.random.normal(t_rng, (B, 1)) * P_std + P_mean
    t = jax.nn.sigmoid(z)

    # Noise
    noise = jax.random.normal(n_rng, x_clean.shape) * noise_scale

    # Interpolate: z = t * x + (1 - t) * e
    t_img = t.reshape(B, 1, 1, 1)
    x_t = t_img * x_clean + (1 - t_img) * noise

    # Target velocity: v = (x - z) / (1 - t)
    denom = jnp.clip(1 - t_img, t_eps, 1.0)
    v_target = (x_clean - x_t) / denom

    def loss_fn(m):
        x_pred = m(x_t, t)
        v_pred = (x_pred - x_t) / denom
        loss = jnp.mean((v_pred - v_target) ** 2)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    return loss, grads, rng


# JIT the function outside like in jaxpt trainers
# Will be used in the training loop


def sample(model, rng, img_size, steps=50, noise_scale=1.0, t_eps=1e-3, batch_size=1):
    """Generate samples using Euler method (matching JiT reference)."""
    print(f"Sampling with {steps} steps...", flush=True)

    x_current = jax.random.normal(rng, (batch_size, img_size, img_size, 3)) * noise_scale
    # Use linspace from 0 to 0.99 with steps+1 points (don't go all the way to 1.0 to avoid instability)
    # This gives us 'steps' intervals to iterate through
    t_values = np.linspace(0.0, 0.99, steps + 1)
    dt = t_values[1] - t_values[0]

    for t_scalar in t_values[:-1]:
        t_vec = jnp.ones((batch_size, 1)) * t_scalar
        # Model predicts the clean image x_1 from the current noised state
        x_clean_pred = model(x_current, t_vec)
        # Compute velocity: v = (x_1 - x_t) / (1 - t)
        denom = 1.0 - t_scalar
        v = (x_clean_pred - x_current) / denom
        # Euler step
        x_current = x_current + v * dt

    return x_current


# ==========================================
# 11. Main
# ==========================================

def main():
    print("=" * 60, flush=True)
    print("JiT Denoising Training (CelebA 64x64)", flush=True)
    print("=" * 60, flush=True)

    # Setup devices and mesh
    devices = jax.devices()
    num_devices = len(devices)
    print(f"\n[0/5] Device setup: {num_devices} device(s)", flush=True)
    print(f"      Platform: {jax.default_backend()}", flush=True)
    mesh = Mesh(devices, ["devices"])

    # Setup data with prefetching
    print("\n[1/5] Setting up data loader with prefetching...", flush=True)
    train_gen = celeba_generator_hf(
        split="train",
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size']
    )
    # Wrap with prefetch generator to overlap data loading with GPU compute
    train_gen = PrefetchGenerator(train_gen, buffer_size=4)
    print(f"    Prefetch buffer: 4 batches", flush=True)

    data_sharding = NamedSharding(mesh, PartitionSpec("devices"))

    # Training config (define before model creation)
    steps_per_epoch = 162000 // CONFIG['batch_size']
    num_epochs = CONFIG['epochs']
    sample_every = 2000
    total_steps = steps_per_epoch * num_epochs

    # Setup model
    print("[2/5] Initializing model...", flush=True)
    rng = jax.random.PRNGKey(CONFIG['seed'])
    rng, init_rng = jax.random.split(rng)

    model, optimizer = create_model_and_optimizer(init_rng, CONFIG, mesh, total_steps)
    print(f"    Model: {CONFIG['dim_model']}d, {CONFIG['depth']} layers", flush=True)
    print(f"    Batch size: {CONFIG['batch_size']}, LR: {CONFIG['lr']}", flush=True)

    # Setup EMA
    ema_tracker = EMATracker(model, CONFIG['ema_decay1'], CONFIG['ema_decay2'])
    print(f"    EMA decays: {CONFIG['ema_decay1']}, {CONFIG['ema_decay2']}", flush=True)

    # Setup output
    output_dir = "generated_samples_jit"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[3/5] Output directory: {output_dir}/", flush=True)

    print(f"[4/5] Training configuration:", flush=True)
    print(f"      Steps per epoch: {steps_per_epoch}", flush=True)
    print(f"      Total epochs: {num_epochs}", flush=True)
    print(f"      Total steps: {total_steps}", flush=True)
    print(f"      Sample every: {sample_every} steps", flush=True)
    print(f"\n{'='*60}", flush=True)
    print("[5/5] Starting training...", flush=True)
    print('='*60, flush=True)

    with mesh:
        global_step = 0

        # JIT the train step inside mesh context like jaxpt does
        train_step_jit = nnx.jit(train_step_fn)

        # Debug first batch
        first_batch = next(train_gen)
        print(f"\nDebug - First batch: shape={first_batch.shape}, "
              f"range=[{first_batch.min():.3f}, {first_batch.max():.3f}]", flush=True)

        # Generate a sample before training to verify sampling works
        print("\nGenerating pre-training sample...", flush=True)
        pretrain_sample = sample(model, rng, img_size=CONFIG['img_size'],
                                 noise_scale=CONFIG['noise_scale'], t_eps=CONFIG['t_eps'])
        img = np.array(pretrain_sample[0])
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        filepath = os.path.join(output_dir, "sample_pretraining.png")
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Pre-training Sample (Random Init)")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Pre-training sample saved: {filepath}", flush=True)

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===", flush=True)
            epoch_losses = []

            for step_in_epoch in range(steps_per_epoch):
                batch = next(train_gen)
                batch = jax.device_put(batch, data_sharding)

                loss, grads, rng = train_step_jit(
                    model, batch, rng,
                    CONFIG['P_mean'], CONFIG['P_std'], CONFIG['noise_scale'], CONFIG['t_eps']
                )
                optimizer.update(model, grads)
                epoch_losses.append(float(loss))

                # Update EMA after each training step
                ema_tracker.update(model)

                if step_in_epoch % 100 == 0:
                    avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:]) if epoch_losses else float(loss)
                    print(f"Epoch {epoch + 1} | Step {step_in_epoch}/{steps_per_epoch} | "
                          f"Loss: {loss:.5f} | Avg: {avg_loss:.5f}", flush=True)

                if global_step % sample_every == 0 and global_step > 0:
                    print(f"Sampling at step {global_step}...", flush=True)
                    # Use EMA model for sampling
                    ema_model = ema_tracker.apply_to_model(model)
                    sample_out = sample(ema_model, rng, img_size=CONFIG['img_size'],
                                        noise_scale=CONFIG['noise_scale'], t_eps=CONFIG['t_eps'])

                    img = np.array(sample_out[0])
                    img = (img + 1.0) / 2.0
                    img = np.clip(img, 0, 1)

                    filename = f"sample_step{global_step:06d}_e{epoch+1}.png"
                    filepath = os.path.join(output_dir, filename)

                    plt.figure(figsize=(4, 4))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Step {global_step}")
                    plt.tight_layout()
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"Saved: {filepath}", flush=True)

                global_step += 1

    print("\n" + "="*60, flush=True)
    print("Training complete!", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
