"""
JiT (Just Image Transformer) implementation matching the official paper.
Uses AdaLN modulation, SwiGLU, QK norm, and proper training configuration.
"""

import sys
import os
import threading
from pprint import pprint
import queue
import time
from datetime import datetime, timedelta
from typing import List, Literal
from dataclasses import dataclass

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Setup devices and mesh
devices = jax.devices()
num_devices = len(devices)
platform = jax.default_backend()
print(f"{num_devices} device(s)")
print(f"Platform: {platform}")
mesh = Mesh(devices, ["devices"])


# Enable mixed precision matmul for better performance
if platform == "cpu":
    jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32")

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

from jaxpt.utils import create_sharded_model

from jax_flow.datasets.celeba_cached import (
    make_dataloader as make_cached_dataloader,
    DataConfig as CachedDataConfig,
    is_cached,
    cache_dataset,
)


@dataclass(eq=True, unsafe_hash=True)
class JiTConfig:
    # Model architecture
    img_size: int = 128
    patch_size: int = 8
    channels: int = 3
    dim_bottleneck: int = 32
    dim_model: int = 256
    depth: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0

    eps = 1e-6
    
    # Training hyperparameters
    batch_size: int = 128
    lr: float = 1e-4
    epochs: int = 50
    warmup_epochs: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    
    # Flow matching parameters
    P_mean: float = -0.8
    P_std: float = 0.8
    t_eps: float = 0.05
    noise_scale: float = 1.0
    
    # EMA parameters
    ema_decay1: float = 0.9999
    ema_decay2: float = 0.999943
    use_ema: bool = False  # Enable EMA for better training stability
    
    # Data loading
    use_cached_data: bool = False  # Enable cached data for faster loading
    
    # Mixed precision settings - automatically adjust based on platform
    dtype: jnp.dtype = jnp.bfloat16  # Will be set based on platform
    param_dtype: jnp.dtype = jnp.float32  # Parameter storage dtype
    
    # Attention implementation
    attn_impl: Literal["xla", "cudnn"] = "cudnn" if platform == "cuda" else "xla"
    output_dir: str = "generated_samples_jit"

config = JiTConfig()
pprint(config)

class RMSNorm(nnx.Module):
    """RMSNorm as used in JiT and Llama models."""
    def __init__(self, hidden_size: int, eps: float,
                 dtype: jnp.dtype,
                 param_dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=param_dtype))

    def __call__(self, x):
        # RMSNorm should preserve the input dtype
        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        # Use rsqrt like PyTorch reference
        x = x * jax.lax.rsqrt(variance + self.eps)
        # Convert weight to match computation dtype
        return x * self.weight.value


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift"""
    scale = jnp.expand_dims(scale, axis=1)
    shift = jnp.expand_dims(shift, axis=1)
    return x * (1.0 + scale) + shift


class ConvBottleneckPatchEmbed(nnx.Module):
    """Conv2d-based bottleneck patch embedding matching JiT reference."""
    def __init__(self, patch_size: int, dim_bottleneck: int, dim_model: int,
                 dtype: jnp.dtype,
                 param_dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.dtype = dtype
        self.param_dtype = param_dtype

        # proj1: Conv2d with stride=patch_size (patchify + expand channels)
        self.proj1 = nnx.Conv(
            in_features=3,
            out_features=dim_bottleneck,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        # proj2: 1x1 Conv (bottleneck projection)
        self.proj2 = nnx.Conv(
            in_features=dim_bottleneck,
            out_features=dim_model,
            kernel_size=(1, 1),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

    def __call__(self, x):
        # x: [B, H, W, C] = [B, 64, 64, 3]
        # Flax NNX Conv expects HWC format

        # Patchify + expand
        x = self.proj1(x)  # [B, 16, 16, 32]
        x = self.proj2(x)  # [B, 16, 16, 192]

        # Flatten spatial to sequence: [B, 16, 16, 192] → [B, 256, 192]
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        return x


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


    @staticmethod
    def rotate_half(x):
        """Rotate half the hidden dims of the input (Llama-style RoPE).
        x shape: [..., D] where D is even
        Returns: [-x_{D/2:}, x_{:D/2}]
        """
        n = x.shape[-1] // 2
        return jnp.concatenate((-x[..., n:], x[..., :n]), axis=-1)


    def __call__(self, x):
        """Apply rotary embeddings.
        x: [B, H, N, D] - batch, heads, seq_len, head_dim
        """
        # Expand cos/sin for batch and head dims: [1, 1, N, D]
        cos = self.freqs_cos.value[None, None, :, :]
        sin = self.freqs_sin.value[None, None, :, :]
        return x * cos + self.rotate_half(x) * sin


class Attention(nnx.Module):
    """Multi-head attention with QK norm and RoPE (matching JiT reference)."""
    def __init__(self, dim: int, num_heads: int,
                 dtype: jnp.dtype,
                 param_dtype: jnp.dtype,
                 attn_impl: Literal["xla", "cudnn"],
                 eps: float,
                 rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.attn_impl = attn_impl
        self.config = config

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=True,
                             dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.proj = nnx.Linear(dim, dim,
                              dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        self.q_norm = RMSNorm(self.head_dim, eps=eps,
                              dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=eps,
                              dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        grid_size = self.config.img_size // self.config.patch_size
        head_dim = self.config.dim_model // self.config.heads
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=head_dim,
            grid_size=grid_size,
            theta=10000.0
        )

    def __call__(self, x):
        """
        x: [B, N, C]
        rope: VisionRotaryEmbeddingFast instance
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        # [B, N, 3*C] -> [B, N, 3, H, D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each [B, N, H, D]

        # Transpose to [B, H, N, D] for QK norm and RoPE
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # QK norm per head
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Transpose back for flash attention: [B, H, N, D] -> [B, N, H, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        out = jax.nn.dot_product_attention(
            q, k, v,
            implementation=self.attn_impl,
            is_causal=False  
        )
        # Final reshape: [B, N, H, D] -> [B, N, C]
        out = out.reshape(B, N, C)
        out = self.proj(out)
        return out


class SwiGLU(nnx.Module):
    def __init__(self, dim: int, hidden_dim: int,
                    dtype: jnp.dtype,
                    param_dtype: jnp.dtype,
                    rngs: nnx.Rngs):
            # JiT uses hidden_dim = int(mlp_ratio * dim * 2/3)
            self.hidden_dim = int(hidden_dim * 2 / 3)
            self.dtype = dtype
            self.param_dtype = param_dtype
            self.w12 = nnx.Linear(dim, 2 * self.hidden_dim, use_bias=True,
                                dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            self.w3 = nnx.Linear(self.hidden_dim, dim, use_bias=True,
                                dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, x):
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nnx.silu(x1) * x2
        return self.w3(hidden)


class JiTBlock(nnx.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float,
                 dtype: jnp.dtype,
                 param_dtype: jnp.dtype,
                 attn_impl: Literal["xla", "cudnn"],
                 eps: float,
                 rngs: nnx.Rngs):
        self.dtype = dtype
        self.param_dtype = param_dtype
        
        self.norm1 = RMSNorm(hidden_size, eps=eps,
                            dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.attn = Attention(hidden_size, num_heads, 
                             dtype=dtype, param_dtype=param_dtype,
                             attn_impl=attn_impl, eps=eps, rngs=rngs)
        self.norm2 = RMSNorm(hidden_size, eps=eps,
                            dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLU(hidden_size, mlp_hidden_dim,
                          dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        # AdaLN modulation: outputs 6 values (Sequential with SiLU + Linear, matching JiT reference)
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(hidden_size, 6 * hidden_size, use_bias=True,
                      dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        )

    def __call__(self, x, c):
        # Get AdaLN params
        params = self.adaLN_modulation(c)  # [B, 6*C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(params, 6, axis=-1)

        # Attention block
        x = x + jnp.expand_dims(gate_msa, 1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
        )

        # MLP block
        x = x + jnp.expand_dims(gate_mlp, 1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nnx.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int,
                 dtype: jnp.dtype,
                 param_dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        self.dtype = dtype
        self.param_dtype = param_dtype
        
        self.norm_final = RMSNorm(hidden_size, eps=1e-6,
                                dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size, patch_size * patch_size * out_channels, use_bias=True,
            dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        # SiLU + Linear (matching JiT reference)
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(hidden_size, 2 * hidden_size, use_bias=True,
                      dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        )

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nnx.Module):
    def __init__(self, hidden_size: int, freq_embed_size: int,
                 dtype: jnp.dtype,
                 param_dtype: jnp.dtype,
                 rngs: nnx.Rngs):
        self.freq_embed_size = freq_embed_size
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.mlp = nnx.Sequential(
            nnx.Linear(freq_embed_size, hidden_size,
                      dtype=dtype, param_dtype=param_dtype, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size,
                      dtype=dtype, param_dtype=param_dtype, rngs=rngs),
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
    def __init__(self, config: JiTConfig, rngs: nnx.Rngs):
        # Convert back to JiTConfig for attribute access
        self.config = config
        param_dtype = self.config.param_dtype
        
        self.patch_embed = ConvBottleneckPatchEmbed(
            self.config.patch_size,
            self.config.dim_bottleneck,
            self.config.dim_model,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            rngs=rngs
        )

        # Time embedding with sinusoidal base (matching JiT reference)
        self.t_embedder = TimestepEmbedder(
            self.config.dim_model, freq_embed_size=256,
            dtype=self.config.dtype, param_dtype=self.config.param_dtype, rngs=rngs
        )

        # Fixed position embeddings (will be initialized)
        num_patches = (self.config.img_size // self.config.patch_size) ** 2
        self.pos_embed = nnx.Param(
            jnp.zeros((1, num_patches, self.config.dim_model), dtype=param_dtype)
        )


        # JiT blocks
        self.blocks = nnx.List([
            JiTBlock(
                hidden_size=self.config.dim_model,
                num_heads=self.config.heads,
                mlp_ratio=self.config.mlp_ratio,
                dtype=self.config.dtype,
                param_dtype=self.config.param_dtype,
                attn_impl=self.config.attn_impl,
                eps=self.config.eps,
                rngs=rngs
            )
            for _ in range(self.config.depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(
            self.config.dim_model,
            self.config.patch_size,
            self.config.channels,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            rngs=rngs
        )

        # Initialize pos embed and weights
        self._init_pos_embed()
        self._init_weights()

    def _init_pos_embed(self):
        """Initialize 2D sin-cos position embeddings."""
        grid_size = int(self.config.img_size // self.config.patch_size)
        embed_dim = self.config.dim_model

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
        for block in self.blocks:
            ada_ln = block.adaLN_modulation
            # Sequential with [silu, Linear] - access the Linear layer
            if isinstance(ada_ln, nnx.Sequential):
                linear_layer = ada_ln.layers[-1]
                if isinstance(linear_layer, nnx.Linear):
                    linear_layer.kernel.value = jnp.zeros_like(linear_layer.kernel.value)
                    if linear_layer.bias is not None:
                        linear_layer.bias.value = jnp.zeros_like(linear_layer.bias.value)

        # Zero out final layer adaLN (Sequential with [silu, Linear])
        final_ada_ln = self.final_layer.adaLN_modulation
        if isinstance(final_ada_ln, nnx.Sequential):
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
        
        # Ensure inputs are in the model's computation dtype
        if x_noisy.dtype != self.config.dtype:
            x_noisy = x_noisy.astype(self.config.dtype)
        if t.dtype != self.config.dtype:
            t = t.astype(self.config.dtype)

        # Patch embed
        x = self.patch_embed(x_noisy)  # [B, N, D]

        # Add positional embeddings
        x = x + self.pos_embed.value

        # Get time embedding
        c = self.t_embedder(t)  # [B, D]

        # Pass through blocks with RoPE
        for block in self.blocks:
            x = block(x, c)

        # Final layer
        x = self.final_layer(x, c)

        # Unpatchify
        return self._unpatchify(x)

    def _unpatchify(self, x_patches):
        B, N, D = x_patches.shape
        p = self.config.patch_size
        h = w = int(math.sqrt(N))
        C = self.config.channels

        x = x_patches.reshape(B, h, w, p, p, C)
        x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))  # [B, C, h, p, w, p]
        x = x.reshape(B, C, h * p, w * p)
        x = jnp.transpose(x, (0, 2, 3, 1))  # [B, H, W, C]
        return x


def celeba_generator_hf(split="train", batch_size=32, img_size=64, seed=42):
    """HuggingFace-based CelebA dataloader with shuffling before each epoch."""
    print(f"Loading CelebA dataset from HuggingFace (split={split})...")
    ds = load_dataset("flwrlabs/celeba", split=split, streaming=True)
    print(f"Dataset loaded successfully! Starting data stream...")

    epoch = 0
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
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
                epoch += 1
                ds = ds.shuffle(seed=seed + epoch, buffer_size=10_000)
                ds_iterator = iter(ds)
                break

        if len(batch_imgs) == batch_size:
            batch = np.stack(batch_imgs, axis=0)
            batch = (batch * 2.0) - 1.0
            yield batch, None


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


@nnx.jit
def step_fn(model, batch, rng, P_mean, P_std, noise_scale, t_eps):
    """Training step logic (not JIT'd - JIT applied in trainer loop)."""
    # Convert numpy batch to correct JAX dtype
    x_clean = jnp.asarray(batch, dtype=model.config.dtype)  # [B, H, W, C]
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



def sample(model, rng, img_size, steps=50, noise_scale=1.0, t_eps=1e-3, batch_size=1):
    """Generate samples using Euler method (matching JiT reference)."""
    print(f"Sampling with {steps} steps...")

    # Use model's dtype for sampling
    model_dtype = model.config.dtype
    x_current = jax.random.normal(rng, (batch_size, img_size, img_size, 3), dtype=model_dtype) * noise_scale
    # Use linspace from 0 to 0.99 with steps+1 points (don't go all the way to 1.0 to avoid instability)
    # This gives us 'steps' intervals to iterate through
    t_values = np.linspace(0.0, 0.99, steps + 1)
    dt = t_values[1] - t_values[0]

    for t_scalar in t_values[:-1]:
        t_vec = jnp.ones((batch_size, 1), dtype=model_dtype) * t_scalar
        # Model predicts the clean image x_1 from the current noised state
        x_clean_pred = model(x_current, t_vec)
        # Compute velocity: v = (x_1 - x_t) / (1 - t)
        # Use t_eps to avoid division by zero as t approaches 1
        denom = jnp.maximum(1.0 - t_scalar, t_eps)
        v = (x_clean_pred - x_current) / denom
        # Euler step
        x_current = x_current + v * dt

    return x_current


def main():
    print("=" * 60)
    print("JiT Denoising Training (CelebA 128x128)")
    print("=" * 60)

    print("\n[1/5] Setting up data loader...")
    if config.use_cached_data:
        if not is_cached(config.img_size, "train"):
            print("    Cached data not found. Caching dataset (one-time setup)...")
            cache_dataset(config.img_size)

        data_cfg = CachedDataConfig(
            batch_size=config.batch_size,
            img_size=config.img_size,
            seed=config.seed,
            shuffle=True,
        )
        train_gen_raw = make_cached_dataloader("train", data_cfg)
        # Increase prefetch buffer for better data pipeline overlap with mixed precision
        train_gen = PrefetchGenerator(train_gen_raw, buffer_size=12)
        print(f"    Using cached data with prefetch buffer: 12 batches")
        num_train_samples = 162000
    else:
        train_gen_raw = celeba_generator_hf(
            split="train",
            batch_size=config.batch_size,
            img_size=config.img_size,
            seed=config.seed
        )
        train_gen = PrefetchGenerator(train_gen_raw, buffer_size=4)
        print(f"    Using streaming data with prefetch buffer: 4 batches")
        num_train_samples = 162000

    steps_per_epoch = num_train_samples // config.batch_size
    num_epochs = config.epochs
    sample_every = 2000
    total_steps = steps_per_epoch * num_epochs

    # Setup model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    init_rngs = nnx.Rngs(init_rng)  # Create nnx.Rngs object from the key

    model = create_sharded_model(JiTDenoisingTransformer, config, init_rngs, mesh)
    tx = optax.adamw(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    with mesh:
        global_step = 0
        step_times = []
        training_start = time.perf_counter()

        first_batch, _ = next(train_gen)
        print(f"\nDebug - First batch: shape={first_batch.shape}, "
              f"range=[{first_batch.min():.3f}, {first_batch.max():.3f}]")

        train_imgs = first_batch[:8]
        train_imgs = (train_imgs + 1.0) / 2.0
        fig, axes = plt.subplots(1, 8, figsize=(16, 2))
        for i, ax in enumerate(axes):
            ax.imshow(train_imgs[i])
            ax.axis('off')
        plt.suptitle('Training Images (First Batch)')
        plt.tight_layout()
        plt.savefig(os.path.join(config.output_dir, "training_batch_sample.png"), dpi=150)
        plt.close()
        print(f"Training batch sample saved: {os.path.join(config.output_dir, 'training_batch_sample.png')}")

        print("\nGenerating pre-training sample...")
        pretrain_sample = sample(model, rng, img_size=config.img_size,
                                 noise_scale=config.noise_scale, t_eps=config.t_eps)
        img = np.array(pretrain_sample[0])
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        filepath = os.path.join(config.output_dir, "sample_pretraining.png")
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Pre-training Sample (Random Init)")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Pre-training sample saved: {filepath}")

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            epoch_losses = []
            epoch_start = time.perf_counter()

            for step_in_epoch in range(steps_per_epoch):
                step_start = time.perf_counter()

                batch, _ = next(train_gen)

                loss, grads, rng = step_fn(
                    model, batch, rng,
                    config.P_mean, config.P_std, config.noise_scale, config.t_eps
                )
                optimizer.update(model, grads)
                epoch_losses.append(float(loss))

                step_time = time.perf_counter() - step_start
                step_times.append(step_time)
                step_times = step_times[-100:]

                if step_in_epoch % 50 == 0:
                    avg_step_time = sum(step_times) / len(step_times)
                    samples_per_sec = config.batch_size / avg_step_time
                    steps_remaining = total_steps - global_step
                    eta_seconds = steps_remaining * avg_step_time
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:]) if epoch_losses else float(loss)

                    print(f"Epoch {epoch+1}/{num_epochs} | Step {step_in_epoch}/{steps_per_epoch} | "
                          f"Loss: {loss:.5f} (avg: {avg_loss:.5f}) | "
                          f"{samples_per_sec:.1f} samples/sec | ETA: {eta_str}")

                if global_step % sample_every == 0 and global_step > 0:
                    print(f"Sampling at step {global_step}...")
                    sample_out = sample(model, rng, img_size=config.img_size,
                                        noise_scale=config.noise_scale, t_eps=config.t_eps)

                    img = np.array(sample_out[0])
                    img = (img + 1.0) / 2.0
                    img = np.clip(img, 0, 1)

                    filename = f"sample_step{global_step:06d}_e{epoch+1}.png"
                    filepath = os.path.join(config.output_dir, filename)

                    plt.figure(figsize=(4, 4))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Step {global_step}")
                    plt.tight_layout()
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"Saved: {filepath}")

                global_step += 1

            epoch_time = time.perf_counter() - epoch_start
            epoch_samples = steps_per_epoch * config.batch_size
            epoch_samples_per_sec = epoch_samples / epoch_time
            print(f"Epoch {epoch+1} completed in {timedelta(seconds=int(epoch_time))} "
                  f"({epoch_samples_per_sec:.1f} samples/sec)")

    total_time = time.perf_counter() - training_start
    print("\n" + "="*60)
    print(f"Training complete! Total time: {timedelta(seconds=int(total_time))}")
    print("="*60)


if __name__ == "__main__":
    main()
