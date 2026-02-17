#!/usr/bin/env python3
"""
JiT (Just Image Transformer) implementation matching the official paper.
Uses AdaLN modulation, SwiGLU, QK norm, and proper training configuration.
"""

import sys
import os
import json
from datetime import datetime

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
    "batch_size": 32,
    "blr": 5e-5,  # Base learning rate (scaled by batch size / 256)
    "epochs": 50,
    "warmup_epochs": 5,
    "seed": 42,
    "P_mean": -0.8,  # Logit-normal params
    "P_std": 0.8,
    "t_eps": 1e-3,
    "noise_scale": 1.0,
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
        x = x * jnp.rsqrt(variance + self.eps)
        x = x.astype(input_dtype)
        return x * self.weight.value


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
# 4. Attention with QK Norm
# ==========================================

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
    """Multi-head attention with QK norm and RoPE."""
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nnx.Linear(dim, dim * 3, use_bias=True, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, use_bias=True, rngs=rngs)

        # QK Norm
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, rngs=rngs)

    def __call__(self, x, rope_cos_sin=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # [B, N, H, D] -> [B, H, N, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if provided
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            # Apply rotation (simplified)
            q_half, k_half = q[..., ::2], k[..., ::2]
            q_rot = jnp.stack([q_half * cos - q[..., 1::2] * sin,
                               q_half * sin + q[..., 1::2] * cos], axis=-1)
            k_rot = jnp.stack([k_half * cos - k[..., 1::2] * sin,
                               k_half * sin + k[..., 1::2] * cos], axis=-1)
            q = q_rot.reshape(q.shape)
            k = k_rot.reshape(k.shape)

        # Attention
        scale = self.head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)

        # Reshape back
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

        # Time embedding MLP
        self.t_embedder = nnx.Sequential(
            nnx.Linear(1, config['dim_model'], use_bias=True, rngs=rngs),
            nnx.silu,
            nnx.Linear(config['dim_model'], config['dim_model'], use_bias=True, rngs=rngs)
        )

        # Fixed position embeddings (will be initialized)
        num_patches = (config['img_size'] // config['patch_size']) ** 2
        self.pos_embed = nnx.Param(jnp.zeros((1, num_patches, config['dim_model'])))

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

        # Initialize pos embed
        self._init_pos_embed()

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

        # Pass through blocks
        for block in self.blocks:
            x = block(x, c, rope=None)

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


# ==========================================
# 10. Training
# ==========================================

def create_model_and_optimizer(rng_key, config, mesh):
    """Create model and optimizer with proper sharding."""
    with mesh:
        rngs = nnx.Rngs(rng_key)
        model = JiTDenoisingTransformer(config, rngs)

        # Initialize with dummy data
        dummy_x = jnp.ones((1, config['img_size'], config['img_size'], config['channels']))
        dummy_t = jnp.ones((1, 1))
        _ = model(dummy_x, dummy_t)

        # Learning rate with scaling
        base_lr = config['blr']
        effective_batch = config['batch_size']
        lr = base_lr * effective_batch / 256

        tx = optax.adamw(learning_rate=lr)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    return model, optimizer


@nnx.jit
def train_step(model, optimizer, batch, rng, config):
    """JiT-style training step with v-loss."""
    x_clean = batch  # [B, H, W, C]
    B = x_clean.shape[0]

    rng, t_rng, n_rng = jax.random.split(rng, 3)

    # Sample t from logit-normal
    z = jax.random.normal(t_rng, (B, 1)) * config['P_std'] + config['P_mean']
    t = jax.nn.sigmoid(z)

    # Noise
    noise = jax.random.normal(n_rng, x_clean.shape) * config['noise_scale']

    # Interpolate
    t_img = t.reshape(B, 1, 1, 1)
    x_t = t_img * x_clean + (1 - t_img) * noise

    # Target velocity
    v_target = (x_clean - x_t) / (1 - t_img + config['t_eps'])

    def loss_fn(m):
        x_pred = m(x_t, t)
        v_pred = (x_pred - x_t) / (1 - t_img + config['t_eps'])

        # MSE loss (matching JiT: mean over all dims then mean over batch)
        loss = jnp.mean((v_pred - v_target) ** 2, axis=(1, 2, 3))
        return jnp.mean(loss)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss, rng


def sample(model, rng, img_size, steps=50, config=None):
    """Generate samples using Euler method."""
    print(f"Sampling with {steps} steps...", flush=True)

    x_current = jax.random.normal(rng, (1, img_size, img_size, 3)) * config['noise_scale']
    t_values = np.linspace(0, 0.99, steps)
    dt = t_values[1] - t_values[0]

    for t_scalar in t_values[:-1]:
        t_vec = jnp.ones((1, 1)) * t_scalar
        x_clean_pred = model(x_current, t_vec)
        denom = 1.0 - t_scalar + config['t_eps']
        v = (x_clean_pred - x_current) / denom
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

    # Setup data
    print("\n[1/5] Setting up data loader...", flush=True)
    train_gen = celeba_generator_hf(
        split="train",
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size']
    )

    data_sharding = NamedSharding(mesh, PartitionSpec("devices"))

    # Setup model
    print("[2/5] Initializing model...", flush=True)
    rng = jax.random.PRNGKey(CONFIG['seed'])
    rng, init_rng = jax.random.split(rng)

    model, optimizer = create_model_and_optimizer(init_rng, CONFIG, mesh)
    print(f"    Model: {CONFIG['dim_model']}d, {CONFIG['depth']} layers", flush=True)
    print(f"    Batch size: {CONFIG['batch_size']}, BLR: {CONFIG['blr']}", flush=True)

    # Setup output
    output_dir = "generated_samples_jit"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[3/5] Output directory: {output_dir}/", flush=True)

    # Training config
    steps_per_epoch = 162000 // CONFIG['batch_size']
    num_epochs = CONFIG['epochs']
    sample_every = 2000

    print(f"[4/5] Training configuration:", flush=True)
    print(f"      Steps per epoch: {steps_per_epoch}", flush=True)
    print(f"      Total epochs: {num_epochs}", flush=True)
    print(f"      Sample every: {sample_every} steps", flush=True)
    print(f"\n{'='*60}", flush=True)
    print("[5/5] Starting training...", flush=True)
    print('='*60, flush=True)

    with mesh:
        global_step = 0

        # Debug first batch
        first_batch = next(train_gen)
        print(f"\nDebug - First batch: shape={first_batch.shape}, "
              f"range=[{first_batch.min():.3f}, {first_batch.max():.3f}]", flush=True)

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===", flush=True)
            epoch_losses = []

            for step_in_epoch in range(steps_per_epoch):
                batch = next(train_gen)
                batch = jax.device_put(batch, data_sharding)

                loss, rng = train_step(model, optimizer, batch, rng, CONFIG)
                epoch_losses.append(float(loss))

                if step_in_epoch % 100 == 0:
                    avg_loss = sum(epoch_losses[-100:]) / len(epoch_losses[-100:]) if epoch_losses else float(loss)
                    print(f"Epoch {epoch + 1} | Step {step_in_epoch}/{steps_per_epoch} | "
                          f"Loss: {loss:.5f} | Avg: {avg_loss:.5f}", flush=True)

                if global_step % sample_every == 0 and global_step > 0:
                    print(f"Sampling at step {global_step}...", flush=True)
                    sample_out = sample(model, rng, img_size=CONFIG['img_size'], config=CONFIG)

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
