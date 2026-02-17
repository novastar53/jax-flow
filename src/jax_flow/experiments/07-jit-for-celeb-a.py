import sys
import json
import os

# Force unbuffered output for real-time logging
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace')
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace')

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import math

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image

import torch
from torch.utils.data import DataLoader

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "img_size": 64,        # 64x64 images
    "patch_size": 4,       # Smaller patches for finer detail
    "dim_raw": 4 * 4 * 3,  # Raw patch dimension (48)
    "channels": 3,
    "dim_bottleneck": 128,  # Linear Bottleneck dimension
    "dim_model": 256,       # Transformer hidden size
    "depth": 6,             # Number of layers
    "heads": 8,
    "mlp_dim": 1024,        # 4x dim_model
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 50,
    "seed": 42
}

# ==========================================
# 2. The Model (Flax)
# ==========================================

class LinearBottleneckPatchEmbed(nn.Module):
    patch_size: int
    dim_bottleneck: int
    dim_model: int

    @nn.compact
    def __call__(self, x):
        # x input: [Batch, H, W, C]
        B, H, W, C = x.shape
        p = self.patch_size
        
        # Patchify logic in JAX
        # 1. Reshape to break H, W into patches
        x = x.reshape(B, H // p, p, W // p, p, C)
        # 2. Transpose to [B, H//p, W//p, p, p, C]
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # 3. Flatten to [B, Num_Patches, Flat_Patch_Dim]
        x = x.reshape(B, -1, p * p * C)
        
        # --- The Linear Bottleneck ---
        # 1. Compress (Filter Noise)
        x = nn.Dense(self.dim_bottleneck, use_bias=False, name="compress")(x)
        # 2. Expand (Project to Model Dim)
        x = nn.Dense(self.dim_model, use_bias=False, name="expand")(x)
        
        return x

def apply_rotary_pos_emb(q, k, freq_cis):
    """Apply rotary position embeddings to q and k."""
    # Reshape for complex multiplication: [B, H, N, D] -> [B, H, N, D//2, 2]
    q_reshape = q.reshape(q.shape[:-1] + (-1, 2))
    k_reshape = k.reshape(k.shape[:-1] + (-1, 2))

    # View as complex numbers
    q_complex = jax.lax.complex(q_reshape[..., 0], q_reshape[..., 1])
    k_complex = jax.lax.complex(k_reshape[..., 0], k_reshape[..., 1])

    # Apply rotation
    q_rotated = q_complex * freq_cis
    k_rotated = k_complex * freq_cis

    # Convert back to real
    q_out = jnp.stack([jnp.real(q_rotated), jnp.imag(q_rotated)], axis=-1).reshape(q.shape)
    k_out = jnp.stack([jnp.real(k_rotated), jnp.imag(k_rotated)], axis=-1).reshape(k.shape)

    return q_out, k_out

def precompute_freqs_cis(dim, end, theta=10000.0):
    """Precompute rotary position embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[:dim//2].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jax.lax.complex(jnp.cos(freqs), jnp.sin(freqs))
    return freqs_cis

class RoPEAttention(nn.Module):
    dim_model: int
    heads: int

    @nn.compact
    def __call__(self, x):
        B, N, D = x.shape
        head_dim = D // self.heads

        # Q, K, V projections
        q = nn.Dense(D, use_bias=False, name="q")(x)
        k = nn.Dense(D, use_bias=False, name="k")(x)
        v = nn.Dense(D, use_bias=False, name="v")(x)

        # Reshape to [B, heads, N, head_dim]
        q = q.reshape(B, N, self.heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.heads, head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE to q and k
        freqs_cis = precompute_freqs_cis(head_dim, N)
        q, k = apply_rotary_pos_emb(q, k, freqs_cis)

        # Attention: [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]
        scale = head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        # [B, H, N, N] @ [B, H, N, D] -> [B, H, N, D]
        out = jnp.matmul(attn, v)

        # Reshape back: [B, H, N, D] -> [B, N, H*D]
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)

        # Output projection
        out = nn.Dense(D, use_bias=False, name="out")(out)
        return out

class TransformerBlock(nn.Module):
    dim_model: int
    heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        # Pre-Norm Architecture with RoPE attention

        # Attention with RoPE
        y = nn.LayerNorm()(x)
        y = RoPEAttention(dim_model=self.dim_model, heads=self.heads)(y)
        x = x + y

        # MLP (SwiGLU as used in the paper)
        y = nn.LayerNorm()(x)
        y_gate = nn.Dense(self.mlp_dim, use_bias=False)(y)
        y_up = nn.Dense(self.mlp_dim, use_bias=False)(y)
        y = jax.nn.swish(y_gate) * y_up
        y = nn.Dense(self.dim_model, use_bias=False)(y)
        x = x + y
        return x


class JustProteinTransformer(nn.Module):
    config: dict

    def setup(self):
        cfg = self.config
        self.patch_embed_layer = LinearBottleneckPatchEmbed(
            patch_size=cfg['patch_size'],
            dim_bottleneck=cfg['dim_bottleneck'],
            dim_model=cfg['dim_model']
        )

    @nn.compact
    def __call__(self, x_noisy, t):
        cfg = self.config

        # 1. Patch Embedding
        # [B, H, W, C] -> [B, N, D]
        x = self.patch_embed_layer(x_noisy)

        # Note: No fixed position embeddings needed - RoPE handles position in attention

        # 2. Add Time Embeddings
        # [B, 1] -> [B, D]
        t_emb = nn.Dense(cfg['dim_model'])(t)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(cfg['dim_model'])(t_emb)
        # Broadcast: [B, 1, D]
        t_emb = jnp.expand_dims(t_emb, axis=1)
        x = x + t_emb

        # 3. Transformer Blocks
        for _ in range(cfg['depth']):
            x = TransformerBlock(
                dim_model=cfg['dim_model'],
                heads=cfg['heads'],
                mlp_dim=cfg['mlp_dim']
            )(x)

        # 4. Output Head (Predicts Clean Patches)
        x = nn.LayerNorm()(x)
        # Project back to raw patch size [B, N, patch_size^2 * C]
        x_pred_patches = nn.Dense(cfg['dim_raw'])(x)

        # 5. Reconstruct Image
        return self.unpatchify(x_pred_patches)

    def unpatchify(self, x_patches):
        """
        Reconstruct [B, H, W, C] from patches [B, N, D]
        """
        B, N, D = x_patches.shape
        p = self.config['patch_size']
        h = w = int(math.sqrt(N))
        C = x_patches.shape[-1] // (p ** 2)
        
        # Reshape to [B, h, w, p, p, C]
        x = x_patches.reshape(B, h, w, p, p, C)
        # Transpose to [B, h, p, w, p, C]
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # Flatten to [B, H, W, C]
        x = x.reshape(B, h * p, w * p, C)
        return x

    def patch_embed(self, x):
        return self.patch_embed_layer(x)


def celeba_generator_hf(split="train", batch_size=32, img_size=64):
    """
    HuggingFace-based CelebA dataloader for the JIT Flow Matching model.
    Uses the flwrlabs/celeba dataset.
    """
    # Load CelebA from HuggingFace (streaming mode to avoid disk space issues)
    print(f"Loading CelebA dataset from HuggingFace (split={split})...", flush=True)
    ds = load_dataset("flwrlabs/celeba", split=split, streaming=True)
    print(f"Dataset loaded successfully! Starting data stream...", flush=True)

    # For memory efficiency, we'll batch as we iterate
    ds_iterator = iter(ds)

    while True:
        batch_imgs = []
        for _ in range(batch_size):
            try:
                sample = next(ds_iterator)
                img = sample["image"]

                # Resize to target size
                img = img.resize((img_size, img_size))

                # Convert PIL to numpy array [H, W, C]
                img_array = np.array(img).astype(np.float32) / 255.0  # [0, 1]

                # Ensure 3 channels (some images might be grayscale)
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                elif img_array.shape[-1] == 4:  # RGBA
                    img_array = img_array[:, :, :3]

                batch_imgs.append(img_array)
            except StopIteration:
                # Restart iterator if exhausted
                ds_iterator = iter(ds)
                break

        if len(batch_imgs) == batch_size:
            # Stack into batch [B, H, W, C]
            batch = np.stack(batch_imgs, axis=0)

            # Rescale to [-1, 1] for Flow Matching
            batch = (batch * 2.0) - 1.0

            yield batch

def sample_protein(state, rng, img_size, steps=100):
    """
    Robust Sampler: Stops just before t=1 to avoid singularity.
    """
    print(f"Sampling with {steps} steps...", flush=True)
    
    # 1. Start with Noise
    x_current = jax.random.normal(rng, (1, img_size, img_size, 3))
    
    # 2. Time steps (0.0 to 0.99)
    # avoiding exactly 1.0 prevents division by zero
    t_values = np.linspace(0, 0.99, steps)
    dt = t_values[1] - t_values[0]
    
    for t_scalar in t_values[:-1]:
        # Broadcast t
        t_vec = jnp.ones((1, 1)) * t_scalar
        
        # Predict Clean Image (x_1)
        x_clean_pred = state.apply_fn({'params': state.params}, x_current, t_vec)
        
        # Calculate Velocity: v = (x_1 - x_t) / (1 - t)
        denom = 1.0 - t_scalar
        v = (x_clean_pred - x_current) / denom
        
        # Euler Step
        x_current = x_current + v * dt
        
    return x_current
# ==========================================
# 4. Training State & Step
# ==========================================

def create_train_state(rng, config):
    model = JustProteinTransformer(config)
    # Init dummy input [1, H, W, C] and time [1, 1]
    dummy_x = jnp.ones([1, config['img_size'], config['img_size'], config['channels']])
    dummy_t = jnp.ones([1, 1])
    
    params = model.init(rng, dummy_x, dummy_t)['params']
    
    tx = optax.adamw(learning_rate=config['lr'])
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@jax.jit
def train_step(state, batch, rng):
    x_clean = batch
    B, H, W, C = x_clean.shape

    rng, t_rng, n_rng = jax.random.split(rng, 3)
    # Logit-normal distribution for t (as used in the paper, μ=-0.8, σ=0.8)
    u = jax.random.normal(t_rng, (B, 1))
    logit_t = -0.8 + 0.8 * u
    t = jax.nn.sigmoid(logit_t)

    x_noise = jax.random.normal(n_rng, x_clean.shape)

    # Interpolate: x_t = (1 - t) * noise + t * x_clean
    t_bc = t.reshape((B, 1, 1, 1))
    x_t = (1 - t_bc) * x_noise + t_bc * x_clean

    def loss_fn(params):
        x_pred = state.apply_fn({'params': params}, x_t, t)
        # v-loss with x-prediction (Section 4.3 and Table 1 of paper)
        # Transform x_pred to velocity space: v_pred = (x_pred - x_t) / (1 - t)
        # Clip denominator to avoid division by zero near t=1 (mentioned in paper)
        denom = jnp.clip(1 - t_bc, 0.05, 1.0)
        v_pred = (x_pred - x_t) / denom
        # Target velocity: v_target = (x_clean - x_t) / (1 - t)
        v_target = (x_clean - x_t) / denom
        loss = jnp.mean((v_pred - v_target) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    state = state.apply_gradients(grads=grads)

    return state, loss, rng


def main():
    print("=" * 60, flush=True)
    print("Starting CelebA Flow Matching Training", flush=True)
    print("=" * 60, flush=True)

    # --- 1. Setup Data ---
    print("\n[1/4] Setting up data loader...", flush=True)
    train_gen = celeba_generator_hf(split="train",
                                batch_size=CONFIG['batch_size'],
                                img_size=CONFIG['img_size'])

    # --- 2. Setup Model ---
    print("[2/4] Initializing model...", flush=True)
    rng = jax.random.PRNGKey(CONFIG['seed'])
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, CONFIG)
    print(f"    Model initialized: {CONFIG['dim_model']}d, {CONFIG['depth']} layers", flush=True)

    # --- 3. Setup Output Directory ---
    output_dir = "generated_samples"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[3/4] Output directory: {output_dir}/", flush=True)

    # --- 4. Training Configuration ---
    # CelebA has ~162k training images
    # With batch_size=32, that's ~5062 steps per epoch
    steps_per_epoch = 162000 // CONFIG['batch_size']
    num_epochs = 10
    sample_every = 2000

    print(f"[4/4] Training configuration:", flush=True)
    print(f"      Steps per epoch: {steps_per_epoch}", flush=True)
    print(f"      Total epochs: {num_epochs}", flush=True)
    print(f"      Sample every: {sample_every} steps", flush=True)
    print(f"\n{'='*60}", flush=True)
    print("Starting training loop...", flush=True)
    print('='*60, flush=True)

    global_step = 0
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===", flush=True)

        for step_in_epoch in range(steps_per_epoch):
            batch = next(train_gen)

            # v-loss train step
            state, loss, rng = train_step(state, batch, rng)

            if step_in_epoch % 100 == 0:
                print(f"Epoch {epoch + 1} | Step {step_in_epoch}/{steps_per_epoch} | Global Step {global_step} | Loss: {loss:.5f}", flush=True)

            if global_step % sample_every == 0 and global_step > 0:
                print(f"Sampling at global step {global_step} (epoch {epoch + 1})...", flush=True)

                # Sample
                sample_out = sample_protein(state, rng, img_size=CONFIG['img_size'])

                # Convert to image
                img = np.array(sample_out[0])
                img = (img + 1.0) / 2.0
                img = np.clip(img, 0, 1)

                # Save to file with step and epoch in filename
                filename = f"sample_step{global_step:06d}_epoch{epoch + 1}.png"
                filepath = os.path.join(output_dir, filename)

                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Step {global_step}, Epoch {epoch + 1}")
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
