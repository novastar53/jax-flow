#!/usr/bin/env python3
"""
Optimized CelebA Flow Matching with jaxpt modules.
Uses flax.nnx and optimized attention (cuDNN where available).
"""

import sys
import os
import json

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

# Add jaxpt to path
sys.path.insert(0, os.path.expanduser("~/dev/jaxpt/src"))
from jaxpt.modules.config import Config
from jaxpt.modules.attention import _calc_slow_attn
from jaxpt.modules.mlp import GLU
from jaxpt.modules.position import calc_rope_omega_llama, RoPE_Llama

# ==========================================
# Non-Causal Self-Attention for Images
# ==========================================

def _calc_attention_non_causal(
    query,
    key,
    value,
    mask=None,
    bias=None,
    implementation: str | None = None,
):
    """Non-causal attention for bidirectional image denoising."""
    output_shape = jnp.asarray(query).shape

    match implementation:
        case "xla" | "cudnn":
            # Non-causal: is_causal=False allows full bidirectional attention
            out = jax.nn.dot_product_attention(
                query,
                key,
                value,
                mask=mask,
                bias=bias,
                is_causal=False,  # KEY DIFFERENCE: No causal masking
                implementation=implementation,
            )
        case _:
            out, _ = _calc_slow_attn(query, key, value, mask, bias)

    return jnp.reshape(out, output_shape)


class SelfAttention_w_RoPE(nnx.Module, RoPE_Llama):
    """Non-causal self-attention with RoPE for image denoising."""

    def __init__(self, config, rope_omega: nnx.Variable, rngs: nnx.Rngs):
        RoPE_Llama.__init__(self, omega=rope_omega)

        self.c_attn = nnx.Linear(
            config.n_embed,
            3 * config.n_embed,
            use_bias=config.attention_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            use_bias=config.attention_bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.n_head = config.n_head
        self.implementation = config.sdpa_implementation

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # B, T, 3 * C
        q, k, v = jnp.split(qkv, 3, axis=-1)  # 3 * (B, T, C)

        q = jnp.reshape(q, (B, T, self.n_head, C // self.n_head))
        k = jnp.reshape(k, (B, T, self.n_head, C // self.n_head))
        v = jnp.reshape(v, (B, T, self.n_head, C // self.n_head))

        # Apply RoPE to q and k
        q = self.apply_rope(q)
        k = self.apply_rope(k)

        # Non-causal attention
        y = _calc_attention_non_causal(q, k, v, mask=mask, implementation=self.implementation)

        y = jnp.reshape(y, (B, T, C))
        y = self.c_proj(y)
        return y


# ==========================================
# 1. Configuration
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
    "mlp_dim": 1024,
    "batch_size": 64,
    "lr": 1e-4,
    "epochs": 1,
    "seed": 42
}

# Create jaxpt Config
class DenoiseConfig(Config):
    def __init__(self):
        super().__init__()
        self.n_embed = CONFIG["dim_model"]
        self.n_head = CONFIG["heads"]
        self.n_kv_head = CONFIG["heads"]  # No GQA for now
        self.n_layer = CONFIG["depth"]
        self.n_mlp_hidden = CONFIG["mlp_dim"]
        self.block_size = (CONFIG["img_size"] // CONFIG["patch_size"]) ** 2
        self.sdpa_implementation = "cudnn" if jax.default_backend() == "gpu" else "xla"
        self.glu_activation = "swish"
        self.mlp_bias = False
        self.attention_bias = False
        self.ln_epsilon = 1e-6
        self.init_stddev = 0.02
        self.rope_base_freq = 10000.0
        self.dtype = jnp.bfloat16  # cuDNN attention requires fp16/bf16/fp8
        self.param_dtype = jnp.float32
        self.use_cache = False

        # Sharding specs - set to None to disable partitioning annotations
        # The mesh context will handle device placement automatically
        self.glu_fc_kernel_sharding = None
        self.glu_fc_bias_sharding = None
        self.glu_gate_kernel_sharding = None
        self.glu_gate_bias_sharding = None
        self.glu_proj_kernel_sharding = None
        self.glu_proj_bias_sharding = None

        self.attn_c_attn_kernel_sharding = None
        self.attn_c_attn_bias_sharding = None
        self.attn_c_proj_kernel_sharding = None
        self.attn_c_proj_bias_sharding = None

# ==========================================
# 2. The Model (flax.nnx)
# ==========================================

class LinearBottleneckPatchEmbed(nnx.Module):
    def __init__(self, patch_size: int, dim_bottleneck: int, dim_model: int, rngs: nnx.Rngs):
        self.patch_size = patch_size
        self.dim_bottleneck = dim_bottleneck
        self.dim_model = dim_model

        # Two-stage bottleneck: compress then expand
        self.compress = nnx.Linear(
            patch_size * patch_size * 3,
            dim_bottleneck,
            use_bias=False,
            rngs=rngs
        )
        self.expand = nnx.Linear(
            dim_bottleneck,
            dim_model,
            use_bias=False,
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

class TransformerBlock(nnx.Module):
    def __init__(self, config: DenoiseConfig, rope_omega: nnx.Variable, rngs: nnx.Rngs):
        # Use LayerNorm to match 07 model
        self.ln1 = nnx.LayerNorm(config.n_embed, epsilon=config.ln_epsilon, rngs=rngs)
        self.attn = SelfAttention_w_RoPE(config, rope_omega, rngs)  # Non-causal for images
        self.ln2 = nnx.LayerNorm(config.n_embed, epsilon=config.ln_epsilon, rngs=rngs)
        self.mlp = GLU(config, rngs)

    def __call__(self, x):
        # Pre-norm with RMSNorm (better for training stability)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DenoisingTransformer(nnx.Module):
    def __init__(self, config: dict, rngs: nnx.Rngs):
        cfg = DenoiseConfig()
        self.config = config
        self.patch_embed = LinearBottleneckPatchEmbed(
            config['patch_size'],
            config['dim_bottleneck'],
            config['dim_model'],
            rngs
        )

        # Time embedding MLP
        self.time_embed = nnx.Sequential(
            nnx.Linear(1, config['dim_model'], rngs=rngs),
            nnx.silu,
            nnx.Linear(config['dim_model'], config['dim_model'], rngs=rngs)
        )

        # Precompute RoPE frequencies
        self.rope_omega = calc_rope_omega_llama(
            n_embed=config['dim_model'],
            n_head=config['heads'],
            block_size=(config['img_size'] // config['patch_size']) ** 2,
            rope_base_freq=cfg.rope_base_freq,
            dtype=cfg.dtype
        )

        # Transformer blocks
        self.blocks = nnx.List([
            TransformerBlock(cfg, self.rope_omega, rngs)
            for _ in range(config['depth'])
        ])

        # Output head
        self.ln_f = nnx.LayerNorm(config['dim_model'], epsilon=cfg.ln_epsilon, rngs=rngs)
        self.head = nnx.Linear(config['dim_model'], config['dim_raw'], use_bias=False, rngs=rngs)

    def __call__(self, x_noisy, t):
        # Patch embed
        x = self.patch_embed(x_noisy)  # [B, N, D]

        # Add time embedding
        t_emb = self.time_embed(t)  # [B, D]
        t_emb = jnp.expand_dims(t_emb, axis=1)  # [B, 1, D]
        x = x + t_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.ln_f(x)
        x = self.head(x)

        # Unpatchify
        return self._unpatchify(x)

    def _unpatchify(self, x_patches):
        B, N, D = x_patches.shape
        p = self.config['patch_size']
        h = w = int(math.sqrt(N))
        C = 3  # RGB

        x = x_patches.reshape(B, h, w, p, p, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, h * p, w * p, C)
        return x

# ==========================================
# 3. Data Loading
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
            batch = (batch * 2.0) - 1.0
            yield batch

# ==========================================
# 4. Training
# ==========================================

def create_model_and_optimizer(rng_key, config, mesh):
    """Create model and optimizer with proper sharding."""
    with mesh:
        rngs = nnx.Rngs(rng_key)
        model = DenoisingTransformer(config, rngs)

        # Initialize with dummy data
        dummy_x = jnp.ones((1, config['img_size'], config['img_size'], config['channels']))
        dummy_t = jnp.ones((1, 1))

        # Run once to initialize
        _ = model(dummy_x, dummy_t)

        # Create optimizer with weight decay mask (exclude biases and norms)
        graphdef, params, _ = nnx.split(model, nnx.Param, nnx.Variable)
        weight_decay_mask = jax.tree.map(
            lambda x: len(x.value.shape) > 1,
            params,
            is_leaf=lambda n: isinstance(n, nnx.Param),
        )

        tx = optax.adamw(
            learning_rate=config['lr'],
            weight_decay=0.1,
            mask=weight_decay_mask,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    return model, optimizer


def download_checkpoint_from_hf(repo_id: str, local_dir: str) -> bool:
    """Download checkpoint from HuggingFace Hub if available."""
    try:
        print(f"Checking for checkpoint at {repo_id}...", flush=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            allow_patterns=["*.nnx", "*.json"],
        )
        # Check if we actually got the checkpoint
        checkpoint_path = os.path.join(local_dir, "model_state.nnx")
        if os.path.exists(checkpoint_path):
            print(f"Downloaded checkpoint from HuggingFace", flush=True)
            return True
        return False
    except Exception as e:
        print(f"Could not download from HuggingFace: {e}", flush=True)
        return False


def load_model_from_checkpoint(checkpoint_dir, config, mesh):
    """Load model from checkpoint if it exists."""
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model_state")
    config_path = os.path.join(checkpoint_dir, "config.json")

    if not os.path.exists(checkpoint_path):
        return None, None

    print(f"Loading checkpoint from {checkpoint_dir}...", flush=True)

    with mesh:
        rngs = nnx.Rngs(jax.random.PRNGKey(config['seed']))
        model = DenoisingTransformer(config, rngs)

        # Initialize with dummy data
        dummy_x = jnp.ones((1, config['img_size'], config['img_size'], config['channels']))
        dummy_t = jnp.ones((1, 1))
        _ = model(dummy_x, dummy_t)

        # Load checkpoint state using orbax
        graphdef, state = nnx.split(model)
        cp = ocp.StandardCheckpointer()
        state = cp.restore(checkpoint_path, target=state)
        model = nnx.merge(graphdef, state)

        # Create optimizer
        graphdef, params, _ = nnx.split(model, nnx.Param, nnx.Variable)
        weight_decay_mask = jax.tree.map(
            lambda x: len(x.value.shape) > 1,
            params,
            is_leaf=lambda n: isinstance(n, nnx.Param),
        )

        tx = optax.adamw(
            learning_rate=config['lr'],
            weight_decay=0.1,
            mask=weight_decay_mask,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    print("Checkpoint loaded successfully!", flush=True)
    return model, optimizer


@nnx.jit
def train_step(model, optimizer, batch, rng):
    """Single training step with v-loss."""
    x_clean = batch
    B = x_clean.shape[0]

    rng, t_rng, n_rng = jax.random.split(rng, 3)

    # Logit-normal distribution for t
    u = jax.random.normal(t_rng, (B, 1))
    logit_t = -0.8 + 0.8 * u
    t = jax.nn.sigmoid(logit_t)

    x_noise = jax.random.normal(n_rng, x_clean.shape)

    # Interpolate
    t_bc = t.reshape((B, 1, 1, 1))
    x_t = (1 - t_bc) * x_noise + t_bc * x_clean

    def loss_fn(m):
        x_pred = m(x_t, t)
        # v-loss with x-prediction
        denom = jnp.clip(1 - t_bc, 0.05, 1.0)
        v_pred = (x_pred - x_t) / denom
        v_target = (x_clean - x_t) / denom
        return jnp.mean((v_pred - v_target) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss, rng

def sample(model, rng, img_size, steps=100):
    """Generate samples using Euler method."""
    print(f"Sampling with {steps} steps...", flush=True)

    x_current = jax.random.normal(rng, (1, img_size, img_size, 3))
    t_values = np.linspace(0, 0.99, steps)
    dt = t_values[1] - t_values[0]

    for t_scalar in t_values[:-1]:
        t_vec = jnp.ones((1, 1)) * t_scalar
        x_clean_pred = model(x_current, t_vec)
        denom = 1.0 - t_scalar
        v = (x_clean_pred - x_current) / denom
        x_current = x_current + v * dt

    return x_current

# ==========================================
# 5. Main
# ==========================================

def main():
    print("=" * 60, flush=True)
    print("Optimized CelebA Flow Matching Training (jaxpt modules)", flush=True)
    print("=" * 60, flush=True)

    # Setup devices and mesh
    devices = jax.devices()
    num_devices = len(devices)
    print(f"\n[0/5] Device setup: {num_devices} device(s)", flush=True)
    print(f"      Platform: {jax.default_backend()}", flush=True)
    print(f"      Devices: {devices}", flush=True)
    mesh = Mesh(devices, ["devices"])

    # Setup data
    print("\n[1/5] Setting up data loader...", flush=True)
    train_gen = celeba_generator_hf(
        split="train",
        batch_size=CONFIG['batch_size'],
        img_size=CONFIG['img_size']
    )

    # Create data sharding
    data_sharding = NamedSharding(mesh, PartitionSpec("devices"))

    # Setup model
    print("[2/5] Initializing model...", flush=True)
    rng = jax.random.PRNGKey(CONFIG['seed'])
    rng, init_rng = jax.random.split(rng)

    # Check for existing checkpoint (local first, then HuggingFace)
    checkpoint_dir = "model_checkpoint"
    hf_repo_id = "vikramp/jax_jit"

    # First try loading from local checkpoint
    model, optimizer = load_model_from_checkpoint(checkpoint_dir, CONFIG, mesh)

    # If no local checkpoint, try downloading from HuggingFace
    if model is None:
        print("No local checkpoint found. Checking HuggingFace Hub...", flush=True)
        if download_checkpoint_from_hf(hf_repo_id, checkpoint_dir):
            model, optimizer = load_model_from_checkpoint(checkpoint_dir, CONFIG, mesh)

    if model is None:
        print("No checkpoint found. Creating new model...", flush=True)
        model, optimizer = create_model_and_optimizer(init_rng, CONFIG, mesh)
        print(f"    Model initialized: {CONFIG['dim_model']}d, {CONFIG['depth']} layers", flush=True)
    else:
        print(f"    Model loaded: {CONFIG['dim_model']}d, {CONFIG['depth']} layers", flush=True)

    print(f"    Using cuDNN attention: {jax.devices()[0].platform == 'gpu'}", flush=True)

    # Setup output
    output_dir = "generated_samples_fast"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[3/5] Output directory: {output_dir}/", flush=True)

    # Training config
    steps_per_epoch = 162000 // CONFIG['batch_size']
    num_epochs = CONFIG['epochs']
    sample_every = 2000

    # Checkpoint config
    checkpoint_dir = os.path.abspath("model_checkpoint")
    checkpoint_every_epochs = 1  # Save every N epochs (also saves at start and end)
    os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(model, optimizer, epoch):
        """Save model checkpoint."""
        print(f"\nSaving checkpoint at epoch {epoch}...", flush=True)

        # Save model state using orbax
        graphdef, state = nnx.split(model)
        checkpoint_path = os.path.join(checkpoint_dir, "model_state")
        cp = ocp.StandardCheckpointer()
        cp.save(checkpoint_path, state)
        cp.wait_until_finished()

        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(CONFIG, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_dir}/", flush=True)

    print(f"[4/5] Training configuration:", flush=True)
    print(f"      Steps per epoch: {steps_per_epoch}", flush=True)
    print(f"      Total epochs: {num_epochs}", flush=True)
    print(f"      Sample every: {sample_every} steps", flush=True)
    print(f"      Checkpoint every: {checkpoint_every_epochs} epoch(s)", flush=True)
    print(f"\n{'='*60}", flush=True)
    print("[5/5] Starting training loop...", flush=True)
    print('='*60, flush=True)

    with mesh:
        global_step = 0

        # Save initial checkpoint (epoch 0)
        save_checkpoint(model, optimizer, 0)

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===", flush=True)

            for step_in_epoch in range(steps_per_epoch):
                batch = next(train_gen)

                # Put batch on devices
                batch = jax.device_put(batch, data_sharding)

                loss, rng = train_step(model, optimizer, batch, rng)

                if step_in_epoch % 100 == 0:
                    print(f"Epoch {epoch + 1} | Step {step_in_epoch}/{steps_per_epoch} | "
                          f"Global Step {global_step} | Loss: {loss:.5f}", flush=True)

                if global_step % sample_every == 0 and global_step > 0:
                    print(f"Sampling at global step {global_step} (epoch {epoch + 1})...", flush=True)

                    sample_out = sample(model, rng, img_size=CONFIG['img_size'])

                    img = np.array(sample_out[0])
                    img = (img + 1.0) / 2.0
                    img = np.clip(img, 0, 1)

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

            # Save checkpoint every N epochs (using modulus)
            if (epoch + 1) % checkpoint_every_epochs == 0:
                save_checkpoint(model, optimizer, epoch + 1)

    print("\n" + "="*60, flush=True)
    print("Training complete!", flush=True)
    print("="*60, flush=True)

    # Upload to Hugging Face
    print("\nUploading to Hugging Face...", flush=True)
    try:
        login()
        upload_folder(
            folder_path=checkpoint_dir,
            repo_id="vikramp/jax_jit",
            repo_type="model"
        )
        print("Successfully uploaded to Hugging Face!", flush=True)
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}", flush=True)

if __name__ == "__main__":
    main()
