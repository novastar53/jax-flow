import sys
import os

import jax
import jax.numpy as jnp
import flax.linen as nn
import torch
from flax.training import train_state
import optax
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.cath import download_cath_dataset, cath_generator

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "img_size": 128,        # 128x128 Distance Map
    "patch_size": 8,       # Large patches
    "dim_raw": 8*8*1,     # Raw patch dimension (256)
    "dim_bottleneck": 32,   # Linear Bottleneck dimension
    "dim_model": 512,       # Transformer hidden size
    "depth": 8,             # Number of layers
    "heads": 8,
    "mlp_dim": 2048,        # 4x dim_model
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 15,
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

class TransformerBlock(nn.Module):
    dim_model: int
    heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        # Pre-Norm Architecture (Standard for ViT)
        
        # Attention
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(num_heads=self.heads)(y)
        x = x + y
        
        # MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.dim_model)(y)
        x = x + y
        return x

# Helper for Sinusoidal Embeddings (Standard Transformer Trick)
def get_sinusoidal_embeddings(n_pos, dim):
    position = np.arange(n_pos)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    
    pe = np.zeros((n_pos, dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return jnp.array(pe)[None, :, :]  # [1, N, D]


class JustProteinTransformer(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x_noisy, t):
        cfg = self.config
        
        # 1. Patch Embedding
        # [B, H, W, 1] -> [B, N, D]
        x = LinearBottleneckPatchEmbed(
            patch_size=cfg['patch_size'],
            dim_bottleneck=cfg['dim_bottleneck'],
            dim_model=cfg['dim_model']
        )(x_noisy)
        
        # 2. Add Sinusoidal Positional Embeddings (Fixed)
        num_patches = x.shape[1]
        # Generate [1, N, D] sine waves
        pos_emb = get_sinusoidal_embeddings(num_patches, cfg['dim_model'])
        # Add to input
        x = x + jnp.array(pos_emb)
        
        # 3. Add Time Embeddings
        # [B, 1] -> [B, D]
        t_emb = nn.Dense(cfg['dim_model'])(t)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(cfg['dim_model'])(t_emb)
        # Broadcast: [B, 1, D]
        t_emb = jnp.expand_dims(t_emb, axis=1)
        x = x + t_emb
        
        # 4. Transformer Blocks
        for _ in range(cfg['depth']):
            x = TransformerBlock(
                dim_model=cfg['dim_model'], 
                heads=cfg['heads'], 
                mlp_dim=cfg['mlp_dim']
            )(x)
            
        # 5. Output Head (Predicts Clean Patches)
        x = nn.LayerNorm()(x)
        # Project back to raw patch size [B, N, patch_size^2 * C]
        x_pred_patches = nn.Dense(cfg['dim_raw'])(x)
        
        # 6. Reconstruct Image
        return self.unpatchify(x_pred_patches)

    def unpatchify(self, x_patches):
        """
        Reconstruct [B, H, W, C] from patches [B, N, D]
        """
        B, N, D = x_patches.shape
        p = self.config['patch_size']
        h = w = int(math.sqrt(N))
        C = 1 # Grayscale
        
        # Reshape to [B, h, w, p, p, C]
        x = x_patches.reshape(B, h, w, p, p, C)
        # Transpose to [B, h, p, w, p, C]
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # Flatten to [B, H, W, C]
        x = x.reshape(B, h * p, w * p, C)
        return x

# ==========================================
# 3. Data Generator (Numpy)
# ==========================================
def numpy_data_generator(batch_size, img_size):
    """
    Generates mock distance maps with ACTUAL structure:
    1. A dark diagonal (backbone).
    2. Random off-diagonal blobs (contacts).
    """
    while True:
        # Start with large noise
        data = []
        for _ in range(batch_size):
            # 1. Create the base grid
            x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
            
            # 2. Add the Main Diagonal (Protein Backbone)
            # Distance increases as you move away from diagonal |i - j|
            dist_from_diag = np.abs(x - y)
            # Sigmoid-like mask: 0 on diagonal, 1 far away
            structure = np.tanh(dist_from_diag / 10.0)
            
            # 3. Add random "interaction blobs" (Tertiary structure)
            # Create a few random spots that are "close" (value 0)
            num_blobs = np.random.randint(2, 5)
            for _ in range(num_blobs):
                cx, cy = np.random.randint(0, img_size, 2)
                # Symmetrize blob centers
                cy = cx 
                # Create a gaussian blob
                blob = np.exp(-((x - cx)**2 + (y - cy)**2) / 100.0)
                structure = structure - blob * 0.5
            
            # Clip to valid range [0, 1]
            structure = np.clip(structure, 0, 1)
            
            # 4. Normalize to [-1, 1] for the model
            structure = structure * 2.0 - 1.0
            
            # Add channel dim
            data.append(structure[:, :, None])
            
        yield np.array(data)

# ==========================================
# 4. Training State & Step
# ==========================================

def create_train_state(rng, config):
    model = JustProteinTransformer(config)
    # Init dummy input [1, H, W, 1] and time [1, 1]
    dummy_x = jnp.ones([1, config['img_size'], config['img_size'], 1])
    dummy_t = jnp.ones([1, 1])
    
    params = model.init(rng, dummy_x, dummy_t)['params']
    
    tx = optax.adamw(learning_rate=config['lr'])
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@jax.jit
def train_step(state, batch, rng):
    """
    Stable Flow Matching Step:
    Predict 'x' and minimize MSE(x_pred, x_clean).
    This avoids the 1/(1-t) singularity.
    """
    x_clean = batch
    B = x_clean.shape[0]
    
    rng, t_rng, n_rng = jax.random.split(rng, 3)
    
    # 1. Sample Time t ~ Uniform[0, 1]
    t = jax.random.uniform(t_rng, (B, 1))
    
    # 2. Sample Noise x0 ~ Normal(0, 1)
    x_noise = jax.random.normal(n_rng, x_clean.shape)
    
    # 3. Interpolate
    t_bc = t.reshape((B, 1, 1, 1))
    x_t = (1 - t_bc) * x_noise + t_bc * x_clean
    
    def loss_fn(params):
        # Model predicts Clean Image (x)
        x_pred = state.apply_fn({'params': params}, x_t, t)
        
        # STABLE LOSS: Simple MSE on the data prediction
        # We do NOT convert to velocity here to avoid dividing by (1-t)
        loss = jnp.mean((x_pred - x_clean) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Optional: Gradient Clipping (Adds extra stability)
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    state = state.apply_gradients(grads=grads)
    
    return state, loss, rng


# ==========================================
# 6. Inference (Sampling)
# ==========================================

def sample_protein(state, rng, img_size, steps=20):
    """
    Robust Sampler: Stops just before t=1 to avoid singularity.
    """
    print(f"Sampling with {steps} steps...")
    
    # 1. Start with Noise
    x_current = jax.random.normal(rng, (1, img_size, img_size, 1))
    
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
# 7. Visualization
# ==========================================
import matplotlib.pyplot as plt

def visualize_sample(x_sample):
    # Remove batch/channel dims -> [H, W]
    img = np.array(x_sample[0, :, :, 0])
    
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='viridis', origin='upper') # 'viridis' is good for distance
    plt.title("Generated Protein Distance Map")
    plt.colorbar(label="Normalized Distance")
    plt.show()


def visualize_protein_sample(x_sample, epoch=None):
    # Remove batch/channel dims -> [H, W]
    # x_sample is [-1, 1], map back to [0, 1] for plotting
    img = np.array(x_sample[0, :, :, 0])
    img = (img + 1.0) / 2.0
    
    plt.figure(figsize=(6,6))
    # 'RdBu_r' is great: Red = Contact (Close), Blue = Far
    # or 'viridis_r' (Yellow = Close, Purple = Far)
    plt.imshow(img, cmap='viridis_r', origin='upper', vmin=0, vmax=1)
    
    title = "Generated Protein"
    if epoch is not None: title += f" (Epoch {epoch})"
    plt.title(title)
    plt.colorbar(label="Normalized Distance")
    plt.show()


def reconstruct_and_check_physics_jax(dist_map_pred):
    """
    Input: dist_map_pred (JAX Array, Shape: [N, N], Unit: Angstroms)
    Returns: Average bond length deviation
    """
    # 1. Symmetrize (Crucial: Distance maps must be symmetric)
    # Average with transpose to remove any numerical asymmetry from generation
    D = (dist_map_pred + dist_map_pred.T) / 2.0
    N = D.shape[0]

    # 2. Classical MDS: Convert Distance Matrix -> 3D Coordinates
    # Formula: B = -0.5 * J * D^2 * J
    # J is the "Centering Matrix": I - 1/N * Ones
    
    # Create Centering Matrix J
    J = jnp.eye(N) - jnp.ones((N, N)) / N
    
    # Apply Double Centering
    # Note: We square D because MDS works on squared distances
    B = -0.5 * J @ (D**2) @ J
    
    # 3. Eigendecomposition
    # eigh is for Hermitian/Symmetric matrices (faster/stable)
    eigvals, eigvecs = jnp.linalg.eigh(B)
    
    # 4. Extract Top 3 Coordinates (X, Y, Z)
    # eigh returns eigenvalues in ascending order. We want the LAST 3 (largest).
    # We clip eigenvalues to be non-negative to avoid NaNs from small numerical noise
    top_vals = jnp.clip(eigvals[-3:], a_min=0.0)
    top_vecs = eigvecs[:, -3:]
    
    # Coords = Eigenvectors * Sqrt(Eigenvalues)
    # Shape: [N, 3]
    coords = top_vecs * jnp.sqrt(top_vals)
    
    # --- PHYSICS CHECK ---

    # 5. Calculate Bond Lengths
    # Distance between residue i and i+1
    # diffs = coords[i+1] - coords[i]
    diffs = coords[1:] - coords[:-1]
    
    # Euclidean norm of these vectors
    bond_lengths = jnp.linalg.norm(diffs, axis=1)
    
    # 6. Statistics (Move to CPU for printing/plotting)
    avg_bond = float(bond_lengths.mean())
    std_bond = float(bond_lengths.std())
    
    print(f"Reconstructed Bond Length: {avg_bond:.3f} Å +/- {std_bond:.3f}")
    
    # 7. Visualization
    # Histogram of bond lengths
    #plt.figure(figsize=(6, 4))
    #plt.hist(np.array(bond_lengths), bins=30, color='skyblue', edgecolor='black')
    #plt.axvline(x=3.8, color='red', linestyle='--', label='Target (3.8 Å)')
    #plt.title("Distribution of Backbone Bond Lengths")
    #plt.xlabel("Bond Length (Angstroms)")
    #plt.ylabel("Count")
    #plt.legend()
    #plt.show()
    
    # Optional: Scatter plot of the 3D backbone
    # Just to see if it looks like a protein or a hairball
    #fig = plt.figure(figsize=(6, 6))
    #ax = fig.add_subplot(111, projection='3d')
    #c = np.array(coords)
    #ax.plot(c[:, 0], c[:, 1], c[:, 2], '-o', markersize=2, alpha=0.7)
    #ax.set_title("Reconstructed 3D Backbone")
    #plt.show()
    
    return avg_bond

# ==========================================
# 5. Main Loop
# ==========================================
def main():
    print("Initializing Protein JiT on CATH...")
    
    # 1. Setup Data
    DATA_PATH = "./data/chain_set.jsonl"
    download_cath_dataset(DATA_PATH)

    # Debug check
    gen = cath_generator("./data/chain_set.jsonl", batch_size=1)
    sample = next(gen)
    print(f"Min: {sample.min()}, Max: {sample.max()}, Shape: {sample.shape}")
    # Should print: Min: -1.0, Max: 1.0, Shape: (1, 128, 128, 1)

    for i in range(5):
        batch = next(gen)
        if np.isnan(batch).any():
            print(f"Batch {i} FAILED: Contains NaNs")
        else:
            print(f"Batch {i} OK: Min={batch.min():.3f}, Max={batch.max():.3f}")
    
    # Create Generator
    train_gen = cath_generator(DATA_PATH, 
                               batch_size=CONFIG['batch_size'], 
                               img_size=CONFIG['img_size'])

    # 2. Setup Training
    rng = jax.random.PRNGKey(CONFIG['seed'])
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, CONFIG)
    
    # 3. Training Loop
    # Real dataset is large, so let's define "steps per epoch"
    steps_per_epoch = 1000 
    
    for epoch in range(CONFIG['epochs']):
        batch_losses = []
        
        for _ in range(steps_per_epoch):
            batch = next(train_gen)
            state, loss, rng = train_step(state, batch, rng)
            batch_losses.append(loss)
            sample = sample_protein(state, rng, img_size=CONFIG['img_size'], steps=100)
            sample_angstroms = (sample[0, :, :, 0] + 1.0) / 2.0 * 30.0
            reconstruct_and_check_physics_jax(sample_angstroms)
           
        avg_loss = np.mean(batch_losses)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.5f}")
        
        sample = sample_protein(state, rng, img_size=CONFIG['img_size'], steps=100)
        # 2. Denormalize to Angstroms [0, 30A]
        # Formula: (x + 1) / 2 * 30.0
        sample_angstroms = (sample[0, :, :, 0] + 1.0) / 2.0 * 30.0

        # 3. Run the Physics Check
        reconstruct_and_check_physics_jax(sample_angstroms)
        # NOW you can pass this to the physics check:
        visualize_protein_sample(sample, epoch)


if __name__ == "__main__":
    main()
