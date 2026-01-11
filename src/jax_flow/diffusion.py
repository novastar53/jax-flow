import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from jax_flow.datasets.fashion_mnist import DataConfig, make_dataloader

cfg = DataConfig(batch_size=16, num_epochs=1, shuffle=True, as_chw=False, sample_size=64)
it = make_dataloader("test", cfg)
imgs, labs = next(it)


# Load and preprocess grayscale image
def load_image(path, resize=(64, 64)):
    img = Image.open(path).convert("L").resize(resize)
    img_np = np.asarray(img).astype(np.float32) / 255.0
    return jnp.array(img_np)

# Forward diffusion step (reparameterization form)
def q_xt_given_xt_minus_1(x_prev, beta_t, rng):
    noise = jax.random.normal(rng, shape=x_prev.shape)
    mean = jnp.sqrt(1 - beta_t) * x_prev
    std = jnp.sqrt(beta_t)
    return mean + std * noise

# Run T steps of forward diffusion
def forward_diffusion(x0, betas, key):
    xts = [x0]
    x_prev = x0
    for t in range(len(betas)):
        key, subkey = jax.random.split(key)
        x_next = q_xt_given_xt_minus_1(x_prev, betas[t], subkey)
        xts.append(x_next)
        x_prev = x_next
    return xts

# Plot multiple images (supports both single images and batches)
def show_images(images, titles=None, cols=5):
    if isinstance(images, (list, tuple)):
        imgs = [np.array(img) for img in images]
    else:
        np_imgs = np.asarray(images)
        if np_imgs.ndim == 2:
            imgs = [np_imgs]
        else:
            imgs = [np_imgs[i] for i in range(np_imgs.shape[0])]

    rows = (len(imgs) + cols - 1) // cols
    plt.figure(figsize=(cols * 2, rows * 2))
    for i, img in enumerate(imgs):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.clip(img, 0, 1), cmap='gray' if img.ndim == 2 else None)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    x0 = imgs.astype(jnp.float32) / 255.0  # Normalize full batch to [0,1]

    T = 10  # number of diffusion steps
    betas = jnp.linspace(1e-5, 0.001, T)
    rng = jax.random.PRNGKey(42)

    xts = forward_diffusion(x0, betas, rng)

    # Show selected steps
    selected = [0, 1, 2, 3, 5, 7, 9, T]
    for t in selected:
        show_images(
            xts[t],
            titles=[f"t={t}, idx={i}" for i in range(xts[t].shape[0])],
            cols=4,
        )
