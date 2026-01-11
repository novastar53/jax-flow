from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

import optax

from jax_flow.datasets.celeb_a import DataConfig, make_dataloader, visualize_batch

train_it = make_dataloader("train")

@dataclass
class Config:
    hidden_size: int = 16


class AutoEncoder(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.conv1 = nnx.Conv(in_features=3, out_features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(in_features=16, out_features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv3 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', rngs=rngs)
        self.linear1 = nnx.Linear(14*14*64, config.hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(config.hidden_size, 14*14*64, rngs=rngs)
        self.deconv1 = nnx.ConvTranspose(in_features=64, out_features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv2 = nnx.ConvTranspose(in_features=32, out_features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv3 = nnx.ConvTranspose(in_features=16, out_features=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME', rngs=rngs)

    def encode(self, batch):
        B, _, _, _ = batch.shape
        x = batch.transpose(0, 2, 3, 1)
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape(B, -1)
        z = self.linear1(x)
        return z

    def decode(self, z):
        B = z.shape[0]
        x = self.linear2(z)
        x = x.reshape(B, 14, 14, 64)
        x = nnx.relu(self.deconv1(x))
        x = nnx.relu(self.deconv2(x))
        y = self.deconv3(x)
        y = y.transpose(0, 3, 1, 2)
        return y

    def __call__(self, batch):
        z = self.encode(batch)
        y = self.decode(z)
        return y, z


def decoder_wrapper(_params_unused, z, _rng=None):
    imgs = m.decode(jnp.array(z))
    arr = np.array(imgs)
    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        arr = arr.transpose(0, 2, 3, 1)
    return arr


def reconstruct_images(model, n_images=8, save_path=None):
    """Reconstruct real images from the dataset and display original vs reconstructed."""
    import matplotlib.pyplot as plt

    test_cfg = DataConfig(batch_size=n_images, num_epochs=1, shuffle=True, as_chw=True)
    test_it = make_dataloader("test", test_cfg)
    images, _ = next(test_it)

    reconstructed, _ = model(images)

    # Convert to HWC for plotting
    orig = np.array(images)
    recon = np.array(reconstructed)
    if orig.shape[1] in (1, 3):
        orig = orig.transpose(0, 2, 3, 1)
        recon = recon.transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
    for i in range(n_images):
        axes[0, i].imshow(np.clip(orig[i], 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        axes[1, i].imshow(np.clip(recon[i], 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved reconstruction to {save_path}")
    plt.show()


@nnx.jit
def step_fn(m, optimizer, x):
    def loss_fn(m, x):
        y, z = m(x)
        loss = jnp.mean((y - x) ** 2)
        return loss, y

    (loss, y), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m, x)
    optimizer.update(m, grads)
    return loss, y


if __name__ == "__main__":
    rngs = nnx.Rngs(default=0)
    config = Config()
    m = AutoEncoder(config, rngs)

    tx = optax.adam(1e-3)
    optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

    for i, (x, labels) in enumerate(train_it):
        loss, y = step_fn(m, optimizer, x)
        print(i, loss)

    # Reconstruct real images to show autoencoder works for reconstruction
    reconstruct_images(m, n_images=8, save_path="ae_reconstruction.png")

    # Generate from random latents (will be poor quality for vanilla AE)
    from jax_flow.generate import plot_samples as generic_plot
    generic_plot(jax.random.PRNGKey(2000), None, decoder_wrapper, n_row=4, latent_dim=config.hidden_size)
