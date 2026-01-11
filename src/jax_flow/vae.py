from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

import optax

from typing import Optional
#from jax_flow.datasets.fashion_mnist import make_dataloader, visualize_batch, FASHION_LABELS
from jax_flow.datasets.celeb_a import DataConfig, make_dataloader, visualize_batch

train_it = make_dataloader("train")

@dataclass
class Config:
    hidden_size: int = 16


class VAE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.conv1 = nnx.Conv(in_features=3, out_features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(in_features=16, out_features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv3 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', rngs=rngs)
        self.linear1 = nnx.Linear(14*14*64, 2 * config.hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(config.hidden_size, 14*14*64, rngs=rngs)
        self.deconv1 = nnx.ConvTranspose(in_features=64, out_features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv2 = nnx.ConvTranspose(in_features=32, out_features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv3 = nnx.ConvTranspose(in_features=16, out_features=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME', rngs=rngs)
    

    def __call__(self, batch, key: jnp.ndarray = None):
        B, _, _, _ = batch.shape
        batch = batch.transpose(0, 2, 3, 1)
        x = self.conv1(batch)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(B, -1)
        x = self.linear1(x)
        mu, log_var = jnp.split(x, 2, axis=1)
        key, subkey = jax.random.split(key)
        epsilon = jax.random.normal(subkey, log_var.shape)
        l = mu + jnp.sqrt(jnp.exp(log_var)) * epsilon
        x = self.linear2(l)
        x = x.reshape(B, 14, 14, 64) 
        x = self.deconv1(x)
        x = self.deconv2(x)
        y = self.deconv3(x)
        assert(batch.shape == y.shape)
        y = y.transpose(0, 3, 1, 2)
        return y, mu, log_var, key


# ---------------------------
# VAE-specific generation
# ---------------------------

def decoder_wrapper(_params_unused, z, _rng=None):
    # `z` may be a numpy or jax array; decode_z returns CHW images
    imgs = decode_z(m, z)
    # convert to HWC if needed
    arr = np.array(imgs)
    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        arr = arr.transpose(0, 2, 3, 1)
    return arr


def decode_z(vae: VAE, z):
    """Decode latent batch z (numpy/jax array) to images using VAE decoder layers.

    Returns images in CHW format matching the forward output.
    """
    x = vae.linear2(z)
    if hasattr(x, 'reshape'):
        B = x.shape[0]
        x = x.reshape(B, 14, 14, 64)
    x = vae.deconv1(x)
    x = vae.deconv2(x)
    y = vae.deconv3(x)
    y = y.transpose(0, 3, 1, 2)
    return y



@nnx.jit
def step_fn(m, x, key):
    def loss_fn(m, x, key):
        y, mu, log_var, key = m(x, key)
        assert(y.shape == x.shape)
        loss =  jnp.sum((y - x) ** 2) + 0.5 * jnp.sum(jnp.exp(log_var) + mu ** 2 - log_var) 
        loss /= y.shape[0]
        return loss, (y, key)

    return nnx.value_and_grad(loss_fn, has_aux=True)(m, x, key)


rngs = nnx.Rngs(default=0)
config = Config()
m = VAE(config, rngs)

tx = optax.adam(1e-3)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

key = jax.random.PRNGKey(42)
for i, (x, labels) in enumerate(train_it):
    (loss, (y, key)), grads = step_fn(m, x, key)
    optimizer.update(m, grads)
    print(i, loss)

from jax_flow.generate import plot_samples as generic_plot
generic_plot(jax.random.PRNGKey(2000), None, decoder_wrapper, n_row=4, latent_dim=config.hidden_size)