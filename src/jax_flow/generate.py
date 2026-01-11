"""Sampling helpers for VAE models.

This module provides a small, backend-agnostic generator that draws samples
from a standard normal prior in latent space and decodes them using a
user-provided decoder function. The decoder function should accept a params
structure and a batch of latents and return decoded images (numpy/jax arrays).

Example decoder wrappers:

# If your decoder is a Flax Module `decoder` and you have `params_decoder`:
def flax_decoder_apply(params_decoder, z, rng=None):
    # If your model expects rngs, pass it as `rngs={'dropout': rng}` or adapt
    if rng is None:
        return decoder.apply(params_decoder, z)
    return decoder.apply(params_decoder, z, rngs={"dropout": rng})

# Then generate images:
from jax import random
key = random.PRNGKey(0)
for imgs in sample_and_decode(key, params_decoder, flax_decoder_apply, num_samples=64, batch_size=16, latent_dim=128):
    # `imgs` is a batch of decoded images
    ...

"""

from typing import Callable, Iterator, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:  # allow import even if jax missing for docs/typing
    jax = None
    jnp = np


def _sample_latents(rng, num: int, latent_dim: int):
    """Draw `num` latent vectors from N(0, I).

    rng: a JAX PRNGKey (or any object with `random.normal` behavior if using numpy)
    Returns an array of shape [num, latent_dim].
    """
    # If JAX is unavailable, or the caller passed rng=None, use numpy RNG
    if jax is None or rng is None:
        return np.random.normal(size=(num, latent_dim)).astype(np.float32)
    return jax.random.normal(rng, (num, latent_dim), dtype=jnp.float32)


def sample_and_decode(
    rng,
    params_decoder,
    decoder_fn: Callable[..., object],
    num_samples: int,
    batch_size: int,
    latent_dim: int,
    decode_rngs: bool = False,
) -> Iterator[object]:
    """Generator that yields decoded image batches.

    Args:
        rng: a JAX PRNGKey. If JAX is not available the function will use numpy RNG.
        params_decoder: parameters (or state) for the decoder_fn.
        decoder_fn: Callable(params_decoder, z_batch, rng_opt) -> decoded_images.
            The third argument is optional and will receive an RNG when `decode_rngs=True`.
        num_samples: total number of samples to generate.
        batch_size: number of samples to produce per yielded batch.
        latent_dim: dimensionality of the latent vector.
        decode_rngs: if True, the generator will pass a unique rng to decoder_fn for each batch.

    Yields:
        Decoded image batches as returned by `decoder_fn`.
    """
    # If JAX is unavailable, or the caller passed rng=None, fall back to numpy RNG
    if jax is None or rng is None:
        # numpy path
        total = num_samples
        i = 0
        while i < total:
            n = min(batch_size, total - i)
            z = _sample_latents(None, n, latent_dim)
            imgs = decoder_fn(params_decoder, z, None)
            yield imgs
            i += n
        return

    # JAX path: manage keys
    key = rng
    generated = 0
    while generated < num_samples:
        n = min(batch_size, num_samples - generated)
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (n, latent_dim), dtype=jnp.float32)
        if decode_rngs:
            key, dek = jax.random.split(key)
            imgs = decoder_fn(params_decoder, z, dek)
        else:
            imgs = decoder_fn(params_decoder, z)
        yield imgs
        generated += n


def sample_grid(rng, params_decoder, decoder_fn: Callable[..., object], n_row: int, latent_dim: int, decode_rngs: bool = False):
    """Convenience: produce an n_row x n_row grid of samples and return a single image.

    Returns a numpy array shaped (H, W, C) suitable for plotting.
    """
    total = n_row * n_row
    batches = list(sample_and_decode(rng, params_decoder, decoder_fn, total, batch_size=total, latent_dim=latent_dim, decode_rngs=decode_rngs))
    if not batches:
        return None
    imgs = batches[0]
    # expect imgs shape (total, H, W, C) or (total, C, H, W)
    arr = np.array(imgs)
    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        # channel-first -> to HWC
        arr = arr.transpose(0, 2, 3, 1)
    # tile into grid
    H, W = arr.shape[1], arr.shape[2]
    grid = np.zeros((n_row * H, n_row * W, arr.shape[3]), dtype=arr.dtype)
    k = 0
    for r in range(n_row):
        for c in range(n_row):
            grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = arr[k]
            k += 1
    return grid


def plot_samples(rng, params_decoder, decoder_fn, n_row: int, latent_dim: int, decode_rngs: bool = False, save_to: Optional[str] = None):
    """Generate an n_row x n_row grid using sample_grid and plot/save it.

    Args mirror `sample_grid` and accepts `save_to` to write the image to disk.
    """
    import matplotlib.pyplot as plt

    grid = sample_grid(rng, params_decoder, decoder_fn, n_row, latent_dim, decode_rngs=decode_rngs)
    if grid is None:
        raise RuntimeError("No samples generated")

    plt.figure(figsize=(n_row * 2, n_row * 2))
    if grid.shape[2] == 1:
        plt.imshow(grid.squeeze(-1), cmap="gray")
    else:
        plt.imshow(np.clip(grid, 0.0, 1.0))
    plt.axis("off")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    plt.show()


def sample_from_model(model_or_params, decoder_fn: Callable[..., object], rng, num_samples: int, batch_size: int, latent_dim: int):
    """Backward-compatible helper similar to the old `sample_from_vae`.

    Accepts either a model instance (which `decoder_fn` will close over) or
    a params structure as `model_or_params`. The `decoder_fn` should accept
    (model_or_params, z, rng_opt) and return decoded images.
    """
    # Reuse sample_and_decode semantics: pass model_or_params as params_decoder
    for imgs in sample_and_decode(rng, model_or_params, decoder_fn, num_samples=num_samples, batch_size=batch_size, latent_dim=latent_dim):
        yield imgs


