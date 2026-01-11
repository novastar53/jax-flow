"""Fashion-MNIST dataloader that yields JAX arrays and a small demo visualizer.

The loader prefers TensorFlow's built-in dataset if available, otherwise tries
torchvision. It returns batches as `jax.numpy` arrays ready for Flax models.

Example:
    cfg = DataConfig(batch_size=16, num_epochs=1, shuffle=True, as_chw=True)
    it = make_dataloader("train", cfg)
    imgs, labels = next(it)

Run demo:
    python -m jax_flow.datasets.fashion_mnist
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


@dataclass
class DataConfig:
    batch_size: int = 32
    num_epochs: Optional[int] = 1
    shuffle: bool = True
    seed: int = 0
    drop_last: bool = True
    # convert to channel-first (C,H,W) if True, otherwise keep HWC
    as_chw: bool = True
    # If set, limit the dataset to the first `sample_size` examples (useful for demos)
    sample_size: Optional[int] = None


def _load_from_tf():
    try:
        import tensorflow as _tf  # type: ignore
    except Exception:
        return None
    (x_train, y_train), (x_test, y_test) = _tf.keras.datasets.fashion_mnist.load_data()
    # expand channel dim
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def _load_from_torchvision():
    try:
        from torchvision import datasets, transforms  # type: ignore
    except Exception:
        return None
    # torchvision will download to ~/.cache if needed
    to_np = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=to_np)
    test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=to_np)
    def ds_to_arrays(ds):
        imgs = []
        labels = []
        for img, lab in ds:
            imgs.append(np.array(img * 255.0, dtype=np.uint8).transpose(1, 2, 0))
            labels.append(int(lab))
        return np.stack(imgs, axis=0), np.array(labels, dtype=np.int64)
    return ds_to_arrays(train), ds_to_arrays(test)


def _get_arrays():
    # Try TF first (commonly available); then torchvision
    res = _load_from_tf()
    if res is not None:
        return res
    res = _load_from_torchvision()
    if res is not None:
        return res
    raise RuntimeError(
        "Could not load Fashion-MNIST: please install tensorflow (`pip install tensorflow`) "
        "or torchvision (`pip install torchvision`)."
    )


def _prepare_split(split: str, cfg: DataConfig):
    ((x_train, y_train), (x_test, y_test)) = _get_arrays()
    if split.startswith("train"):
        imgs, labs = x_train, y_train
    else:
        imgs, labs = x_test, y_test

    # Optionally slice for small demos
    if cfg.sample_size is not None:
        imgs = imgs[: cfg.sample_size]
        labs = labs[: cfg.sample_size]

    # Normalize to float32 in [0,1]
    imgs = imgs.astype(np.float32) / 255.0

    return imgs, labs


def _batch_iterator(split: str, cfg: DataConfig) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    imgs, labs = _prepare_split(split, cfg)
    n = imgs.shape[0]
    rng = np.random.default_rng(cfg.seed)

    epoch = 0
    while cfg.num_epochs is None or epoch < (cfg.num_epochs or 0):
        idx = np.arange(n)
        if cfg.shuffle and split.startswith("train"):
            rng.shuffle(idx)

        for start in range(0, n, cfg.batch_size):
            end = start + cfg.batch_size
            if end > n:
                if cfg.drop_last:
                    break
                end = n
            bidx = idx[start:end]
            batch_imgs = imgs[bidx]
            batch_labs = labs[bidx]

            if cfg.as_chw:
                # HWC -> CHW
                batch_imgs = batch_imgs.transpose(0, 3, 1, 2)

            # Convert to jax arrays
            batch_imgs_j = jnp.array(batch_imgs)
            batch_labs_j = jnp.array(batch_labs, dtype=jnp.int32)
            yield batch_imgs_j, batch_labs_j

        epoch += 1


def make_dataloader(split: str, cfg: Optional[DataConfig] = None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Return an iterator over batches for `split` in {"train", "test"}.

    The iterator yields (images, labels) where images are jax arrays of shape
    [B, C, H, W] if cfg.as_chw else [B, H, W, C].
    """
    if cfg is None:
        cfg = DataConfig()
    assert split in {"train", "test", "validation"} or split.startswith("train") or split.startswith("test")
    return _batch_iterator(split, cfg)


def visualize_batch(images: jnp.ndarray, labels: jnp.ndarray, class_names: Optional[dict] = None, max_display: int = 16):
    """Display a grid of images with labels.

    images: [B, C, H, W] or [B, H, W, C]; function will detect format.
    """
    imgs = np.array(images)
    labs = np.array(labels)
    # convert to HWC for plt
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3):
        # CHW
        imgs = imgs.transpose(0, 2, 3, 1)

    B = imgs.shape[0]
    n = min(B, max_display)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        im = imgs[i]
        if im.shape[-1] == 1:
            im = im.squeeze(-1)
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(np.clip(im, 0.0, 1.0))
        ax.axis("off")
        lab = int(labs[i])
        title = str(lab) if class_names is None else class_names.get(lab, str(lab))
        ax.set_title(title, fontsize=8)

    for j in range(n, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


FASHION_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


if __name__ == "__main__":
    # Quick demo: visualize a small sample without downloading the whole dataset
    cfg = DataConfig(batch_size=16, num_epochs=1, shuffle=True, as_chw=False, sample_size=64)
    it = make_dataloader("test", cfg)
    imgs, labs = next(it)
    print("Batch shapes:", imgs.shape, labs.shape)
    visualize_batch(imgs, labs, class_names=FASHION_LABELS, max_display=16)
