"""CelebA dataloader that yields JAX arrays resized to 128x128.

Follows the API of `fashion_mnist.py`: prefer torchvision (PyTorch) backend, fall back to TFDS.
Returns batches as `jax.numpy` arrays ready for Flax models.

Example:
    cfg = DataConfig(batch_size=16, num_epochs=1, shuffle=True, as_chw=True)
    it = make_dataloader("train", cfg)
    imgs, labels = next(it)

Run demo:
    python -m jax_flow.datasets.celeb_a
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


@dataclass
class DataConfig:
    batch_size: int = 32
    num_epochs: Optional[int] = 1
    shuffle: bool = False
    seed: int = 0
    drop_last: bool = True
    # convert to channel-first (C,H,W) if True, otherwise keep HWC
    as_chw: bool = True
    # If set, limit the dataset to the first `sample_size` examples (useful for demos)
    sample_size: Optional[int] = None
    # Target image size (H, W)
    image_size: Tuple[int, int] = (56, 56)


def _load_from_tfds(sample_size: Optional[int] = None):
    try:
        import tensorflow_datasets as _tfds  # type: ignore
    except Exception:
        return None

    def ds_to_arrays(split_name: str):
        ds = _tfds.load('celeb_a', split=split_name, as_supervised=False)
        imgs = []
        attrs = []
        for i, item in enumerate(ds):
            img = item.get('image') if isinstance(item, dict) else item['image']
            # convert to numpy
            arr = np.array(img)
            imgs.append(arr)
            # attributes may or may not be present; keep placeholder
            attrs.append(0)
            if sample_size is not None and len(imgs) >= sample_size:
                break
        if not imgs:
            return None
        return np.stack(imgs, axis=0), np.array(attrs, dtype=np.int64)

    # TFDS provides 'train' split; allow 'train' and 'test' mapping
    try:
        train = ds_to_arrays('train')
        test = ds_to_arrays('test')
    except Exception:
        # Some TFDS versions may only provide a single split; return train only
        train = ds_to_arrays('train')
        test = None

    return (train, test)


def _load_from_torchvision(sample_size: Optional[int] = None, image_size=(128, 128)):
    try:
        from torchvision import datasets, transforms  # type: ignore
    except Exception:
        return None

    to_np = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    def ds_to_arrays(split: str):
        try:
            ds = datasets.CelebA(root='./data', split=split, download=True, transform=to_np)
        except Exception as e:
            # propagate the exception to caller handling
            raise
        imgs = []
        attrs = []
        for i, (img, target) in enumerate(ds):
            # img is a tensor C,H,W in [0,1]; convert to HWC float32
            arr = np.array(img.numpy()).transpose(1, 2, 0).astype(np.float32)
            imgs.append(arr)
            attrs.append(int(target) if isinstance(target, (int, np.integer)) else 0)
            if sample_size is not None and len(imgs) >= sample_size:
                break
        if not imgs:
            return None
        return np.stack(imgs, axis=0), np.array(attrs, dtype=np.int64)

    train = None
    test = None
    # Try several common split names; collect exceptions to include in debug if needed
    split_names = ['train', 'train_split', 'valid', 'test']
    for s in split_names:
        try:
            if train is None:
                train = ds_to_arrays(s)
        except Exception as e:
            # ignore and continue trying other split names
            print(e)
            train = train
    for s in ['test', 'valid', 'train']:
        try:
            if test is None:
                test = ds_to_arrays(s)
        except Exception as e:
            print(e)
            test = test

    # If neither split produced images, signal failure so caller can try other backends
    if train is None and test is None:
        return None
    return (train, test)


def _get_arrays(cfg: DataConfig):
    # Backend selection: default prefers torchvision, but can be forced with env var
    import os
    backend = os.environ.get('CELEBA_BACKEND', 'auto').lower()

    # If forced to torchvision, try only that and raise a helpful error if unavailable
    if backend == 'torchvision':
        res = _load_from_torchvision(sample_size=cfg.sample_size, image_size=cfg.image_size)
        if res is None:
            raise RuntimeError('torchvision backend requested via CELEBA_BACKEND but torchvision is not available or failed')
        return res

    # If forced to tfds, try only tfds (but catch download/checksum errors and raise informative message)
    if backend == 'tfds':
        try:
            res = _load_from_tfds(sample_size=cfg.sample_size)
        except Exception as e:
            raise RuntimeError(f'TFDS backend requested but failed: {e}')
        if res is None:
            raise RuntimeError('TFDS backend requested via CELEBA_BACKEND but tensorflow_datasets is not available or failed')
        return res

    # auto: prefer torchvision, otherwise try TFDS (with safe fallback if TFDS raises download errors)
    res = _load_from_torchvision(sample_size=cfg.sample_size, image_size=cfg.image_size)
    if res is not None:
        return res

    try:
        res = _load_from_tfds(sample_size=cfg.sample_size)
    except Exception as e:
        # TFDS could fail due to download/checksum issues (Google Drive redirects). Provide a helpful message.
        import warnings
        warnings.warn(f'TFDS backend failed with error: {e}. If you prefer torchvision, install it (`pip install torchvision`) and set CELEBA_BACKEND=torchvision')
        return None
    if res is not None:
        return res
    raise RuntimeError(
        "Could not load CelebA: please install tensorflow-datasets (`pip install tensorflow-datasets`) "
        "or torchvision (`pip install torchvision`)."
    )


def _resize_image_arr(arr: np.ndarray, size: Tuple[int, int]):
    # arr assumed HWC or HW; convert via PIL for resizing
    from PIL import Image

    if arr.ndim == 2:
        mode = 'L'
    elif arr.shape[2] == 3:
        mode = 'RGB'
    else:
        mode = 'RGBA' if arr.shape[2] == 4 else 'RGB'
    im = Image.fromarray((arr * 255).astype(np.uint8) if arr.dtype == np.float32 else arr.astype(np.uint8))
    # Pillow newer versions expose Resampling enum
    # Pillow >=9 exposes Image.Resampling; older versions use Image.BILINEAR
    resample = getattr(getattr(Image, 'Resampling', Image), 'BILINEAR', getattr(Image, 'BILINEAR', 1))
    im = im.resize(size, resample)
    out = np.array(im)
    # ensure HWC
    if out.ndim == 2:
        out = np.expand_dims(out, -1)
    return out.astype(np.float32) / 255.0


def _prepare_split(split: str, cfg: DataConfig):
    res = _get_arrays(cfg)
    if res is None:
        raise RuntimeError('No dataset backend available for CelebA (tried torchvision and tfds). Install torchvision or use CELEBA_BACKEND to force a backend.')

    train_tuple, test_tuple = res
    if train_tuple is None:
        raise RuntimeError('Dataset backend returned no training split for CelebA; check backend availability and permissions')
    x_train, y_train = train_tuple
    if test_tuple is not None:
        x_test, y_test = test_tuple
    else:
        x_test, y_test = None, None

    if split.startswith('train'):
        imgs, labs = x_train, y_train
    else:
        imgs, labs = (x_test, y_test) if x_test is not None else (x_train, y_train)

    if imgs is None:
        raise RuntimeError('Requested split not available')

    # Ensure labs is present; if not, create placeholder zeros
    if labs is None:
        labs = np.zeros((imgs.shape[0],), dtype=np.int64)

    # Optionally slice for demos
    if cfg.sample_size is not None:
        imgs = imgs[: cfg.sample_size]
        labs = labs[: cfg.sample_size]

    # Ensure HWC float32 in [0,1] and resize to target image_size
    prepared = []
    for im in imgs:
        arr = im.astype(np.float32)
        # If values in [0,1] already, keep; if 0-255 ints, normalize after resize
        if arr.max() > 1.5:
            # assume uint8
            arr = arr.astype(np.uint8)
            arr = _resize_image_arr(arr, cfg.image_size)
        else:
            # float in [0,1]
            arr = _resize_image_arr((arr * 255).astype(np.uint8), cfg.image_size)
        prepared.append(arr)
    imgs = np.stack(prepared, axis=0)

    return imgs, labs


def _batch_iterator(split: str, cfg: DataConfig) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    imgs, labs = _prepare_split(split, cfg)
    n = imgs.shape[0]
    rng = np.random.default_rng(cfg.seed)

    epoch = 0
    while cfg.num_epochs is None or epoch < (cfg.num_epochs or 0):
        idx = np.arange(n)
        if cfg.shuffle and split.startswith('train'):
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
                batch_imgs = batch_imgs.transpose(0, 3, 1, 2)

            batch_imgs_j = jnp.array(batch_imgs)
            batch_labs_j = jnp.array(batch_labs, dtype=jnp.int32)
            yield batch_imgs_j, batch_labs_j

        epoch += 1


def make_dataloader(split: str, cfg: Optional[DataConfig] = None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    if cfg is None:
        cfg = DataConfig()
    assert split in {"train", "test", "validation"} or split.startswith('train') or split.startswith('test')
    return _batch_iterator(split, cfg)


def visualize_batch(images: jnp.ndarray, labels: jnp.ndarray = None, max_display: int = 16):
    """Display a grid of images (compatible with Fashion-MNIST visualize_batch)."""
    imgs = np.array(images)
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3):
        imgs = imgs.transpose(0, 2, 3, 1)

    labs = np.array(labels) if labels is not None else np.arange(imgs.shape[0])

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
            ax.imshow(im, cmap='gray')
        else:
            ax.imshow(np.clip(im, 0.0, 1.0))
        ax.axis('off')
        ax.set_title(str(int(labs[i])), fontsize=8)

    for j in range(n, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    cfg = DataConfig(batch_size=8, num_epochs=1, shuffle=True, as_chw=False, sample_size=64)
    it = make_dataloader('train', cfg)
    imgs, labs = next(it)
    print('Batch shapes:', imgs.shape, labs.shape)
    visualize_batch(imgs, labs, max_display=16)
