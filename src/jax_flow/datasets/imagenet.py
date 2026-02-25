"""ImageNet dataloader that yields JAX arrays resized to manageable dimensions.

Follows the API of `celeb_a.py`: prefer torchvision backend, with support for
resizing images to a manageable size for demonstrations. Returns batches as
`jax.numpy` arrays ready for Flax models.

Example:
    cfg = DataConfig(batch_size=16, num_epochs=1, shuffle=True, as_chw=True)
    it = make_dataloader("train", cfg)
    imgs, labels = next(it)

Run demo:
    python -m jax_flow.datasets.imagenet
"""

from __future__ import annotations

import os

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
    image_size: Tuple[int, int] = (128, 128)
    # Path to ImageNet root directory (should contain 'train' and 'val' subdirs)
    data_dir: str = "./data/imagenet"


def _load_from_torchvision(data_dir: str, sample_size: Optional[int] = None, image_size=(128, 128)):
    """Load ImageNet using torchvision's ImageNet dataset wrapper."""
    try:
        from torchvision import datasets, transforms  # type: ignore
    except Exception:
        return None

    to_np = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    def ds_to_arrays(split: str):
        split_dir = "train" if split.startswith("train") else "val"
        split_path = f"{data_dir}/{split_dir}"
        try:
            ds = datasets.ImageFolder(root=split_path, transform=to_np)
        except Exception as e:
            # ImageNet directory not found or not accessible
            raise RuntimeError(
                f"Could not load ImageNet from {split_path}. "
                f"Please ensure ImageNet is downloaded and available at {data_dir} "
                f"with 'train' and 'val' subdirectories."
            ) from e

        imgs = []
        labels = []
        for i, (img, target) in enumerate(ds):
            # img is a tensor C,H,W in [0,1]; convert to HWC float32
            arr = np.array(img.numpy()).transpose(1, 2, 0).astype(np.float32)
            imgs.append(arr)
            labels.append(int(target))
            if sample_size is not None and len(imgs) >= sample_size:
                break
        if not imgs:
            return None
        return np.stack(imgs, axis=0), np.array(labels, dtype=np.int64)

    try:
        train = ds_to_arrays("train")
    except Exception as e:
        print(f"Warning: Could not load train split: {e}")
        train = None

    try:
        test = ds_to_arrays("val")
    except Exception as e:
        print(f"Warning: Could not load val split: {e}")
        test = None

    # If neither split produced images, signal failure
    if train is None and test is None:
        return None
    return (train, test)


def _load_from_huggingface(data_dir: str, sample_size: Optional[int] = None, image_size=(128, 128)):
    """Load ImageNet using HuggingFace datasets.

    Uses the 'ILSVRC/imagenet-1k' dataset from HuggingFace.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None

    def process_dataset(ds, target_sample_size: Optional[int] = None):
        """Process a dataset iterable into arrays."""
        from PIL import Image
        import io

        imgs = []
        labels = []

        for item in ds:
            try:
                pil_img = item.get('image') if isinstance(item, dict) else None
                if pil_img is None:
                    continue

                # Handle different image formats
                if isinstance(pil_img, bytes):
                    pil_img = Image.open(io.BytesIO(pil_img))
                elif not isinstance(pil_img, Image.Image):
                    if isinstance(pil_img, dict) and 'bytes' in pil_img:
                        pil_img = Image.open(io.BytesIO(pil_img['bytes']))
                    else:
                        continue

                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                pil_img = pil_img.resize(image_size, Image.BILINEAR)
                arr = np.array(pil_img).astype(np.float32) / 255.0
                imgs.append(arr)

                # Try to get label, use 0 if not available
                label = item.get('label', 0) if isinstance(item, dict) else 0
                labels.append(int(label) if label is not None else 0)

                if target_sample_size is not None and len(imgs) >= target_sample_size:
                    break
            except Exception:
                continue

        if not imgs:
            return None
        return np.stack(imgs, axis=0), np.array(labels, dtype=np.int64)

    # Load ImageNet-1k from HuggingFace
    ds_train = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    ds_val = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    train = process_dataset(ds_train, sample_size)
    test = process_dataset(ds_val, sample_size)

    if train is not None or test is not None:
        print(f"Successfully loaded ImageNet from HF: train={train[0].shape if train else 0}, val={test[0].shape if test else 0}")
        return (train, test)

    return None


def _get_arrays(cfg: DataConfig):
    """Load ImageNet arrays using available backend."""
    import os
    backend = os.environ.get('IMAGENET_BACKEND', 'auto').lower()

    # Prefer torchvision for local ImageNet installations
    if backend in ('auto', 'torchvision'):
        res = _load_from_torchvision(
            data_dir=cfg.data_dir,
            sample_size=cfg.sample_size,
            image_size=cfg.image_size
        )
        if res is not None:
            return res

    # Fallback to HuggingFace datasets
    if backend in ('auto', 'huggingface'):
        res = _load_from_huggingface(
            data_dir=cfg.data_dir,
            sample_size=cfg.sample_size,
            image_size=cfg.image_size
        )
        if res is not None:
            return res

    raise RuntimeError(
        "Could not load ImageNet. Please either:\n"
        "1. Set IMAGENET_BACKEND=torchvision and ensure ImageNet is available at "
        f"{cfg.data_dir} with 'train' and 'val' subdirectories, or\n"
        "2. Set IMAGENET_BACKEND=huggingface and install datasets (`pip install datasets`)"
    )


def _prepare_split(split: str, cfg: DataConfig):
    res = _get_arrays(cfg)
    if res is None:
        raise RuntimeError('No dataset backend available for ImageNet.')

    train_tuple, test_tuple = res
    if train_tuple is None:
        raise RuntimeError('Dataset backend returned no training split for ImageNet.')
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
