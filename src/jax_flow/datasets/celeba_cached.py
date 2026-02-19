"""CelebA dataset with disk caching for fast training.

Usage:
    # First time: cache the dataset
    python -m jax_flow.datasets.celeba_cached --cache --img-size 128

    # In training: use cached data
    from jax_flow.datasets.celeba_cached import make_dataloader, DataConfig
    it = make_dataloader("train", DataConfig(batch_size=128))
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    jnp = np


CACHE_DIR = Path.home() / ".cache" / "jax_flow" / "celeba"


@dataclass
class DataConfig:
    batch_size: int = 128
    num_epochs: Optional[int] = None
    shuffle: bool = True
    seed: int = 42
    drop_last: bool = True
    img_size: int = 128


def get_cache_path(img_size: int, split: str) -> Path:
    return CACHE_DIR / f"celeba_{img_size}x{img_size}_{split}.npy"


def is_cached(img_size: int, split: str = "train") -> bool:
    return get_cache_path(img_size, split).exists()


def cache_dataset(img_size: int = 128, max_samples: Optional[int] = None) -> None:
    """Download and cache CelebA dataset at specified resolution."""
    from datasets import load_dataset
    from PIL import Image

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation"]:
        print(f"Caching {split} split at {img_size}x{img_size}...", flush=True)
        ds = load_dataset("flwrlabs/celeba", split=split, streaming=False)

        images = []
        for i, sample in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            if i % 1000 == 0:
                print(f"  Processed {i} images...", flush=True)

            img = sample["image"]
            img = img.resize((img_size, img_size), Image.BILINEAR)
            arr = np.array(img).astype(np.float32) / 255.0

            if len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[:, :, :3]

            images.append(arr)

        data = np.stack(images, axis=0)
        data = data * 2.0 - 1.0

        cache_path = get_cache_path(img_size, split)
        np.save(cache_path, data)
        print(f"  Saved {len(images)} images to {cache_path}", flush=True)

    print("Caching complete!", flush=True)


def _load_cached_data(img_size: int, split: str) -> np.ndarray:
    cache_path = get_cache_path(img_size, split)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached dataset not found at {cache_path}. "
            f"Run: python -m jax_flow.datasets.celeba_cached --cache --img-size {img_size}"
        )
    return np.load(cache_path, mmap_mode="r")


def _batch_iterator(
    data: np.ndarray, cfg: DataConfig
) -> Iterator[Tuple[np.ndarray, None]]:
    n = data.shape[0]
    rng = np.random.default_rng(cfg.seed)

    epoch = 0
    while cfg.num_epochs is None or epoch < cfg.num_epochs:
        idx = np.arange(n)
        if cfg.shuffle:
            rng.shuffle(idx)

        for start in range(0, n, cfg.batch_size):
            end = start + cfg.batch_size
            if end > n:
                if cfg.drop_last:
                    break
                end = n
            bidx = idx[start:end]
            batch = np.array(data[bidx], dtype=np.float32)
            yield batch, None

        epoch += 1


def make_dataloader(
    split: str = "train", cfg: Optional[DataConfig] = None
) -> Iterator[Tuple[np.ndarray, None]]:
    """Create a fast dataloader from cached CelebA data.

    Args:
        split: "train" or "validation"
        cfg: DataConfig with batch_size, img_size, etc.

    Yields:
        (images, None) where images shape is [B, H, W, C] in range [-1, 1]
    """
    if cfg is None:
        cfg = DataConfig()

    split_map = {"test": "validation", "validation": "validation", "train": "train"}
    actual_split = split_map.get(split, split)

    data = _load_cached_data(cfg.img_size, actual_split)
    print(f"Loaded cached CelebA {actual_split}: {data.shape}", flush=True)

    return _batch_iterator(data, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CelebA caching utility")
    parser.add_argument("--cache", action="store_true", help="Cache the dataset")
    parser.add_argument("--img-size", type=int, default=128, help="Image size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for testing)")
    args = parser.parse_args()

    if args.cache:
        cache_dataset(args.img_size, args.max_samples)
    else:
        print(f"Cache status for {args.img_size}x{args.img_size}:")
        for split in ["train", "validation"]:
            cached = is_cached(args.img_size, split)
            print(f"  {split}: {'cached' if cached else 'not cached'}")
