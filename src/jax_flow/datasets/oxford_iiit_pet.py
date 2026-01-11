"""Oxford-IIIT Pet dataloader yielding JAX batches and simple visualization helpers.

The loader mirrors the Fashion-MNIST and CelebA helpers: prefer torchvision if
available, fall back to tensorflow-datasets. Images are resized to a configurable
resolution and normalized to float32 in [0, 1].

Example:
    cfg = DataConfig(batch_size=8, num_epochs=1, shuffle=True, image_size=(224, 224))
    it = make_dataloader("train", cfg)
    imgs, labels, masks = next(it)

Run demo:
    python -m jax_flow.datasets.oxford_iiit_pet
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
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
    # Target image size (H, W)
    image_size: Tuple[int, int] = (224, 224)
    # Whether to include segmentation masks in the batches
    include_masks: bool = True


# Cache datasets keyed by target image size so multiple iterators reuse loaded arrays
DatasetSplit = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
_DATA_CACHE: dict[Tuple[int, int], Tuple[DatasetSplit, Optional[DatasetSplit]]] = {}


def _resize_image_arr(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize HWC or HW uint8 array to (H, W) using bilinear interpolation."""
    from PIL import Image

    if arr.ndim == 2:
        mode = "L"
        pil_in = Image.fromarray(arr, mode=mode)
    else:
        channels = arr.shape[2]
        if channels == 1:
            mode = "L"
            pil_in = Image.fromarray(arr.squeeze(-1), mode=mode)
        elif channels == 3:
            mode = "RGB"
            pil_in = Image.fromarray(arr, mode=mode)
        elif channels == 4:
            mode = "RGBA"
            pil_in = Image.fromarray(arr, mode=mode)
        else:
            # Fallback: let Pillow infer
            pil_in = Image.fromarray(arr)

    h, w = size
    resample_enum = getattr(getattr(Image, "Resampling", Image), "BILINEAR", getattr(Image, "BILINEAR", 1))
    pil_out = pil_in.resize((w, h), resample=resample_enum)
    out = np.array(pil_out)
    if out.ndim == 2:
        out = np.expand_dims(out, -1)
    return out


def _resize_mask_arr(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize HW uint8/int mask to (H, W) using nearest-neighbor to preserve labels."""
    from PIL import Image

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    pil_in = Image.fromarray(arr.astype(np.uint8), mode="L")
    h, w = size
    resample_enum = getattr(getattr(Image, "Resampling", Image), "NEAREST", getattr(Image, "NEAREST", 0))
    pil_out = pil_in.resize((w, h), resample=resample_enum)
    return np.array(pil_out, dtype=np.uint8)


def _prepare_image(arr: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert arbitrary image array to float32 HWC in [0,1] at the target size."""
    np_arr = np.array(arr)
    if np_arr.ndim == 2:
        np_arr = np.expand_dims(np_arr, -1)

    if np_arr.dtype == np.uint8:
        arr_uint8 = np_arr
    else:
        if np_arr.max() <= 1.01:
            arr_uint8 = (np.clip(np_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr_uint8 = np.clip(np_arr, 0.0, 255.0).astype(np.uint8)

    resized = _resize_image_arr(arr_uint8, image_size)
    return resized.astype(np.float32) / 255.0


def _prepare_mask(arr: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert mask array to zero-based int32 HW at the target size."""
    np_arr = np.array(arr)
    if np_arr.ndim == 3 and np_arr.shape[-1] == 1:
        np_arr = np_arr.squeeze(-1)
    if np_arr.dtype != np.uint8:
        np_arr = np.clip(np_arr, 0, 255).astype(np.uint8)
    resized = _resize_mask_arr(np_arr, image_size)
    mask = resized.astype(np.int32)
    # Dataset encodes classes starting at 1; shift to zero-based indices for Optax
    mask = np.where(mask > 0, mask - 1, 0)
    return mask


def _load_from_torchvision(image_size: Tuple[int, int]):
    try:
        from torchvision import datasets  # type: ignore
    except Exception:
        return None

    def ds_to_arrays(split: str):
        try:
            ds = datasets.OxfordIIITPet(
                root="./data",
                split=split,
                target_types=("category", "segmentation"),
                download=True,
            )
        except Exception:
            return None

        imgs = []
        labels = []
        masks = []
        for img, target in ds:
            if isinstance(target, tuple):
                label, seg_mask = target
            else:
                label, seg_mask = target, None
            # torchvision returns PIL images; ensure RGB for consistency
            arr = _prepare_image(img.convert("RGB"), image_size)
            imgs.append(arr)
            labels.append(int(label))
            if seg_mask is not None:
                masks.append(_prepare_mask(seg_mask, image_size))
            else:
                masks.append(np.zeros(image_size, dtype=np.int32))
        if not imgs:
            return None
        masks_arr = np.stack(masks, axis=0) if masks else None
        return np.stack(imgs, axis=0), np.array(labels, dtype=np.int64), masks_arr

    train = ds_to_arrays("trainval")
    if train is None:
        return None
    test = ds_to_arrays("test")
    return (train, test)


def _load_from_tfds(image_size: Tuple[int, int]):
    try:
        import tensorflow_datasets as tfds  # type: ignore
    except Exception:
        return None

    def ds_to_arrays(split: str):
        try:
            ds = tfds.load("oxford_iiit_pet", split=split, as_supervised=False)
        except Exception:
            return None
        imgs = []
        labels = []
        masks = []
        try:
            iterable = tfds.as_numpy(ds)
        except AttributeError:
            iterable = ds
        for example in iterable:
            # tfds examples can be dict-like objects or dicts
            img = example["image"] if isinstance(example, dict) else example.image
            label = example["label"] if isinstance(example, dict) else example.label
            seg_mask = (
                example["segmentation_mask"]
                if isinstance(example, dict)
                else getattr(example, "segmentation_mask", None)
            )
            arr = _prepare_image(img, image_size)
            imgs.append(arr)
            labels.append(int(label))
            if seg_mask is not None:
                masks.append(_prepare_mask(seg_mask, image_size))
            else:
                masks.append(np.zeros(image_size, dtype=np.int32))
        if not imgs:
            return None
        masks_arr = np.stack(masks, axis=0) if masks else None
        return np.stack(imgs, axis=0), np.array(labels, dtype=np.int64), masks_arr

    train = ds_to_arrays("train")
    if train is None:
        return None
    test = ds_to_arrays("test")
    return (train, test)


def _load_dataset(image_size: Tuple[int, int]):
    backend = os.environ.get("OXFORD_PET_BACKEND", "auto").lower()

    if backend == "torchvision":
        res = _load_from_torchvision(image_size)
        if res is None:
            raise RuntimeError(
                "torchvision backend requested via OXFORD_PET_BACKEND but torchvision "
                "is not available or failed to load Oxford-IIIT Pet."
            )
        return res

    if backend == "tfds":
        res = _load_from_tfds(image_size)
        if res is None:
            raise RuntimeError(
                "TFDS backend requested via OXFORD_PET_BACKEND but tensorflow-datasets "
                "is not available or failed to load Oxford-IIIT Pet."
            )
        return res

    res = _load_from_torchvision(image_size)
    if res is not None:
        return res

    res = _load_from_tfds(image_size)
    if res is not None:
        return res

    raise RuntimeError(
        "Could not load Oxford-IIIT Pet dataset: install torchvision (`pip install torchvision`) "
        "or tensorflow-datasets (`pip install tensorflow-datasets`)."
    )


def _get_arrays(cfg: DataConfig):
    key = cfg.image_size
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = _load_dataset(cfg.image_size)
    return _DATA_CACHE[key]


def _prepare_split(split: str, cfg: DataConfig):
    train_tuple, test_tuple = _get_arrays(cfg)
    x_train, y_train, m_train = train_tuple
    if test_tuple is not None:
        x_test, y_test, m_test = test_tuple
    else:
        x_test, y_test, m_test = None, None, None

    lsplit = split.lower()
    if lsplit.startswith("train"):
        imgs, labs, masks = x_train, y_train, m_train
    elif lsplit.startswith("test") or lsplit.startswith("val"):
        if x_test is not None:
            imgs, labs, masks = x_test, y_test, m_test
        else:
            imgs, labs, masks = x_train, y_train, m_train
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train', 'test', or 'validation'.")

    if cfg.sample_size is not None:
        imgs = imgs[: cfg.sample_size]
        labs = labs[: cfg.sample_size]
        if masks is not None:
            masks = masks[: cfg.sample_size]

    return imgs, labs, masks


def _batch_iterator(split: str, cfg: DataConfig) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]]:
    imgs, labs, masks = _prepare_split(split, cfg)
    n = imgs.shape[0]
    rng = np.random.default_rng(cfg.seed)
    include_masks = cfg.include_masks and masks is not None

    epoch = 0
    while cfg.num_epochs is None or epoch < (cfg.num_epochs or 0):
        idx = np.arange(n)
        if cfg.shuffle and split.lower().startswith("train"):
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
            batch_masks_j = None

            if cfg.as_chw:
                batch_imgs = batch_imgs.transpose(0, 3, 1, 2)

            batch_imgs_j = jnp.array(batch_imgs)
            batch_labs_j = jnp.array(batch_labs, dtype=jnp.int32)
            if include_masks:
                batch_masks = masks[bidx]
                batch_masks_j = jnp.array(batch_masks, dtype=jnp.int32)
            yield batch_imgs_j, batch_labs_j, batch_masks_j

        epoch += 1


def make_dataloader(split: str, cfg: Optional[DataConfig] = None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]]:
    """Return an iterator over batches for `split` in {"train", "test", "validation"}.

    Yields `(images, labels, masks)` where `masks` is `None` when segmentation masks
    are disabled or unavailable.
    """
    if cfg is None:
        cfg = DataConfig()
    assert split in {"train", "test", "validation"} or split.startswith("train") or split.startswith("test")
    return _batch_iterator(split, cfg)


def visualize_batch(
    images: jnp.ndarray,
    labels: Optional[jnp.ndarray] = None,
    masks: Optional[jnp.ndarray] = None,
    class_names: Optional[dict[int, str]] = None,
    max_display: int = 16,
    mask_alpha: float = 0.4,
    mask_cmap: str = "magma",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Display a grid of images with optional class labels and segmentation overlays.

    If ``save_path`` is provided the figure is written to disk; set ``show`` to False in
    headless environments.
    """
    imgs = np.array(images)
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3):
        imgs = imgs.transpose(0, 2, 3, 1)

    if labels is None:
        labs = np.arange(imgs.shape[0])
    else:
        labs = np.array(labels)

    mask_arr = None
    if masks is not None:
        mask_arr = np.array(masks)
        if mask_arr.ndim == 4 and mask_arr.shape[1] == 1:
            mask_arr = mask_arr[:, 0, :, :]
        if mask_arr.ndim == 4 and mask_arr.shape[-1] == 1:
            mask_arr = mask_arr.squeeze(-1)

    save_path_obj = Path(save_path) if save_path is not None else None

    n = min(imgs.shape[0], max_display)
    if n == 0:
        return
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        ax = axes[i]
        im = imgs[i]
        if im.shape[-1] == 1:
            ax.imshow(im.squeeze(-1), cmap="gray")
        else:
            ax.imshow(np.clip(im, 0.0, 1.0))
        if mask_arr is not None:
            mask_img = mask_arr[i]
            ax.imshow(mask_img, cmap=mask_cmap, alpha=mask_alpha, interpolation="nearest")
            unique_vals = np.unique(mask_img)
            unique_vals = unique_vals[unique_vals > 0]
        else:
            unique_vals = None
        ax.axis("off")
        lab = int(labs[i])
        title = class_names.get(lab, str(lab)) if class_names is not None else str(lab)
        if unique_vals is not None and unique_vals.size > 0:
            vals_str = ",".join(map(str, unique_vals[:4]))
            if unique_vals.size > 4:
                vals_str += ",..."
            title = f"{title} | mask:{vals_str}"
        ax.set_title(title, fontsize=8)

    for j in range(n, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path_obj is not None:
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_obj, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    cfg = DataConfig(batch_size=8, num_epochs=1, shuffle=True, as_chw=False, sample_size=32)
    iterator = make_dataloader("train", cfg)
    images, labels, masks = next(iterator)
    mask_shape = None if masks is None else masks.shape
    print("Batch shapes:", images.shape, labels.shape, mask_shape)
    visualize_batch(images, labels, masks, max_display=16)
