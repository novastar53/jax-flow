from functools import partial

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax


from jax_flow.datasets.oxford_iiit_pet import DataConfig, make_dataloader, visualize_batch
from jax_flow.modules.pos_emb import PosEmbedding


class ConvBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, *, rngs):
        self.time_mlp = nnx.Linear(in_channels, out_channels * 2, rngs=rngs)
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.gn1 = nnx.GroupNorm(out_channels, num_groups=8, rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.gn2 = nnx.GroupNorm(out_channels, num_groups=8, rngs=rngs)

    def __call__(self, x, t):
        t = nnx.relu(self.time_mlp(t))
        scale, shift = jnp.split(t, 2, axis=1)
        x = self.conv1(x)
        x = self.gn1(x)
        x = x * (scale + 1) + shift
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = nnx.relu(x)
        return x


class Down(nnx.Module):
    def __init__(self, in_channels, out_channels, *, rngs):
        self.down = nnx.Conv(in_channels, in_channels, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.block = ConvBlock(in_channels, out_channels, rngs=rngs)
    
    def __call__(self, x, t):
        #x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = self.down(x)
        x = self.block(x, t)
        return x


class Up(nnx.Module):
    def __init__(self, in_channels, skip_channels, out_channels, *, rngs):
        self.up = nnx.ConvTranspose(in_channels, in_channels, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.block = ConvBlock(in_channels+skip_channels, out_channels, rngs=rngs)
    
    def __call__(self, x, skip, t):
        #x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), method="linear")
        x = self.up(x)
        skip = jax.image.resize(skip, shape=(x.shape[0], x.shape[1], x.shape[2], skip.shape[3]), method="linear")
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.block(x, t)
        return x


class UNet(nnx.Module):
    def __init__(self, in_channels, num_classes, *, rngs):
        self.time_mlp = nnx.Sequential(
            PosEmbedding(64),
            nnx.Linear(64, 64*4, rngs=rngs),
            partial(nnx.gelu, approximate=True),
            nnx.Linear(64*4, 64*4, rngs=rngs)
        )
        self.enc1 = Down(in_channels, 64, rngs=rngs)
        self.enc2 = Down(64, 128, rngs=rngs) 
        self.enc3 = Down(128, 256, rngs=rngs)
        self.enc4 = Down(256, 512, rngs=rngs)

        self.middle = ConvBlock(512, 1024, rngs=rngs)

        self.up1 = Up(1024, 512, 512, rngs=rngs)
        self.up2 = Up(512, 256, 256, rngs=rngs)
        self.up3 = Up(256, 128, 128, rngs=rngs)
        self.up4 = Up(128, 64, 64, rngs=rngs)

        self.out_conv = nnx.Conv(64, num_classes, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x, time=None):
        t = None
        if time:
            t = self.time_mlp(time)
        e1 = self.enc1(x, t) # 1, 112, 112, 64
        e2 = self.enc2(e1, t) # 1, 56, 56, 128
        e3 = self.enc3(e2, t) # 1, 28, 28, 256
        e4 = self.enc4(e3, t) # 1, 14, 14, 512
        
        mid = self.middle(e4) # 1, 14, 14, 1024

        d1 = self.up1(mid, e4, t) # 1, 28, 28, 256
        d2 = self.up2(d1, e3, t) # 1, 56, 56, 128
        d3 = self.up3(d2, e2, t) # 1, 112, 112, 64
        d4 = self.up4(d3, e1, t) # 1, 224, 224, 64

        logits = self.out_conv(d4) # 1, 224, 224, 3
        return logits


if __name__ == "__main__":
    rngs = nnx.Rngs(default=0)
    model = UNet(in_channels=3, num_classes=3, rngs=rngs)
    tx = optax.adam(1e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def step_fn(model, optimizer, inputs, labels):
        def loss_fn(model, inputs, labels):
            logits = model(inputs)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels, axis=-1)
            return loss.mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, labels)
        optimizer.update(model, grads)
        return loss, model, optimizer

    cfg = DataConfig(batch_size=128, num_epochs=50, shuffle=True, as_chw=False)
    iterator = make_dataloader("train", cfg)
    step = 0
    for images, labels, masks in iterator:
        loss, model, optimizer = step_fn(model, optimizer, images, masks)
        step += 1
        if step % 10 == 0:
            print(f"Step {step}: loss={float(loss):.4f}")

    print("Training complete. Sampling test batch for visualization...")

    test_cfg = DataConfig(batch_size=8, num_epochs=1, shuffle=False, as_chw=False)
    test_iterator = make_dataloader("test", test_cfg)
    test_images, test_labels, test_masks = next(test_iterator)

    logits = model(test_images)
    pred_masks = jnp.argmax(logits, axis=-1).astype(jnp.int32)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("outputs") / "unet"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"predicted_overlays_{timestamp}.png"

    print("Predicted segmentation overlays:")
    visualize_batch(
        test_images,
        labels=test_labels,
        masks=pred_masks,
        max_display=8,
        mask_alpha=0.5,
        mask_cmap="viridis",
        save_path=pred_path,
        show=False,
    )
    print(f"Saved predicted overlays to {pred_path}")

    if test_masks is not None:
        gt_path = output_dir / f"ground_truth_overlays_{timestamp}.png"
        print("Ground truth segmentation overlays:")
        visualize_batch(
            test_images,
            labels=test_labels,
            masks=test_masks,
            max_display=8,
            mask_alpha=0.5,
            mask_cmap="magma",
            save_path=gt_path,
            show=False,
        )
        print(f"Saved ground truth overlays to {gt_path}")
