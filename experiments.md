# Experiments

## Experiment 1: Vanilla Autoencoder on CelebA

**Date:** 2026-01-11

**Model:** Vanilla convolutional autoencoder
- Encoder: 3 conv layers (3→16→32→64 channels), linear to latent
- Latent dimension: 16
- Decoder: linear from latent, 3 transposed conv layers (64→32→16→3)
- Activations: ReLU between layers
- Loss: MSE reconstruction loss

**Dataset:** CelebA, resized to 56×56, normalized to [0,1]

**Training:**
- Optimizer: Adam, lr=1e-3
- ~5000 steps (1 epoch)
- Final loss: ~0.015-0.018

### Results

**Generation from random latents:** Poor quality
- Distorted, artifact-heavy images
- Strange color patterns (inverted colors, blue tints)
- Barely recognizable as faces

**Reconstruction of real images:** Better but blurry
- Face structure preserved
- Colors generally correct
- Significant blur/smoothing of details

### Analysis

**Why generation fails:**
Vanilla autoencoders have no regularization on the latent space. The encoder maps training images to arbitrary points that may be sparse or irregularly distributed. When sampling random vectors from N(0,1), these points fall in "holes" where the decoder was never trained, producing garbage outputs.

**Why reconstructions are blurry:**
1. **MSE loss** encourages predicting the mean of plausible outputs rather than sharp details. When uncertain, the model outputs blurry predictions to minimize squared error.
2. **Small latent dimension** (16) limits capacity to encode fine details.
3. **No perceptual or adversarial loss** - nothing encourages perceptually sharp outputs.

### Conclusion

Vanilla autoencoders are suitable for reconstruction tasks but not for generation. The lack of latent space regularization means random sampling doesn't produce meaningful outputs. For generation, VAEs (with KL regularization) or GANs are needed.

---

## Experiment 2: Variational Autoencoder (VAE) on CelebA

**Date:** 2026-01-11

**Model:** Convolutional VAE
- Encoder: 3 conv layers (3→16→32→64 channels), linear to 2×latent (mu, log_var)
- Latent dimension: 16
- Decoder: linear from latent, 3 transposed conv layers (64→32→16→3)
- Reparameterization trick: z = mu + exp(log_var/2) * epsilon
- Loss: MSE reconstruction + KL divergence

**Dataset:** CelebA, resized to 56×56, normalized to [0,1]

**Training:**
- Optimizer: Adam, lr=1e-3
- ~5000 steps (1 epoch)

### Results

**Generation from random latents:** Much better than vanilla AE
- Recognizable face structures
- Coherent colors and features
- Still blurry but clearly face-like

**Reconstruction of real images:** Worse than vanilla AE
- More blur than vanilla autoencoder
- Less sharp details
- Colors slightly washed out

### Analysis: The Reconstruction-Generation Trade-off

VAE shows the opposite pattern from vanilla AE: better generation, worse reconstruction. This is due to the KL divergence term.

**How KL helps generation:**
Forces the encoder to produce latents distributed as N(0,1). Random samples from N(0,1) are now in-distribution for the decoder.

**How KL hurts reconstruction:**
The encoder wants to:
- Push `mu` values apart to distinguish different images
- Shrink `log_var` (reduce variance) to encode precise information

But KL divergence pulls:
- `mu` toward 0
- `log_var` toward 0 (variance = 1)

This tension means the encoder can't fully utilize the latent space for reconstruction - it must compromise to maintain a "nice" latent distribution.

### Comparison: Vanilla AE vs VAE

| Metric | Vanilla AE | VAE |
|--------|------------|-----|
| Reconstruction quality | Better | Worse |
| Generation quality | Poor (artifacts) | Better (coherent) |
| Latent space structure | Unstructured | N(0,1) regularized |
| Loss function | MSE only | MSE + KL |

### Conclusion

VAEs trade reconstruction quality for generation capability. The KL term is essential for generation but actively degrades reconstruction. This trade-off is fundamental to the VAE framework and has motivated techniques like beta-VAE (adjustable KL weight) and VAE-GAN hybrids.

---

## Experiment Roadmap

1. [x] Vanilla Autoencoder - baseline reconstruction, no generation capability
2. [ ] Variational Autoencoder (VAE) - add KL regularization for generation
3. [ ] Diffusion models - iterative denoising approach
4. [ ] Flow models - invertible transformations, exact likelihood
