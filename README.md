# PhaseInPainting — InSAR Interferometric Phase Reconstruction

> General implementation of **"A novel two-stage adversarial joint learning model for reconstructing InSAR phase in decorrelated areas"**  
> Abdallah, M., Wu, S., & Ding, X. (2026). *Science of Remote Sensing*, 13, 100373.  
> [https://doi.org/10.1016/j.srs.2026.100373](https://doi.org/10.1016/j.srs.2026.100373)

---

## Overview

SAR coherence masking removes entire regions of an interferogram — decorrelated pixels, shadow/layover zones, water bodies, agricultural fields. The masked region severs the *fringe contour network* (the phase-cycle boundaries that carry the topographic or deformation signal). Naive inpainting fails because phase is circular: π and −π are the same value.

**PhaseInPainting** solves this in two stages (following the paper's terminology):

| Stage | Paper Name | Task | Input | Output |
|-------|-----------|------|-------|--------|
| 1 | **EMS** — Edge Mapping Stage | A pre-trained CNN detects existing fringe lines; a GAN reconnects the discontinuous fringes | `[phase_masked, contour_masked, mask]` | Sigmoid contour map |
| 2 | **PPS** — Phase Predicting Stage | A second GAN uses the reconnected fringe map as structural guidance to reconstruct phase values | `[phase_masked, contour_complete, mask]` | Tanh phase map ∈ [−1, 1] |

### Key Results (from the paper)

| Metric | Value |
|--------|-------|
| Fringe reconnection overall accuracy (OA) | **84 %** |
| Phase reconstruction SSIM | **96 %** |
| Cross-correlation on Greater Bay Area (with fine-tuning) | **0.72 – 0.87** |

Validated on real co-seismic interferograms: **Tonopah, Nevada earthquake** (M 6.5, 15 May 2020) and **Western Xizang earthquake** (M 6.3, 22 July 2020).

---

## Repository Structure

```
PhaseInPainting/
├── config.py              # Config dataclass — all hyperparameters
├── generate.py            # Offline dataset generation script
├── train.py               # Training script (modes 1 / 2 / 3)
├── evaluate.py            # Evaluation / inference script
│
├── data/
│   ├── simdem.py          # DEMSimulator — fractal / diamond-square terrain
│   ├── simphase.py        # InSARSimulator — DEM → wrapped interferogram
│   ├── simmask.py         # RealMaskLoader — binary masks from coherence MAT files
│   ├── utils.py           # phase_to_net / net_to_phase, wrap_phase_tensor, phase_to_rgb
│   └── visualise.py       # plot_sample(), plot_training_logs()
│
└── network/
    ├── dataset.py         # InSARDataset — offline TIFF dataset reader
    ├── modules.py         # UNetGenerator, PatchDiscriminator, InSARVGGCanny
    ├── loss.py            # LSGANLoss, FeatureMatchingLoss, CircularPhaseLoss, ContourLoss
    ├── metric.py          # phase_circular_mae/rmse, phase_psnr/ssim, fringe_region_metrics
    └── models.py          # InSAREdgeConnect — two-stage model
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision
- numpy, scipy, Pillow, tifffile, scikit-image

---

## Quick Start

### 1 — Generate the offline dataset

```bash
python generate.py --train_size 20000 --test_size 2000
```

Writes float32 GeoTIFFs to `outputs/dataset/{train,test}/{ifg,mask,edge}/`.  
IFGs are saved in wrapped-phase radians; masks and edge maps in `[0, 1]`.

### 2 — Train

```bash
# Joint training — both stages simultaneously (recommended)
python train.py --mode 3

# Stage 1 only (contour generator)
python train.py --mode 1

# Stage 2 only (phase inpainter, requires a stage-1 checkpoint)
python train.py --mode 2
```

Checkpoints are written to `outputs/run_default/` and training resumes automatically from `latest.pth` if present.

### 3 — Evaluate

```bash
python evaluate.py --ckpt outputs/run_default/latest.pth
```

---

## Network Architecture

### Generator — `UNetGenerator`

Five-level U-Net with skip connections, spectral normalisation, and Instance Norm.

| Block | Type | Channels |
|-------|------|----------|
| Encoder × 5 | Strided conv → IN → LeakyReLU | 64 → 512 |
| Bottleneck × 8 | Dilated residual blocks (dilation 1/2/4) | 512 |
| Decoder × 5 | Bilinear upsample → conv → IN → ReLU (+ skip) | 512 → 64 |
| Output | Conv → Sigmoid / Tanh | 1 |

### Discriminator — `PatchDiscriminator`

70×70 PatchGAN with spectral normalisation (3 strided-conv layers, base channels = 64).

### Fringe-line Detector — `InSARVGGCanny`

Frozen VGG19 feature extractor that maps normalised phase φ/π ∈ [−1, 1] to a binary fringe-contour map. Used in the EMS to identify existing fringe lines (both during dataset generation and at inference time).

---

## Loss Functions

### Stage 1 — Contour Generator

| Loss | Weight | Notes |
|------|--------|-------|
| LSGAN (adversarial) | `adv_weight = 1.0` | Real target = 0.9 (one-sided label smoothing) |
| Feature matching | `fm_weight = 10.0` | L1 distance between discriminator feature maps |
| Contour BCE | `contour_loss_weight = 1.0` | Positive-class weighted binary cross-entropy |

### Stage 2 — Phase Inpainter

| Loss | Weight | Notes |
|------|--------|-------|
| LSGAN (adversarial) | `adv_weight = 1.0` | |
| Feature matching | `fm_weight = 10.0` | |
| Circular phase loss | `phase_loss_weight = 10.0` | See below |

**Circular phase loss** — standard L1 is incorrect for wrapped phase because π ≡ −π. Instead:

$$\mathcal{L}_{\text{circ}} = \bigl(1 - \cos(\hat{\phi} - \phi)\bigr) + \lvert\sin(\hat{\phi} - \phi)\rvert + \lvert\cos(\hat{\phi} - \phi) - 1\rvert$$

with a 2× weight applied inside the masked region.

### GAN Stability

- **R1 gradient penalty** (weight = 10.0) on discriminator real inputs.
- Discriminator updated every `d_update_freq = 2` generator steps.

---

## Phase Normalisation

All normalisation is centralised in `data/utils.py`:

| Function | Direction | Formula |
|----------|-----------|---------|
| `phase_to_net(x)` | radians → network | `x / π` → [−1, 1] |
| `net_to_phase(x)` | network → radians | `x × π` → [−π, π] |
| `wrap_phase_tensor(x)` | unwrapped → [−π, π] | `atan2(sin x, cos x)` |

---

## Outputs

| Path | Contents |
|------|----------|
| `outputs/dataset/{train,test}/` | Pre-generated TIFF splits |
| `outputs/run_default/latest.pth` | Latest checkpoint (auto-resume) |
| `outputs/run_default/ckpt_NNNNNNN.pth` | Periodic snapshots |
| `outputs/run_default/sample_NNNNNNN.png` | Visual samples during training |
| `outputs/run_default/logs/train.csv` | Per-iteration losses |
| `outputs/run_default/logs/val.csv` | Validation metrics |
| `outputs/run_default/logs/config.json` | Config snapshot |

---

## Metrics

| Metric | Description |
|--------|-------------|
| `val_mae_deg` | Mean absolute circular error (°) inside the masked region |
| `val_rmse_deg` | Root mean squared circular error (°) inside the masked region |
| `fringe_f1` | Pixel-level F1 for predicted vs. GT fringe-contour maps |
| `valid_region_mae` | MAE in the *unmasked* region (sanity check — should remain near 0) |
| `phase_psnr` | PSNR of the reconstructed phase in the masked region |
| `phase_ssim` | SSIM of the reconstructed phase in the masked region |

---

## Configuration Reference

All settings live in `config.py` as a `Config` dataclass and can be overridden via CLI flags.

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | `3` | Training stage: `1` = contour, `2` = phase, `3` = joint |
| `data_root` | `outputs/dataset` | Root of the offline TIFF dataset |
| `shape` | `(512, 512)` | Spatial resolution of generated samples |
| `train_size` | `20 000` | Number of training samples to generate |
| `batch_size` | `4` | Mini-batch size |
| `base_ch` | `64` | Generator base channel count |
| `n_res` | `8` | Residual blocks in the generator bottleneck |
| `lr` | `1e-4` | Generator learning rate |
| `d2g_lr_ratio` | `0.1` | Discriminator LR = G LR × this ratio |
| `beta1` / `beta2` | `0.0` / `0.9` | Adam β parameters |
| `max_iters` | `200 000` | Total training iterations |
| `save_interval` | `5 000` | Checkpoint save frequency (iterations) |
| `val_interval` | `2 000` | Validation frequency (iterations) |
| `perp_baseline` | `25.0 m` | Perpendicular baseline (controls fringe density) |
| `min_masked_fraction` | `0.0` | Minimum mask coverage per sample |
| `max_masked_fraction` | `0.45` | Maximum mask coverage per sample |
| `r1_weight` | `10.0` | R1 gradient penalty weight |
| `d_update_freq` | `2` | Discriminator update interval (generator steps) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{abdallah2026phaseinpainting,
  title   = {A novel two-stage adversarial joint learning model for reconstructing
             {InSAR} phase in decorrelated areas},
  author  = {Abdallah, Mahmoud and Wu, Songbo and Ding, Xiaoli},
  journal = {Science of Remote Sensing},
  volume  = {13},
  pages   = {100373},
  year    = {2026},
  doi     = {10.1016/j.srs.2026.100373},
  url     = {https://www.sciencedirect.com/science/article/pii/S2666017226000118}
}
```

---

## References

- Abdallah, M., Wu, S., & Ding, X. (2026). [A novel two-stage adversarial joint learning model for reconstructing InSAR phase in decorrelated areas](https://doi.org/10.1016/j.srs.2026.100373). *Science of Remote Sensing*, 13, 100373.
- Nazeri, K. et al. (2019). [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://arxiv.org/abs/1901.00212). *ICCV Workshops*.
- Hanssen, R. F. (2001). *Radar Interferometry: Data Interpretation and Error Analysis*. Springer.
- Wang, X. et al. (2018). [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585). *CVPR*.

---

## License

MIT
