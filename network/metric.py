"""
Evaluation metrics for InSAR phase reconstruction.

All functions accept torch Tensors in network-space (phase divided by π,
i.e. values in [-1, 1]) or in radians, as documented per function.

Phase metrics
-------------
phase_circular_mae   : Mean absolute circular error over the missing region.
phase_circular_rmse  : Root mean squared circular error over missing region.
phase_psnr           : PSNR (dB) over the full reconstructed phase image.
phase_ssim           : SSIM over the full reconstructed phase image.
phase_region_metrics : MAE/RMSE/PSNR/SSIM for missing region & full IFG.
valid_region_mae     : Circular MAE over the known region (sanity check).

Fringe-contour metrics
----------------------
fringe_region_metrics: Per-region F1/precision/recall/accuracy (masked & valid).
"""

import math

import torch
import torch.nn.functional as F
import numpy as np


def phase_circular_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_space: str = "net",
) -> float:
    """
    Mean absolute circular error (degrees) inside the **masked** region.

    Circular error correctly handles the periodicity of wrapped phase:
        err = |atan2(sin(Δφ), cos(Δφ))|

    Args:
        pred, target : (B, 1, H, W) phase tensors.
        mask         : (B, 1, H, W) binary mask (1=valid, 0=missing).
        input_space  : 'net' if values are in [-1, 1] (divide by π);
                       'rad' if values are in [-π, π].

    Returns:
        Scalar float — mean absolute circular error in degrees.
    """
    scale = np.pi if input_space == "net" else 1.0
    phi_pred = pred.float() * scale
    phi_gt = target.float() * scale
    diff = phi_pred - phi_gt

    err = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()  # [0, π]

    inv_mask = 1.0 - mask
    denom = inv_mask.sum().clamp(min=1.0)
    mae_rad = (err * inv_mask).sum() / denom
    return float(mae_rad) * 180.0 / np.pi


def phase_circular_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_space: str = "net",
) -> float:
    """
    Root mean squared circular error (degrees) inside the **masked** region.

    Args:
        pred, target : (B, 1, H, W) phase tensors.
        mask         : (B, 1, H, W) binary mask (1=valid, 0=missing).
        input_space  : 'net' or 'rad' (see ``phase_circular_mae``).

    Returns:
        Scalar float — RMSE in degrees.
    """
    scale = np.pi if input_space == "net" else 1.0
    phi_pred = pred.float() * scale
    phi_gt = target.float() * scale
    diff = phi_pred - phi_gt

    err_sq = torch.atan2(torch.sin(diff), torch.cos(diff)) ** 2

    inv_mask = 1.0 - mask
    denom = inv_mask.sum().clamp(min=1.0)
    rmse_rad = torch.sqrt((err_sq * inv_mask).sum() / denom)
    return float(rmse_rad) * 180.0 / np.pi


def phase_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    input_space: str = "net",
) -> float:
    """
    Peak signal-to-noise ratio (dB) over the **full** reconstructed phase.

    Args:
        pred, target : (B, 1, H, W) phase tensors.
        input_space  : 'net' — values in [-1, 1]; 'rad' — values in [-π, π].
                       Data range is set accordingly (2.0 or 2π).

    Returns:
        Scalar float — PSNR in dB.  Higher is better.
    """
    data_range = 2.0 if input_space == "net" else 2.0 * np.pi
    mse = ((pred.float() - target.float()) ** 2).mean()
    if mse.item() == 0:
        return float("inf")
    return float(10.0 * math.log10(data_range ** 2 / (mse.item() + 1e-12)))


def phase_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    input_space: str = "net",
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Structural similarity index (SSIM) over the **full** reconstructed phase.

    Uses a Gaussian-weighted sliding window (standard Wang et al. 2004).

    Args:
        pred, target : (B, 1, H, W) phase tensors.
        input_space  : 'net' ([-1, 1]) or 'rad' ([-π, π]).
        window_size  : Side length of the Gaussian window (default 11).
        sigma        : Standard deviation of the Gaussian (default 1.5).

    Returns:
        Scalar float — SSIM in [-1, 1].  Higher is better (1 = identical).
    """
    data_range = 2.0 if input_space == "net" else 2.0 * np.pi
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Build 1-D Gaussian kernel, then outer-product → 2-D
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.outer(gauss).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    kernel = kernel.to(pred.device)

    pad = window_size // 2
    x = pred.float()
    y = target.float()

    mu_x  = F.conv2d(x, kernel, padding=pad)
    mu_y  = F.conv2d(y, kernel, padding=pad)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_xx = F.conv2d(x * x, kernel, padding=pad) - mu_xx
    sig_yy = F.conv2d(y * y, kernel, padding=pad) - mu_yy
    sig_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)
    den = (mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2)
    ssim_map = num / den
    return float(ssim_map.mean())


def phase_region_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_space: str = "net",
    window_size: int = 11,
    sigma: float = 1.5,
) -> dict:
    """
    Phase reconstruction metrics split by region:
      masked_* : missing region only (mask=0)
      valid_*  : full IFG (all pixels)

    Each region reports: mae_deg, rmse_deg, psnr, ssim.
    SSIM is computed as a full per-pixel map (Wang et al. 2004) and then
    averaged over the relevant pixels.

    Args:
        pred, target : (B, 1, H, W) phase tensors.
        mask         : (B, 1, H, W) binary mask (1=valid, 0=missing).
        input_space  : 'net' ([-1, 1]) or 'rad' ([-π, π]).
        window_size  : Gaussian window side length for SSIM (default 11).
        sigma        : Gaussian sigma for SSIM (default 1.5).

    Returns:
        dict with 8 keys: masked_{mae_deg,rmse_deg,psnr,ssim} (missing region)
        and valid_{mae_deg,rmse_deg,psnr,ssim} (full IFG).
    """
    data_range = 2.0 if input_space == "net" else 2.0 * np.pi
    scale      = np.pi if input_space == "net" else 1.0
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    x = pred.float()
    y = target.float()

    # --- circular error maps ---
    diff     = x * scale - y * scale
    circ_err = torch.atan2(torch.sin(diff), torch.cos(diff))   # signed, radians
    circ_abs = circ_err.abs()
    sq_err   = (x - y) ** 2

    # --- SSIM map ---
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss  = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss  = gauss / gauss.sum()
    kernel = gauss.outer(gauss).unsqueeze(0).unsqueeze(0).to(pred.device)
    pad    = window_size // 2

    mu_x   = F.conv2d(x,     kernel, padding=pad)
    mu_y   = F.conv2d(y,     kernel, padding=pad)
    mu_xx  = mu_x * mu_x
    mu_yy  = mu_y * mu_y
    mu_xy  = mu_x * mu_y
    sig_xx = F.conv2d(x * x, kernel, padding=pad) - mu_xx
    sig_yy = F.conv2d(y * y, kernel, padding=pad) - mu_yy
    sig_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy
    ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)) / \
               ((mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2))

    def _stats(region_bool):
        n        = region_bool.sum().clamp(min=1.0)
        mae_rad  = circ_abs[region_bool].sum() / n
        rmse_rad = torch.sqrt((circ_err[region_bool] ** 2).sum() / n)
        mse      = sq_err[region_bool].sum() / n
        psnr_val = float(10.0 * math.log10(data_range ** 2 / (mse.item() + 1e-12)))
        ssim_val = float(ssim_map[region_bool].mean())
        return {
            "mae_deg":  float(mae_rad)  * 180.0 / np.pi,
            "rmse_deg": float(rmse_rad) * 180.0 / np.pi,
            "psnr":     psnr_val,
            "ssim":     ssim_val,
        }

    masked_stats = _stats((1.0 - mask).bool())  # missing region
    valid_stats  = _stats(torch.ones_like(mask).bool())  # full IFG
    return {
        **{f"masked_{k}": v for k, v in masked_stats.items()},
        **{f"valid_{k}":  v for k, v in valid_stats.items()},
    }


def fringe_region_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Fringe-contour classification metrics for both the **masked** (missing,
    mask=0) and **valid** (known, mask=1) regions.

    Args:
        pred      : (B, 1, H, W) predicted contour map in [0, 1].
        target    : (B, 1, H, W) ground-truth binary contour map.
        mask      : (B, 1, H, W) binary mask (1=valid, 0=missing).
        threshold : Binarisation threshold for ``pred``.

    Returns:
        dict with keys prefixed by ``masked_`` (missing region only) and
        ``valid_`` (full IFG), each containing f1, precision, recall,
        accuracy  (all in [0, 1]).
        ``masked_*`` covers the missing region (mask=0);
        ``valid_*`` covers the full IFG (all pixels).
    """
    pred_bin = (pred >= threshold).float()
    tgt_bin  = (target >= threshold).float()

    def _stats(p, t):
        tp = (p * t).sum()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()
        tn = ((1.0 - p) * (1.0 - t)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2.0 * precision * recall / (precision + recall + 1e-8)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        return {
            "f1":        float(f1),
            "precision": float(precision),
            "recall":    float(recall),
            "accuracy":  float(accuracy),
        }

    inv_mask = (1.0 - mask).bool()
    masked_stats = _stats(pred_bin[inv_mask], tgt_bin[inv_mask])  # missing region
    valid_stats  = _stats(pred_bin.flatten(), tgt_bin.flatten())  # full IFG

    return {
        **{f"masked_{k}": v for k, v in masked_stats.items()},
        **{f"valid_{k}":  v for k, v in valid_stats.items()},
    }


def valid_region_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    input_space: str = "net",
) -> float:
    """
    Mean absolute circular error (degrees) over the **valid** (known) region.

    Useful as a sanity check — a well-trained model should keep this near 0.

    Args:
        pred, target : (B, 1, H, W) phase tensors.
        mask         : (B, 1, H, W) binary mask (1=valid, 0=missing).
        input_space  : 'net' or 'rad'.

    Returns:
        Scalar float — MAE in degrees.
    """
    scale = np.pi if input_space == "net" else 1.0
    phi_pred = pred.float() * scale
    phi_gt = target.float() * scale
    diff = phi_pred - phi_gt

    err = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()

    denom = mask.sum().clamp(min=1.0)
    mae_rad = (err * mask).sum() / denom
    return float(mae_rad) * 180.0 / np.pi
