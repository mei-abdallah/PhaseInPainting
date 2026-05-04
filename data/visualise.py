"""
Visualisation helpers for InSAR interferogram reconstruction.

Primary entry points
--------------------
plot_sample(...)
    Multi-panel figure showing DEM, wrapped phase (GT + masked), fringe
    contours (GT + masked/predicted) and coherence mask side-by-side.

plot_training_logs(...)
    Loss and validation-metric curves from the CSV logs produced by train.py.

All functions accept both **numpy arrays** and **torch Tensors** (CPU or GPU).
Phase arrays are expected in either network-space ([-1, 1]) or radians ([-π, π]).
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
from data.utils import net_to_phase, phase_to_rgb as _phase_to_rgb_util

try:
    import matplotlib
    matplotlib.use("Agg")          # safe default; caller can switch backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Convert a torch Tensor *or* numpy array to a 2-D float32 numpy array."""
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    # squeeze leading batch / channel dims that are size-1
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    return x


def _phase_cmap(phase_arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Map a wrapped-phase array to an RGBA image using a cyclic HSV colormap.

    Args:
        phase_arr : 2-D float32, values in [-π, π]  *or*  [-1, 1] (net-space).
        mask      : 2-D float32 in [0, 1].  Pixels where mask < 0.5 are drawn
                    in dark grey so the missing region is immediately visible.

    Returns:
        RGBA uint8 array of shape (H, W, 4).
    """
    arr = phase_arr.copy()
    # auto-detect net-space (values in [-1,1]) and convert to radians
    if arr.max() <= 1.01 and arr.min() >= -1.01:
        arr = net_to_phase(arr)

    # delegate to utils.phase_to_rgb for the HSV mapping
    rgba_rgb = _phase_to_rgb_util(arr, mask=(mask if mask is not None else None))
    # phase_to_rgb returns uint8 RGB; add alpha channel
    rgba = np.concatenate(
        [rgba_rgb, np.full((*rgba_rgb.shape[:2], 1), 255, np.uint8)], axis=-1
    )
    if mask is not None:
        rgba[mask < 0.5] = [40, 40, 40, 255]
    return rgba


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_sample(
    dem: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    phase_gt: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    contour_gt: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    phase_masked: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    contour_masked: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    phase_pred: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    contour_pred: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    figsize_per_panel: float = 3.5,
) -> Optional["plt.Figure"]:
    """
    Multi-panel figure for one InSAR sample.

    Panels are shown left-to-right in this order (skipped if None):
      1. DEM          — terrain elevation, viridis colourmap
      2. Phase GT     — full wrapped phase, cyclic HSV colourmap
      3. Contour GT   — ground-truth fringe lines, grey
      4. Mask         — binary coherence mask, grey (white=valid)
      5. Phase masked — phase with mask applied (missing region = dark grey)
      6. Contour masked — contours with mask applied
      7. Phase pred   — network output phase (if available)
      8. Contour pred — network output contours (if available)

    Args:
        dem, phase_gt, contour_gt, mask : Input data panels.
        phase_masked, contour_masked    : Masked input panels.
        phase_pred, contour_pred        : Network prediction panels (optional).
        save_path  : If given, save figure to this path (PNG/PDF/…).
        show       : If True, call ``plt.show()``.
        title      : Optional super-title for the whole figure.
        figsize_per_panel : Width (inches) allocated to each panel.

    Returns:
        The ``matplotlib.figure.Figure`` object (useful for notebooks).
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot_sample().")

    # Build ordered list of (label, array, render_mode)
    panels = []

    def _add(label, arr, mode):
        if arr is not None:
            panels.append((label, _to_numpy(arr), mode))

    _add("DEM",            dem,            "viridis")
    _add("Phase GT",       phase_gt,       "phase")
    _add("Contour GT",     contour_gt,     "binary")
    _add("Mask",           mask,           "binary")
    _add("Phase (masked)", phase_masked,   "phase_masked")
    _add("Contour (mask)", contour_masked, "binary")
    _add("Phase (pred)",   phase_pred,     "phase")
    _add("Contour (pred)", contour_pred,   "binary")

    if not panels:
        raise ValueError("No panels to plot.  Provide at least one array.")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel * n, figsize_per_panel + 0.5))
    if n == 1:
        axes = [axes]

    # Resolve the mask array for overlay (if provided)
    mask_np = _to_numpy(mask) if mask is not None else None

    for ax, (label, arr, mode) in zip(axes, panels):
        if mode == "viridis":
            im = ax.imshow(arr, cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif mode == "phase":
            rgba = _phase_cmap(arr)
            ax.imshow(rgba)
        elif mode == "phase_masked":
            rgba = _phase_cmap(arr, mask=mask_np)
            ax.imshow(rgba)
        elif mode == "binary":
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr, cmap="gray")

        ax.set_title(label, fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_training_logs(
    log_dir: str,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional["plt.Figure"]:
    """
    Plot loss curves and validation metrics from the CSV logs written by train.py.

    Reads ``<log_dir>/train.csv`` and (optionally) ``<log_dir>/val.csv``.

    Args:
        log_dir   : Directory containing train.csv / val.csv.
        save_path : If given, save the figure here.
        show      : Call ``plt.show()`` after plotting.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required.")

    import csv

    def _read_csv(path):
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return {}
        out = {k: [] for k in rows[0]}
        for row in rows:
            for k, v in row.items():
                try:
                    out[k].append(float(v))
                except (ValueError, TypeError):
                    out[k].append(v)
        return out

    train = _read_csv(os.path.join(log_dir, "train.csv"))
    val   = _read_csv(os.path.join(log_dir, "val.csv"))

    has_train = bool(train)
    has_val   = bool(val)

    if not has_train and not has_val:
        raise FileNotFoundError(f"No CSV log files found in '{log_dir}'.")

    n_rows = 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()

    iters = train.get("iteration", [])

    # Panel 0 — contour generator / discriminator losses
    ax = axes[0]
    if "loss_cg" in train:
        ax.plot(iters, train["loss_cg"], label="cG", alpha=0.8)
    if "loss_cd" in train:
        ax.plot(iters, train["loss_cd"], label="cD", alpha=0.8)
    ax.set_title("Contour G / D loss")
    ax.set_xlabel("iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 1 — phase generator / discriminator losses
    ax = axes[1]
    if "loss_pg" in train:
        ax.plot(iters, train["loss_pg"], label="pG", alpha=0.8)
    if "loss_pd" in train:
        ax.plot(iters, train["loss_pd"], label="pD", alpha=0.8)
    ax.set_title("Phase G / D loss")
    ax.set_xlabel("iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2 — per-component contour loss terms
    ax = axes[2]
    for key, label in [("cnt_masked", "cnt masked"), ("cnt_valid", "cnt valid"),
                        ("ph_masked", "ph masked"), ("ph_valid", "ph valid")]:
        if key in train:
            ax.plot(iters, train[key], label=label, alpha=0.7)
    ax.set_title("Task loss components")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3 — validation metrics
    ax = axes[3]
    if has_val:
        viters = val.get("iteration", [])
        if "val_mae_deg" in val:
            ax.plot(viters, val["val_mae_deg"], marker="o", ms=4, label="MAE (°)", color="steelblue")
        if "val_rmse_deg" in val:
            ax.plot(viters, val["val_rmse_deg"], marker="s", ms=4, label="RMSE (°)", color="tomato")
        ax.set_ylabel("degrees", color="steelblue")
        if "fringe_f1" in val:
            ax2 = ax.twinx()
            ax2.plot(viters, val["fringe_f1"], marker="^", ms=4, label="Fringe F1",
                     color="green", linestyle="--")
            ax2.set_ylabel("F1", color="green")
            ax2.set_ylim(0, 1.05)
            ax2.legend(loc="lower right", fontsize=8)
    ax.set_title("Validation metrics")
    ax.set_xlabel("iteration")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training log — {log_dir}", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()

    return fig
