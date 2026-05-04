"""
Phase utilities for InSAR interferogram processing.

Provides:
- Phase contour / fringe-line extraction (analogous to Canny edges for
  natural images, but adapted for the circular nature of wrapped phase).
- Phase wrapping / unwrapping helpers.
- Visualisation helpers (phase-to-colour HSV map).
"""

import numpy as np
import torch
from typing import Optional


def wrap_phase_tensor(phase: torch.Tensor) -> torch.Tensor:
    """Wrap a torch phase tensor to [-π, π] (differentiable via atan2)."""
    return torch.atan2(torch.sin(phase), torch.cos(phase))


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def phase_to_net(phase: np.ndarray) -> np.ndarray:
    """Scale wrapped phase from [-π, π] to [-1, 1] for network input."""
    return (phase / np.pi).astype(np.float32)


def net_to_phase(x: np.ndarray) -> np.ndarray:
    """Scale network output from [-1, 1] back to [-π, π]."""
    return (x * np.pi).astype(np.float32)


def phase_to_net_tensor(phase: torch.Tensor) -> torch.Tensor:
    return phase / np.pi


def net_to_phase_tensor(x: torch.Tensor) -> torch.Tensor:
    return x * np.pi


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def phase_to_rgb(
    wrapped_phase: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert wrapped phase to an HSV-based RGB image for visualisation.

    The hue encodes the phase value (0 → 2π mapped to 0 → 360°),
    saturation = 1, value = 1 (or 0.3 for masked pixels).

    Args:
        wrapped_phase : 2-D float32 array in [-π, π].
        mask          : Optional binary mask (1=valid, 0=masked).

    Returns:
        uint8 RGB array of shape (H, W, 3).
    """
    import matplotlib.colors as mcolors

    hue = (wrapped_phase + np.pi) / (2 * np.pi)  # [0, 1]
    sat = np.ones_like(hue)
    val = np.ones_like(hue)

    if mask is not None:
        val[mask == 0] = 0.3
        sat[mask == 0] = 0.0

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)
