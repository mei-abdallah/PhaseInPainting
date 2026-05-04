"""
Real coherence mask loader for InSAR phase reconstruction.

Loads binary masks from real InSAR coherence (*cc_adf*.mat) files.
Convention: 1 = valid pixel, 0 = masked / decorrelated.
"""

import glob
import os
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio


class RealMaskLoader:
    """
    Load binary masks from real InSAR coherence MAT files.

    On initialization, all matching files under base_dir are scanned and
    cached in memory. Each call to sample() returns a random crop of the
    requested shape, thresholded into a binary mask.

    Args:
        base_dir: Root directory to scan (for example, .../japan/bad).
        coh_threshold: Coherence threshold; coherence >= threshold is valid.
        pattern: Recursive glob pattern under base_dir.
    """

    def __init__(
        self,
        base_dir: str,
        coh_threshold: float = 0.4,
        pattern: str = "**/*cc_adf*.mat",
    ):
        self.coh_threshold = coh_threshold
        self._arrays: list[np.ndarray] = []

        paths = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        if not paths:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found under '{base_dir}'"
            )

        for path in sorted(paths):
            mat = sio.loadmat(path)
            keys = [k for k in mat.keys() if not k.startswith("_")]
            if not keys:
                continue
            arr = mat[keys[0]].astype(np.float32)
            self._arrays.append(arr)

        if not self._arrays:
            raise ValueError(
                f"No coherence arrays were loaded from '{base_dir}'."
            )

        print(f"Loaded {len(self._arrays)} coherence arrays")

    def sample(
        self,
        shape: Tuple[int, int],
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Return a random binary mask crop with shape (H, W).

        Args:
            shape: Output crop shape (H, W).
            rng: Optional numpy RandomState for reproducibility.

        Returns:
            float32 binary mask, shape (H, W), with 1=valid and 0=masked.
        """
        if rng is None:
            rng = np.random.RandomState()

        ny, nx = shape

        candidates = [
            a for a in self._arrays
            if a.shape[0] >= ny and a.shape[1] >= nx
        ]
        if not candidates:
            smallest = min(a.shape for a in self._arrays)
            raise ValueError(
                f"No loaded coherence array is large enough for shape {shape}. "
                f"Smallest array: {smallest}"
            )

        arr = candidates[rng.randint(len(candidates))]
        y0 = rng.randint(0, arr.shape[0] - ny + 1)
        x0 = rng.randint(0, arr.shape[1] - nx + 1)
        crop = arr[y0:y0 + ny, x0:x0 + nx]

        return (crop >= self.coh_threshold).astype(np.float32)
