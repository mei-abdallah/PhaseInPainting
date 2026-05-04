"""
InSAR Phase Reconstruction – Offline Dataset.

Reads pre-generated TIFF samples from disk (created by generate.py).

Each sample contains:
  - phase_masked  : wrapped phase with mask applied (zeros in masked region)
  - contour_masked: fringe-contour map with mask applied
  - mask          : binary mask (1=valid, 0=missing)
  - phase_gt      : ground-truth wrapped phase [-1, 1]
  - contour_gt    : ground-truth fringe-contour map [0, 1]

All tensors are float32, shape (1, H, W).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict
from pathlib import Path

from PIL import Image
from data.utils import phase_to_net


class InSARDataset(Dataset):
    """
    Offline dataset reader for pre-generated TIFF samples.

    Expected structure::

        root/split/ifg/*.tif
        root/split/edge/*.tif
        root/split/mask/*.tif

    Requirements:
      - IFG and edge must share the same filename stem (1:1 pairing).
      - Mask can be sampled from any file in the split to increase variety.

    Args:
        root         : Dataset root, e.g. ``outputs/dataset``.
        split        : ``train`` or ``test``.
        random_mask  : If True, sample a random mask for each IFG/edge pair.
                       If False, use mask with the same filename when present.
        augment      : Enable data augmentation (random flips).
        flip_h_prob  : Probability of horizontal flip when augment=True.
        flip_v_prob  : Probability of vertical flip when augment=True.
        seed         : Base RNG seed; each sample uses (seed + idx).
    """

    def __init__(
        self,
        root: str = "outputs/dataset",
        split: str = "train",
        random_mask: bool = True,
        augment: bool = False,
        flip_h_prob: float = 0.5,
        flip_v_prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.random_mask = random_mask
        self.augment = augment
        self.flip_h_prob = flip_h_prob
        self.flip_v_prob = flip_v_prob
        self.base_seed = seed

        split_root = self.root / split
        self.ifg_dir = split_root / "ifg"
        self.edge_dir = split_root / "edge"
        self.mask_dir = split_root / "mask"

        if not self.ifg_dir.exists() or not self.edge_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Missing one or more split folders under '{split_root}'. "
                f"Expected ifg/, edge/, mask/."
            )

        self.ifg_paths = sorted(self.ifg_dir.glob("*.tif"))
        self.mask_paths = sorted(self.mask_dir.glob("*.tif"))
        if not self.ifg_paths:
            raise FileNotFoundError(f"No IFG TIFF files found in '{self.ifg_dir}'.")
        if not self.mask_paths:
            raise FileNotFoundError(f"No mask TIFF files found in '{self.mask_dir}'.")

        # Build strict IFG↔edge pairing by identical filename.
        self.pairs = []
        for ifg_p in self.ifg_paths:
            edge_p = self.edge_dir / ifg_p.name
            if not edge_p.exists():
                raise FileNotFoundError(
                    f"Missing edge file for IFG '{ifg_p.name}' in '{self.edge_dir}'."
                )
            self.pairs.append((ifg_p, edge_p))

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _read_tiff(path: Path) -> np.ndarray:
        arr = np.array(Image.open(path), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D TIFF at '{path}', got shape {arr.shape}.")
        return arr

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr[np.newaxis]).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seed = None if self.base_seed is None else self.base_seed + idx
        rng = np.random.RandomState(seed)

        ifg_path, edge_path = self.pairs[idx]
        ifg_rad = self._read_tiff(ifg_path)                # expected [-pi, pi]
        contour_gt = self._read_tiff(edge_path)            # expected [0, 1]

        if self.random_mask:
            mask_path = self.mask_paths[rng.randint(len(self.mask_paths))]
        else:
            same_name = self.mask_dir / ifg_path.name
            mask_path = same_name if same_name.exists() else self.mask_paths[idx % len(self.mask_paths)]

        mask = self._read_tiff(mask_path)

        if ifg_rad.shape != contour_gt.shape or ifg_rad.shape != mask.shape:
            raise ValueError(
                "Shape mismatch among IFG/edge/mask: "
                f"ifg={ifg_rad.shape}, edge={contour_gt.shape}, mask={mask.shape}."
            )

        # Convert to model space.
        phase_norm = phase_to_net(np.clip(ifg_rad, -np.pi, np.pi))  # [-1, 1]
        contour_gt = np.clip(contour_gt, 0.0, 1.0).astype(np.float32)         # [0, 1]
        mask = (mask >= 0.5).astype(np.float32)                               # binary 0/1

        # Optional augmentation: flips must be applied consistently to all channels.
        if self.augment:
            if rng.rand() < self.flip_h_prob:
                phase_norm = np.fliplr(phase_norm).copy()
                contour_gt = np.fliplr(contour_gt).copy()
                mask = np.fliplr(mask).copy()
            if rng.rand() < self.flip_v_prob:
                phase_norm = np.flipud(phase_norm).copy()
                contour_gt = np.flipud(contour_gt).copy()
                mask = np.flipud(mask).copy()

        phase_masked = phase_norm * mask
        contour_masked = contour_gt * mask

        return {
            "phase_masked": self._to_tensor(phase_masked),
            "contour_masked": self._to_tensor(contour_masked),
            "mask": self._to_tensor(mask),
            "phase_gt": self._to_tensor(phase_norm),
            "contour_gt": self._to_tensor(contour_gt),
        }

