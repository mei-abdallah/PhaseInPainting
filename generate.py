"""
Generate an offline InSAR dataset on disk.

Creates train/test splits under:
    outputs/dataset/train/{ifg,mask,edge}
    outputs/dataset/test/{ifg,mask,edge}

Data format:
    - TIFF float32 files.
    - IFG is saved in wrapped phase radians [-pi, pi].
    - Mask and edge are saved as float32 in [0, 1].

This preserves numerical values while keeping files viewable in most
scientific image tools.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from config import Config
from data.simdem import DEMSimulator
from data.simphase import InSARSimulator
from data.simmask import RealMaskLoader
from data.utils import phase_to_net
from network.modules import InSARVGGCanny


def _save_tiff_float32(path: Path, arr: np.ndarray) -> None:
    """Save a 2-D array as float32 TIFF without value scaling."""
    a32 = arr.astype(np.float32, copy=False)
    Image.fromarray(a32, mode="F").save(str(path), format="TIFF")


def _prepare_split_dirs(split_root: Path) -> Tuple[Path, Path, Path]:
    ifg_dir = split_root / "ifg"
    mask_dir = split_root / "mask"
    edge_dir = split_root / "edge"

    for d in (ifg_dir, mask_dir, edge_dir):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.tif"):
            p.unlink()

    return ifg_dir, mask_dir, edge_dir


def _export_split(
    split_name: str,
    n_samples: int,
    split_seed: int | None,
    shape: Tuple[int, int],
    out_root: Path,
    mask_dir: str,
    dem_method: str,
    coh_threshold: float,
    perp_baseline: float,
    min_masked_fraction: float,
    max_masked_fraction: float,
    max_mask_tries: int,
) -> None:
    split_root = out_root / split_name
    ifg_dir, mask_out_dir, edge_dir = _prepare_split_dirs(split_root)

    rng = np.random.RandomState(split_seed)

    dem_sim = DEMSimulator(shape=shape, seed=int(rng.randint(0, 2**31)))
    ifg_sim = InSARSimulator(perp_baseline=perp_baseline)
    mask_loader = RealMaskLoader(base_dir=mask_dir, coh_threshold=coh_threshold)
    edge_detector = InSARVGGCanny()  

    print(f"[generate] writing {split_name}: {n_samples} samples")
    for i in range(n_samples):
        sample_rng = np.random.RandomState(int(rng.randint(0, 2**31)))

        # Try up to max_mask_tries to get a mask with the right fraction
        mask = None
        for _ in range(max_mask_tries):
            m = mask_loader.sample(shape, rng=sample_rng)
            masked_fraction = 1.0 - float(m.mean())
            if min_masked_fraction <= masked_fraction <= max_masked_fraction:
                mask = m
                break

        if mask is None:
            raise RuntimeError(
                f"Could not find sample with {min_masked_fraction:.2f} <= masked_fraction <= {max_masked_fraction:.2f} "
                f"after {max_mask_tries} tries for split='{split_name}', sample={i}. "
                f"Try adjusting --min_masked_fraction / --max_masked_fraction or --coh_threshold."
            )

        # Re-seed DEM sim per sample for variety
        dem_sim.seed = int(sample_rng.randint(0, 2**31))
        np.random.seed(dem_sim.seed)
        dem = dem_sim.generate_dem(method=dem_method)
        ifg_rad = ifg_sim.generate_interferogram(dem, rng=sample_rng, deformation=True)
        # predict() takes normalised phase φ/π ∈ [-1,1] as numpy (H,W)
        edge = edge_detector.predict(phase_to_net(ifg_rad))

        stem = f"{i:07d}.tif"
        _save_tiff_float32(ifg_dir / stem, ifg_rad)
        _save_tiff_float32(mask_out_dir / stem, mask)
        _save_tiff_float32(edge_dir / stem, edge)

        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            print(f"  [{split_name}] {i + 1}/{n_samples}")


def _parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Generate offline TIFF dataset")
    parser.add_argument("--train_size",  type=int,   default=cfg.train_size)
    parser.add_argument("--test_size",   type=int,   default=cfg.test_size)
    parser.add_argument("--shape",       type=int,   nargs=2, default=list(cfg.shape))
    parser.add_argument("--out_root",    type=str,   default=cfg.data_root)
    parser.add_argument("--mask_dir",    type=str,   default=cfg.mask_dir)
    parser.add_argument("--dem_method",  type=str,   default=cfg.dem_method)
    parser.add_argument("--coh_threshold", type=float, default=cfg.coh_threshold)
    parser.add_argument("--perp_baseline", type=float, default=cfg.perp_baseline)
    parser.add_argument("--seed",        type=int,   default=(cfg.seed if cfg.seed is not None else 42))
    parser.add_argument("--min_masked_fraction", type=float, default=cfg.min_masked_fraction,
                        help="Minimum fraction of masked pixels per sample (0-1).")
    parser.add_argument("--max_masked_fraction", type=float, default=cfg.max_masked_fraction,
                        help="Maximum fraction of masked pixels per sample (0-1).")
    parser.add_argument("--max_mask_tries", type=int, default=cfg.max_mask_tries,
                        help="Max resampling attempts to satisfy mask fraction bounds.")
    return parser.parse_args()



def main() -> None:
    args = _parse_args()

    shape = (int(args.shape[0]), int(args.shape[1]))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_seed = int(args.seed)
    test_seed = train_seed + 99999

    _export_split(
        split_name="train",
        n_samples=int(args.train_size),
        split_seed=train_seed,
        shape=shape,
        out_root=out_root,
        mask_dir=args.mask_dir,
        dem_method=args.dem_method,
        coh_threshold=float(args.coh_threshold),
        perp_baseline=float(args.perp_baseline),
        min_masked_fraction=float(args.min_masked_fraction),
        max_masked_fraction=float(args.max_masked_fraction),
        max_mask_tries=int(args.max_mask_tries),
    )
    _export_split(
        split_name="test",
        n_samples=int(args.test_size),
        split_seed=test_seed,
        shape=shape,
        out_root=out_root,
        mask_dir=args.mask_dir,
        dem_method=args.dem_method,
        coh_threshold=float(args.coh_threshold),
        perp_baseline=float(args.perp_baseline),
        min_masked_fraction=float(args.min_masked_fraction),
        max_masked_fraction=float(args.max_masked_fraction),
        max_mask_tries=int(args.max_mask_tries),
    )

    meta = {
        "format": "tiff-float32",
        "ifg_units": "radians_wrapped_-pi_to_pi",
        "mask_range": "0_to_1",
        "edge_range": "0_to_1",
        "train_size": int(args.train_size),
        "test_size": int(args.test_size),
        "shape": [shape[0], shape[1]],
        "mask_dir": args.mask_dir,
        "dem_method": args.dem_method,
        "coh_threshold": float(args.coh_threshold),
        "perp_baseline": float(args.perp_baseline),
        "seed": int(args.seed),
        "min_masked_fraction": float(args.min_masked_fraction),
        "max_mask_tries": int(args.max_mask_tries),
    }
    with open(out_root / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[generate] done")
    print(f"[generate] output root: {out_root}")


if __name__ == "__main__":
    main()
