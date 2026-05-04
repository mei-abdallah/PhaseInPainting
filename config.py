"""
Configuration for InSAR-EdgeConnect training and inference.

All hyper-parameters are collected in a single dataclass so they can be
serialised to / from YAML or passed directly in Python.

Usage::

    from config import Config
    cfg = Config()               # default values
    cfg.batch_size = 4           # override interactively

    # Or load from YAML:
    import yaml
    with open("config.yaml") as f:
        cfg = Config(**yaml.safe_load(f))
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Training mode: 1=contour only, 2=phase only, 3=joint (both stages) #
    # ------------------------------------------------------------------ #
    mode: int = 3

    # ------------------------------------------------------------------ #
    # Data                                                               #
    # ------------------------------------------------------------------ #
    batch_size: int = 4
    num_workers: int = 4

    # Offline dataset (TIFF) settings — reads from data_root/{train_split,val_split}/{ifg,edge,mask}
    data_root: str = "outputs/dataset"
    train_split: str = "train"
    val_split: str = "test"
    random_mask: bool = True           # sample any mask per IFG for variety
    augment: bool = True               # random flip augmentation
    flip_h_prob: float = 0.5
    flip_v_prob: float = 0.5

    # ------------------------------------------------------------------ #
    # Generation (generate.py)                                           #
    # ------------------------------------------------------------------ #
    shape: tuple = (512, 512)          # spatial size of each sample
    train_size: int = 20_000
    test_size: int = 2_000
    dem_method: str = "fractal"        # 'fractal' | 'diamond_square'
    mask_dir: str = "outputs/coherencemaps"
    coh_threshold: float = 0.4
    perp_baseline: float = 25.0        # metres (controls fringe rate)
    min_masked_fraction: float = 0.0
    max_masked_fraction: float = 0.45
    max_mask_tries: int = 500

    # ------------------------------------------------------------------ #
    # Model                                                              #
    # ------------------------------------------------------------------ #
    base_ch: int = 64                         # generator base channels
    n_res: int = 8                            # residual blocks per generator
    arch: str = "unet"                        # generator architecture
    adv_weight: float = 1.0
    fm_weight: float = 10.0
    phase_loss_weight: float = 10.0
    contour_loss_weight: float = 1.0
    r1_weight: float = 10.0
    d_update_freq: int = 2                    # train D every N generator steps

    # ------------------------------------------------------------------ #
    # Optimiser                                                          #
    # ------------------------------------------------------------------ #
    lr: float = 1e-4
    d2g_lr_ratio: float = 0.1                # D lr = G lr × d2g_lr_ratio
    beta1: float = 0.0
    beta2: float = 0.9

    # ------------------------------------------------------------------ #
    # Training schedule                                                  #
    # ------------------------------------------------------------------ #
    max_iters: int = 200_000
    save_interval: int = 5_000
    sample_interval: int = 1_000
    log_interval: int = 50
    val_interval: int = 2_000

    # ------------------------------------------------------------------ #
    # I/O                                                                #
    # ------------------------------------------------------------------ #
    run_dir: str = "outputs/run_default"
    # Checkpoints from prior stages for warm-starting sub-networks in mode 3.
    # pretrain_ckpt_contour = mode-1 checkpoint (contour_gen + contour_disc)
    # pretrain_ckpt_phase   = mode-2 checkpoint (phase_gen + phase_disc)
    # Mode 2 does not need either (it trains on GT edges directly).
    pretrain_ckpt_contour: Optional[str] = None
    pretrain_ckpt_phase: Optional[str] = None
    seed: Optional[int] = 42

    # ------------------------------------------------------------------ #
    # Hardware                                                           #
    # ------------------------------------------------------------------ #
    gpu_ids: List[int] = field(default_factory=lambda: [0])
