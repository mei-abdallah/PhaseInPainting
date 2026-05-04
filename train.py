"""
Training script for InSAR-EdgeConnect.

Supports three training modes (set via cfg.mode):
  1 — Train contour generator + contour discriminator only.
  2 — Train phase inpainter + phase discriminator only
      (uses ground-truth contours from dataset).
  3 — Joint training of both stages (Stage 1 → Stage 2 pipeline).

Usage::

    python train.py                     # default Config
    python train.py --mode 1            # contour stage only
    python train.py --batch_size 8 --lr 2e-4
"""

import argparse
import csv
import json
import os
import itertools
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from network.dataset import InSARDataset
from network.models import InSAREdgeConnect
from network.metric import phase_region_metrics, fringe_region_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: InSAREdgeConnect,
    opt_cg: torch.optim.Optimizer,
    opt_cd: torch.optim.Optimizer,
    opt_pg: torch.optim.Optimizer,
    opt_pd: torch.optim.Optimizer,
    iteration: int,
    path: str,
) -> None:
    torch.save(
        {
            "iteration": iteration,
            "contour_gen": model.contour_gen.state_dict(),
            "contour_disc": model.contour_disc.state_dict(),
            "phase_gen": model.phase_gen.state_dict(),
            "phase_disc": model.phase_disc.state_dict(),
            "opt_cg": opt_cg.state_dict(),
            "opt_cd": opt_cd.state_dict(),
            "opt_pg": opt_pg.state_dict(),
            "opt_pd": opt_pd.state_dict(),
        },
        path,
    )


def _load_pretrain_checkpoint(model: InSAREdgeConnect, path: str, keys: list) -> None:
    """Load specific sub-network weights from a checkpoint (no optimizer/iteration)."""
    ckpt = torch.load(path, map_location="cpu")
    loaded: list[str] = []
    net_map = {
        "contour_gen":  model.contour_gen,
        "contour_disc": model.contour_disc,
        "phase_gen":    model.phase_gen,
        "phase_disc":   model.phase_disc,
    }
    for key in keys:
        if key in ckpt:
            net_map[key].load_state_dict(ckpt[key])
            loaded.append(key)
    print(f"[train] pretrain loaded {loaded} from {path}")


def _load_checkpoint(
    model: InSAREdgeConnect,
    opt_cg: torch.optim.Optimizer,
    opt_cd: torch.optim.Optimizer,
    opt_pg: torch.optim.Optimizer,
    opt_pd: torch.optim.Optimizer,
    path: str,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.contour_gen.load_state_dict(ckpt["contour_gen"])
    model.contour_disc.load_state_dict(ckpt["contour_disc"])
    model.phase_gen.load_state_dict(ckpt["phase_gen"])
    model.phase_disc.load_state_dict(ckpt["phase_disc"])
    opt_cg.load_state_dict(ckpt["opt_cg"])
    opt_cd.load_state_dict(ckpt["opt_cd"])
    opt_pg.load_state_dict(ckpt["opt_pg"])
    opt_pd.load_state_dict(ckpt["opt_pd"])
    return ckpt["iteration"]


@torch.no_grad()
def _save_sample(
    model: InSAREdgeConnect,
    batch: dict,
    device: torch.device,
    path: str,
    mode: int = 3,
) -> None:
    """Save a grid of [Phase GT | Contour GT | Mask GT | Phase masked | Contour masked | Contour pred | Phase pred]."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    phase_masked = batch["phase_masked"][:4].to(device)
    contour_masked = batch["contour_masked"][:4].to(device)
    mask = batch["mask"][:4].to(device)
    phase_gt = batch["phase_gt"][:4].to(device)
    contour_gt = batch["contour_gt"][:4].to(device)

    phase_out, contour_out = model(phase_masked, contour_masked, mask)

    # In mode 2 the phase_gen is trained on GT contours, not predicted ones.
    # Re-run phase_gen with GT contours so the sample reflects actual training.
    if mode == 2:
        inp2 = torch.cat([phase_masked, contour_gt, mask], dim=1)
        phase_pred2 = model.phase_gen(inp2)
        phase_out = phase_masked + (1.0 - mask) * phase_pred2
        # contour_out from model.forward() is the contour_gen output (random in mode 2);
        # replace with GT so the Contour(pred) panel is meaningful
        contour_out = contour_gt

    n = phase_masked.shape[0]
    # order: GT reference first, then inputs, then predictions
    titles = ["Phase (GT)", "Contour (GT)", "Mask (GT)",
              "Phase (masked)", "Contour (masked)",
              "Contour (pred)", "Phase (pred)"]
    rows = [
        phase_gt.cpu().numpy()[:, 0],
        contour_gt.cpu().numpy()[:, 0],
        mask.cpu().numpy()[:, 0],
        phase_masked.cpu().numpy()[:, 0],
        contour_masked.cpu().numpy()[:, 0],
        contour_out.cpu().numpy()[:, 0],
        phase_out.cpu().numpy()[:, 0],
    ]
    # gray for binary/mask panels (indices 1,2,4,5), RdBu for phase panels
    _gray_cols = {1, 2, 4, 5}

    fig, axes = plt.subplots(n, len(titles), figsize=(3 * len(titles), 3 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        for j, (row, title) in enumerate(zip(rows, titles)):
            ax = axes[i][j]
            if j in _gray_cols:
                # binary/mask panels: data in [0,1], threshold pred contour for display
                data = (row[i] > 0.5).astype(float) if title == "Contour (pred)" else row[i]
                ax.imshow(data, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(row[i], cmap="RdBu", vmin=-1, vmax=1)
            if i == 0:
                ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    model.train()


class _CSVLogger:
    """
    Append-mode CSV logger.  First write creates the file and writes headers.
    Subsequent writes append rows.  Thread-safe for single-process training.
    """

    def __init__(self, path: str, fieldnames: list) -> None:
        self.path = path
        self.fieldnames = fieldnames
        existed = os.path.exists(path)
        self._fh = open(path, "a", newline="", buffering=1)  # line-buffered
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames, extrasaction="ignore")
        if not existed:
            self._writer.writeheader()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    # ---- device ----
    if torch.cuda.is_available() and cfg.gpu_ids:
        device = torch.device(f"cuda:{cfg.gpu_ids[0]}")
    else:
        device = torch.device("cpu")
    print(f"[train] device = {device}")

    # ---- directories (already created in _parse_args) ----
    _ckpt_dir   = os.path.join(cfg.run_dir, "checkpoints")
    _sample_dir = os.path.join(cfg.run_dir, "sample")
    _log_dir    = os.path.join(cfg.run_dir, "logs")

    # ---- loggers ----
    _TRAIN_FIELDS = [
        "iteration", "timestamp", "mode",
        "loss_cg", "loss_cd", "cnt_valid", "cnt_masked", "cnt_adv", "cnt_fm",
        "loss_pg", "loss_pd", "ph_valid", "ph_masked", "ph_adv", "ph_fm",
    ]
    _VAL_FIELDS = [
        "iteration", "timestamp",
        "phase_masked_mae", "phase_masked_rmse", "phase_masked_psnr", "phase_masked_ssim",
        "phase_valid_mae",  "phase_valid_rmse",  "phase_valid_psnr",  "phase_valid_ssim",
        "fringe_masked_f1", "fringe_masked_precision", "fringe_masked_recall", "fringe_masked_acc",
        "fringe_valid_f1",  "fringe_valid_precision",  "fringe_valid_recall",  "fringe_valid_acc",
    ]
    train_logger = _CSVLogger(os.path.join(_log_dir, "train.csv"), _TRAIN_FIELDS)
    val_logger   = _CSVLogger(os.path.join(_log_dir, "val.csv"),   _VAL_FIELDS)

    # Save a human-readable config snapshot next to the logs
    cfg_path = os.path.join(_log_dir, "config.json")
    if not os.path.exists(cfg_path):
        import dataclasses
        with open(cfg_path, "w") as _f:
            json.dump(dataclasses.asdict(cfg), _f, indent=2)

    # ---- datasets ----
    print(f"[train] dataset root: '{cfg.data_root}'")
    train_ds = InSARDataset(
        root=cfg.data_root,
        split=cfg.train_split,
        random_mask=cfg.random_mask,
        augment=cfg.augment,
        flip_h_prob=cfg.flip_h_prob,
        flip_v_prob=cfg.flip_v_prob,
        seed=cfg.seed,
    )
    val_ds = InSARDataset(
        root=cfg.data_root,
        split=cfg.val_split,
        random_mask=False,
        augment=False,
        seed=(cfg.seed + 99999 if cfg.seed is not None else None),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(4, cfg.batch_size), shuffle=False,
        num_workers=0, drop_last=False,
    )

    # ---- model ----
    model = InSAREdgeConnect(
        base_ch=cfg.base_ch,
        n_res=cfg.n_res,
        adv_weight=cfg.adv_weight,
        fm_weight=cfg.fm_weight,
        phase_loss_weight=cfg.phase_loss_weight,
        contour_loss_weight=cfg.contour_loss_weight,
        r1_weight=cfg.r1_weight,
    ).to(device)

    # ---- optimisers ----
    d_lr = cfg.lr * cfg.d2g_lr_ratio
    opt_cg = torch.optim.Adam(
        model.contour_gen.parameters(),
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
    )
    opt_cd = torch.optim.Adam(
        model.contour_disc.parameters(),
        lr=d_lr, betas=(cfg.beta1, cfg.beta2),
    )
    opt_pg = torch.optim.Adam(
        model.phase_gen.parameters(),
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
    )
    opt_pd = torch.optim.Adam(
        model.phase_disc.parameters(),
        lr=d_lr, betas=(cfg.beta1, cfg.beta2),
    )

    # ---- resume if checkpoint exists, else warm-start from pretrain ckpts ----
    iteration = 0
    latest = os.path.join(_ckpt_dir, "latest.pth")
    if os.path.exists(latest):
        iteration = _load_checkpoint(model, opt_cg, opt_cd, opt_pg, opt_pd, latest)
        print(f"[train] resumed from iteration {iteration}")
    elif cfg.pretrain_ckpt_contour or cfg.pretrain_ckpt_phase:
        if cfg.pretrain_ckpt_contour:
            _load_pretrain_checkpoint(model, cfg.pretrain_ckpt_contour, ["contour_gen", "contour_disc"])
        if cfg.pretrain_ckpt_phase:
            _load_pretrain_checkpoint(model, cfg.pretrain_ckpt_phase,   ["phase_gen",   "phase_disc"])

    # ---- training ----
    model.train()
    train_iter = itertools.cycle(train_loader)
    val_batch = next(iter(val_loader))

    t0 = time.time()
    loss_cd = loss_cg = loss_pd = loss_pg = torch.tensor(0.0)
    info_cg: dict = {}
    info_pg: dict = {}
    while iteration < cfg.max_iters:
        batch = next(train_iter)

        phase_masked  = batch["phase_masked"].to(device)
        contour_masked = batch["contour_masked"].to(device)
        mask          = batch["mask"].to(device)
        phase_gt      = batch["phase_gt"].to(device)
        contour_gt    = batch["contour_gt"].to(device)

        # ============================================================
        # Stage 1 — Contour Generator
        # ============================================================
        if cfg.mode in (1, 3):
            # --- Discriminator step (every d_update_freq iters) ---
            if iteration % cfg.d_update_freq == 0:
                opt_cd.zero_grad()
                loss_cd = model.contour_discriminator_loss(
                    phase_masked, contour_masked, contour_gt, mask
                )
                loss_cd.backward()
                opt_cd.step()

            # --- Generator step ---
            opt_cg.zero_grad()
            loss_cg, info_cg = model.contour_generator_loss(
                phase_masked, contour_masked, contour_gt, mask
            )
            loss_cg.backward()
            opt_cg.step()

        # ============================================================
        # Stage 2 — Phase Inpainter
        # ============================================================
        if cfg.mode in (2, 3):
            # Use predicted contours in joint mode, GT otherwise
            with torch.no_grad():
                if cfg.mode == 3:
                    inp1 = torch.cat([phase_masked, contour_masked, mask], dim=1)
                    contour_pred = model.contour_gen(inp1)
                    contour_complete = contour_masked + (1.0 - mask) * contour_pred
                else:
                    contour_complete = contour_gt

            # --- Discriminator step (every d_update_freq iters) ---
            if iteration % cfg.d_update_freq == 0:
                opt_pd.zero_grad()
                loss_pd = model.phase_discriminator_loss(
                    phase_masked, contour_complete, phase_gt, mask
                )
                loss_pd.backward()
                opt_pd.step()

            # --- Generator step ---
            opt_pg.zero_grad()
            loss_pg, info_pg = model.phase_generator_loss(
                phase_masked, contour_complete, phase_gt, mask
            )
            loss_pg.backward()
            opt_pg.step()

        iteration += 1

        # ---- logging ----
        if iteration % cfg.log_interval == 0:
            elapsed = time.time() - t0
            msg = f"[{iteration:6d}/{cfg.max_iters}] t={elapsed:.0f}s"
            log_row: dict = {
                "iteration": iteration,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode": cfg.mode,
            }
            if cfg.mode in (1, 3):
                msg += (f"  cG={loss_cg.item():.3f} cD={loss_cd.item():.3f}"
                        f"  cnt_masked={info_cg['cnt_masked']:.3f}")
                log_row.update({
                    "loss_cg":   round(loss_cg.item(), 5),
                    "loss_cd":   round(loss_cd.item(), 5),
                    "cnt_valid": round(info_cg.get("cnt_valid", 0.0), 5),
                    "cnt_masked":round(info_cg["cnt_masked"], 5),
                    "cnt_adv":   round(info_cg.get("cnt_adv", 0.0), 5),
                    "cnt_fm":    round(info_cg.get("cnt_fm", 0.0), 5),
                })
            if cfg.mode in (2, 3):
                msg += (f"  pG={loss_pg.item():.3f} pD={loss_pd.item():.3f}"
                        f"  ph_masked={info_pg['ph_masked']:.3f}")
                log_row.update({
                    "loss_pg":  round(loss_pg.item(), 5),
                    "loss_pd":  round(loss_pd.item(), 5),
                    "ph_valid": round(info_pg.get("ph_valid", 0.0), 5),
                    "ph_masked":round(info_pg["ph_masked"], 5),
                    "ph_adv":   round(info_pg.get("ph_adv", 0.0), 5),
                    "ph_fm":    round(info_pg.get("ph_fm", 0.0), 5),
                })
            print(msg)
            train_logger.write(log_row)

        # ---- sample images ----
        if iteration % cfg.sample_interval == 0:
            sample_path = os.path.join(_sample_dir, f"sample_{iteration:07d}.png")
            _save_sample(model, val_batch, device, sample_path, cfg.mode)

        # ---- validation metric ----
        if iteration % cfg.val_interval == 0 and cfg.mode in (2, 3):
            model.eval()
            total_pm_mae = total_pm_rmse = total_pm_psnr = total_pm_ssim = 0.0
            total_pv_mae = total_pv_rmse = total_pv_psnr = total_pv_ssim = 0.0
            total_mf1 = total_mprec = total_mrec = total_macc = 0.0
            total_vf1 = total_vprec = total_vrec = total_vacc = 0.0
            n_val = 0
            with torch.no_grad():
                for vb in val_loader:
                    vm  = vb["phase_masked"].to(device)
                    vc  = vb["contour_masked"].to(device)
                    vk  = vb["mask"].to(device)
                    vg  = vb["phase_gt"].to(device)
                    vcg = vb["contour_gt"].to(device)
                    ph_out, cnt_out = model(vm, vc, vk)
                    bs = vm.shape[0]
                    p = phase_region_metrics(ph_out, vg, vk, input_space="net")
                    total_pm_mae  += p["masked_mae_deg"]  * bs
                    total_pm_rmse += p["masked_rmse_deg"] * bs
                    total_pm_psnr += p["masked_psnr"]     * bs
                    total_pm_ssim += p["masked_ssim"]     * bs
                    total_pv_mae  += p["valid_mae_deg"]   * bs
                    total_pv_rmse += p["valid_rmse_deg"]  * bs
                    total_pv_psnr += p["valid_psnr"]      * bs
                    total_pv_ssim += p["valid_ssim"]      * bs
                    m = fringe_region_metrics(cnt_out, vcg, vk)
                    total_mf1   += m["masked_f1"]        * bs
                    total_mprec += m["masked_precision"] * bs
                    total_mrec  += m["masked_recall"]    * bs
                    total_macc  += m["masked_accuracy"]  * bs
                    total_vf1   += m["valid_f1"]         * bs
                    total_vprec += m["valid_precision"]  * bs
                    total_vrec  += m["valid_recall"]     * bs
                    total_vacc  += m["valid_accuracy"]   * bs
                    n_val += bs
            pm_mae  = total_pm_mae  / n_val
            pm_rmse = total_pm_rmse / n_val
            pm_psnr = total_pm_psnr / n_val
            pm_ssim = total_pm_ssim / n_val
            pv_mae  = total_pv_mae  / n_val
            pv_rmse = total_pv_rmse / n_val
            pv_psnr = total_pv_psnr / n_val
            pv_ssim = total_pv_ssim / n_val
            val_mf1   = total_mf1   / n_val
            val_mprec = total_mprec / n_val
            val_mrec  = total_mrec  / n_val
            val_macc  = total_macc  / n_val
            val_vf1   = total_vf1   / n_val
            val_vprec = total_vprec / n_val
            val_vrec  = total_vrec  / n_val
            val_vacc  = total_vacc  / n_val
            print(f"  [val] Phase Masked  MAE={pm_mae:.2f}° RMSE={pm_rmse:.2f}° PSNR={pm_psnr:.2f}dB SSIM={pm_ssim:.4f}")
            print(f"        Phase Valid    MAE={pv_mae:.2f}° RMSE={pv_rmse:.2f}° PSNR={pv_psnr:.2f}dB SSIM={pv_ssim:.4f}")
            print(f"        Masked  F1={val_mf1:.4f} P={val_mprec:.4f} R={val_mrec:.4f} Acc={val_macc:.4f}")
            print(f"        Valid   F1={val_vf1:.4f} P={val_vprec:.4f} R={val_vrec:.4f} Acc={val_vacc:.4f}")
            val_logger.write({
                "iteration":               iteration,
                "timestamp":               time.strftime("%Y-%m-%dT%H:%M:%S"),
                "phase_masked_mae":        round(pm_mae,   4),
                "phase_masked_rmse":       round(pm_rmse,  4),
                "phase_masked_psnr":       round(pm_psnr,  4),
                "phase_masked_ssim":       round(pm_ssim,  6),
                "phase_valid_mae":         round(pv_mae,   4),
                "phase_valid_rmse":        round(pv_rmse,  4),
                "phase_valid_psnr":        round(pv_psnr,  4),
                "phase_valid_ssim":        round(pv_ssim,  6),
                "fringe_masked_f1":        round(val_mf1,   6),
                "fringe_masked_precision": round(val_mprec, 6),
                "fringe_masked_recall":    round(val_mrec,  6),
                "fringe_masked_acc":       round(val_macc,  6),
                "fringe_valid_f1":         round(val_vf1,   6),
                "fringe_valid_precision":  round(val_vprec, 6),
                "fringe_valid_recall":     round(val_vrec,  6),
                "fringe_valid_acc":        round(val_vacc,  6),
            })
            model.train()

        # ---- checkpoint ----
        if iteration % cfg.save_interval == 0:
            _save_checkpoint(
                model, opt_cg, opt_cd, opt_pg, opt_pd, iteration,
                os.path.join(_ckpt_dir, f"ckpt_{iteration:07d}.pth"),
            )
            _save_checkpoint(
                model, opt_cg, opt_cd, opt_pg, opt_pd, iteration,
                os.path.join(_ckpt_dir, "latest.pth")
            )

    train_logger.close()
    val_logger.close()
    print("[train] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train InSAR-EdgeConnect")

    parser.add_argument("--mode",          type=int,   default=cfg.mode)
    parser.add_argument("--batch_size",    type=int,   default=cfg.batch_size)
    parser.add_argument("--lr",            type=float, default=cfg.lr)
    parser.add_argument("--max_iters",     type=int,   default=cfg.max_iters)
    parser.add_argument("--base_ch",       type=int,   default=cfg.base_ch)
    parser.add_argument("--n_res",         type=int,   default=cfg.n_res)
    parser.add_argument("--data_root",     type=str,   default=cfg.data_root)
    parser.add_argument("--train_split",   type=str,   default=cfg.train_split)
    parser.add_argument("--val_split",     type=str,   default=cfg.val_split)
    parser.add_argument("--random_mask",   type=int,   choices=[0, 1], default=int(cfg.random_mask))
    parser.add_argument("--augment",       type=int,   choices=[0, 1], default=int(cfg.augment))
    parser.add_argument("--flip_h_prob",   type=float, default=cfg.flip_h_prob)
    parser.add_argument("--flip_v_prob",   type=float, default=cfg.flip_v_prob)
    parser.add_argument("--r1_weight",     type=float, default=cfg.r1_weight)
    parser.add_argument("--d_update_freq", type=int,   default=cfg.d_update_freq)
    parser.add_argument("--seed",          type=int,   default=cfg.seed)
    parser.add_argument("--gpu_id",        type=int,   default=cfg.gpu_ids[0],
                        help="GPU index to use (e.g. 0, 1, 3).")
    parser.add_argument("--run_dir",       type=str,   default=None,
                        help="Root dir for this run. If omitted, auto-creates "
                             "outputs/run_mode<N>_<YYYYMMDD_HHMMSS>/")
    parser.add_argument("--pretrain_ckpt_contour", type=str, default=None,
                        help="Mode-1 checkpoint: warm-starts contour_gen + contour_disc for mode 3.")
    parser.add_argument("--pretrain_ckpt_phase",   type=str, default=None,
                        help="Mode-2 checkpoint: warm-starts phase_gen + phase_disc for mode 3.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Auto-build a timestamped run directory when --run_dir is not given
    # ------------------------------------------------------------------
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.run_dir = os.path.join("outputs", f"run_mode{args.mode}_{ts}")
    else:
        cfg.run_dir = args.run_dir

    cfg.gpu_ids        = [args.gpu_id]
    cfg.mode           = args.mode
    cfg.batch_size     = args.batch_size
    cfg.lr             = args.lr
    cfg.max_iters      = args.max_iters
    cfg.base_ch        = args.base_ch
    cfg.n_res          = args.n_res
    cfg.data_root      = args.data_root
    cfg.train_split    = args.train_split
    cfg.val_split      = args.val_split
    cfg.random_mask    = bool(args.random_mask)
    cfg.augment        = bool(args.augment)
    cfg.flip_h_prob    = args.flip_h_prob
    cfg.flip_v_prob    = args.flip_v_prob
    cfg.r1_weight      = args.r1_weight
    cfg.d_update_freq  = args.d_update_freq
    # Create all output dirs now so subprocesses never fail on missing dirs
    for subdir in ("checkpoints", "sample", "logs"):
        Path(cfg.run_dir, subdir).mkdir(parents=True, exist_ok=True)
    print(f"[train] run_dir = {cfg.run_dir}")
    cfg.pretrain_ckpt_contour = args.pretrain_ckpt_contour
    cfg.pretrain_ckpt_phase   = args.pretrain_ckpt_phase
    cfg.seed           = args.seed
    return cfg


if __name__ == "__main__":
    train(_parse_args())
