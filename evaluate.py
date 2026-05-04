"""
Evaluate a trained InSAR-EdgeConnect checkpoint on the test set.

Usage::

    python evaluate.py --ckpt outputs/run_mode3_20260425_002334/checkpoints/latest.pth
    python evaluate.py --ckpt outputs/run_mode3_20260425_090614/checkpoints/latest.pth \
                       --gpu_id 1 --batch_size 8

Prints a summary table and writes results/eval_<run_name>.json.
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from config import Config
from network.dataset import InSARDataset
from network.models import InSAREdgeConnect
from network.metric import phase_region_metrics, fringe_region_metrics


def evaluate(ckpt_path: str, gpu_id: int = 0, batch_size: int = 4) -> dict:
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device = {device}")
    print(f"[eval] checkpoint = {ckpt_path}")

    # --- try to load training config snapshot ---
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    cfg_path = os.path.join(run_dir, "logs", "config.json")
    cfg = Config()
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k, v in saved.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        print(f"[eval] loaded config from {cfg_path}")

    # --- dataset ---
    test_ds = InSARDataset(
        root=cfg.data_root,
        split=cfg.val_split,
        random_mask=False,
        augment=False,
        seed=0,
    )
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, drop_last=False)
    print(f"[eval] test samples = {len(test_ds)}")

    # --- model ---
    model = InSAREdgeConnect(
        base_ch=cfg.base_ch,
        n_res=cfg.n_res,
        adv_weight=cfg.adv_weight,
        fm_weight=cfg.fm_weight,
        phase_loss_weight=cfg.phase_loss_weight,
        contour_loss_weight=cfg.contour_loss_weight,
        r1_weight=cfg.r1_weight,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.contour_gen.load_state_dict(ckpt["contour_gen"])
    model.contour_disc.load_state_dict(ckpt["contour_disc"])
    model.phase_gen.load_state_dict(ckpt["phase_gen"])
    model.phase_disc.load_state_dict(ckpt["phase_disc"])
    iteration = ckpt.get("iteration", "?")
    print(f"[eval] checkpoint iteration = {iteration}")
    model.eval()

    # --- accumulators ---
    totals = {k: 0.0 for k in [
        "phase_masked_mae", "phase_masked_rmse", "phase_masked_psnr", "phase_masked_ssim",
        "phase_valid_mae",  "phase_valid_rmse",  "phase_valid_psnr",  "phase_valid_ssim",
        "fringe_masked_f1", "fringe_masked_precision", "fringe_masked_recall", "fringe_masked_acc",
        "fringe_valid_f1",  "fringe_valid_precision",  "fringe_valid_recall",  "fringe_valid_acc",
    ]}
    n = 0

    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            vm  = batch["phase_masked"].to(device)
            vc  = batch["contour_masked"].to(device)
            vk  = batch["mask"].to(device)
            vg  = batch["phase_gt"].to(device)
            vcg = batch["contour_gt"].to(device)
            ph_out, cnt_out = model(vm, vc, vk)
            bs = vm.shape[0]

            p = phase_region_metrics(ph_out, vg, vk, input_space="net")
            f = fringe_region_metrics(cnt_out, vcg, vk)

            totals["phase_masked_mae"]        += p["masked_mae_deg"]  * bs
            totals["phase_masked_rmse"]       += p["masked_rmse_deg"] * bs
            totals["phase_masked_psnr"]       += p["masked_psnr"]     * bs
            totals["phase_masked_ssim"]       += p["masked_ssim"]     * bs
            totals["phase_valid_mae"]         += p["valid_mae_deg"]   * bs
            totals["phase_valid_rmse"]        += p["valid_rmse_deg"]  * bs
            totals["phase_valid_psnr"]        += p["valid_psnr"]      * bs
            totals["phase_valid_ssim"]        += p["valid_ssim"]      * bs
            totals["fringe_masked_f1"]        += f["masked_f1"]        * bs
            totals["fringe_masked_precision"] += f["masked_precision"] * bs
            totals["fringe_masked_recall"]    += f["masked_recall"]    * bs
            totals["fringe_masked_acc"]       += f["masked_accuracy"]  * bs
            totals["fringe_valid_f1"]         += f["valid_f1"]         * bs
            totals["fringe_valid_precision"]  += f["valid_precision"]  * bs
            totals["fringe_valid_recall"]     += f["valid_recall"]     * bs
            totals["fringe_valid_acc"]        += f["valid_accuracy"]   * bs

            n += bs
            if (i + 1) % 50 == 0:
                print(f"  {n}/{len(test_ds)} samples ...", flush=True)

    elapsed = time.time() - t0
    results = {k: round(v / n, 6) for k, v in totals.items()}
    results["n_samples"]  = n
    results["iteration"]  = iteration
    results["checkpoint"] = ckpt_path
    results["elapsed_s"]  = round(elapsed, 1)

    # --- print table ---
    print(f"\n{'='*62}")
    print(f"  Run: {os.path.basename(run_dir)}")
    print(f"  Iter: {iteration}   Samples: {n}   Time: {elapsed:.1f}s")
    print(f"{'='*62}")
    print(f"  Phase  [masked]  MAE={results['phase_masked_mae']:.2f}°  "
          f"RMSE={results['phase_masked_rmse']:.2f}°  "
          f"PSNR={results['phase_masked_psnr']:.2f}dB  "
          f"SSIM={results['phase_masked_ssim']:.4f}")
    print(f"  Phase  [valid ]  MAE={results['phase_valid_mae']:.2f}°  "
          f"RMSE={results['phase_valid_rmse']:.2f}°  "
          f"PSNR={results['phase_valid_psnr']:.2f}dB  "
          f"SSIM={results['phase_valid_ssim']:.4f}")
    print(f"  Fringe [masked]  F1={results['fringe_masked_f1']:.4f}  "
          f"P={results['fringe_masked_precision']:.4f}  "
          f"R={results['fringe_masked_recall']:.4f}  "
          f"Acc={results['fringe_masked_acc']:.4f}")
    print(f"  Fringe [valid ]  F1={results['fringe_valid_f1']:.4f}  "
          f"P={results['fringe_valid_precision']:.4f}  "
          f"R={results['fringe_valid_recall']:.4f}  "
          f"Acc={results['fringe_valid_acc']:.4f}")
    print(f"{'='*62}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate InSAR-EdgeConnect checkpoint")
    parser.add_argument("--ckpt",       type=str, required=True,
                        help="Path to checkpoint (.pth)")
    parser.add_argument("--gpu_id",     type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_dir",    type=str, default="results",
                        help="Directory to write JSON results")
    args = parser.parse_args()

    results = evaluate(args.ckpt, gpu_id=args.gpu_id, batch_size=args.batch_size)

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    out_path = os.path.join(args.out_dir, f"eval_{run_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] results saved to {out_path}")


if __name__ == "__main__":
    main()
