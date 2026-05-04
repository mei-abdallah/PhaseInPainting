"""
InSAR-EdgeConnect: Two-Stage GAN for Masked Interferogram Reconstruction.

Adaptation of EdgeConnect (Nazeri et al., 2019) for InSAR wrapped-phase
inpainting.

Stage 1 — Contour Generator
    Hallucinate missing fringe-contour lines (the "edges" of the
    interferogram) inside the masked region.  Contour lines are the
    boundaries of phase-cycle fringes; they carry the structural information
    needed by Stage 2.

Stage 2 — Phase Inpainter
    Given the original masked phase *and* the completed contour map from
    Stage 1, predict the missing phase values.

Both stages are trained with their own PatchGAN discriminator plus
feature-matching and task-specific losses (ContourLoss for Stage 1,
CircularPhaseLoss for Stage 2).

Inference (forward pass)
    1.  ContourGenerator predicts contours inside the mask.
    2.  Merge predicted contours with known contours:
            contour_complete = contour_known + (1 - mask) * contour_pred
    3.  PhaseInpainter uses the merged contour map to reconstruct phase.
    4.  Merge predicted phase with known phase:
            phase_complete = phase_known + (1 - mask) * phase_pred

Network inputs
--------------
All inputs are float32 tensors of shape (B, C, H, W), with phase in [-1, 1]
(network-space, i.e. wrapped_phase / π) and contours in [0, 1].

Stage 1 input : [phase_masked, contour_masked, mask]         — 3 channels
Stage 2 input : [phase_masked, contour_complete, mask]       — 3 channels
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from network.modules import (
    build_contour_generator,
    build_phase_generator,
    build_discriminator,
    DiscriminatorWithFeatures,
)
from network.loss import (
    LSGANLoss,
    FeatureMatchingLoss,
    CircularPhaseLoss,
    ContourLoss,
)




# ---------------------------------------------------------------------------
# InSAREdgeConnect model
# ---------------------------------------------------------------------------

class InSAREdgeConnect(nn.Module):
    """
    Two-stage InSAR phase reconstruction model.

    Args:
        base_ch          : Base filter count for generators.
        n_res            : Number of residual blocks at each generator bottleneck.
        adv_weight       : Adversarial loss weight.
        fm_weight        : Feature-matching loss weight.
        phase_loss_weight: Weight for the circular phase loss.
        contour_loss_weight: Weight for the contour BCE loss.
    """

    def __init__(
        self,
        base_ch: int = 64,
        n_res: int = 8,
        adv_weight: float = 1.0,
        fm_weight: float = 10.0,
        phase_loss_weight: float = 10.0,
        contour_loss_weight: float = 1.0,
        r1_weight: float = 10.0,
    ):
        super().__init__()

        # ---------- Generators ----------
        self.contour_gen = build_contour_generator(base_ch=base_ch, n_res=n_res)
        self.phase_gen   = build_phase_generator(base_ch=base_ch, n_res=n_res)

        # ---------- Discriminators ----------
        # Contour D: real/fake contour map concatenated with masked phase
        self.contour_disc = DiscriminatorWithFeatures(build_discriminator(in_ch=2))
        # Phase D: real/fake phase concatenated with completed contour map
        self.phase_disc = DiscriminatorWithFeatures(build_discriminator(in_ch=2))

        # ---------- Losses ----------
        self.adv_loss = LSGANLoss()
        self.fm_loss = FeatureMatchingLoss(weight=fm_weight)
        self.phase_loss = CircularPhaseLoss()
        self.contour_loss = ContourLoss()

        # ---------- Weights ----------
        self.adv_w = adv_weight
        self.phase_loss_w = phase_loss_weight
        self.contour_loss_w = contour_loss_weight
        self.r1_weight = r1_weight

    # ------------------------------------------------------------------
    # Forward (inference only — training uses separate methods below)
    # ------------------------------------------------------------------

    def forward(
        self,
        phase_masked: torch.Tensor,
        contour_masked: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full two-stage forward pass.

        Args:
            phase_masked   : (B, 1, H, W) masked phase in [-1, 1].
            contour_masked : (B, 1, H, W) masked contour map in [0, 1].
            mask           : (B, 1, H, W) binary mask (1=valid, 0=missing).

        Returns:
            phase_complete  : (B, 1, H, W) reconstructed phase in [-1, 1].
            contour_complete: (B, 1, H, W) reconstructed contour map in [0, 1].
        """
        # Stage 1: predict contours in masked region
        inp1 = torch.cat([phase_masked, contour_masked, mask], dim=1)
        contour_pred = self.contour_gen(inp1)
        contour_complete = contour_masked + (1.0 - mask) * contour_pred

        # Stage 2: inpaint phase using completed contours
        inp2 = torch.cat([phase_masked, contour_complete.detach(), mask], dim=1)
        phase_pred = self.phase_gen(inp2)
        phase_complete = phase_masked + (1.0 - mask) * phase_pred

        return phase_complete, contour_complete

    # ------------------------------------------------------------------
    # Stage 1: Contour generator training step
    # ------------------------------------------------------------------

    def contour_generator_loss(
        self,
        phase_masked: torch.Tensor,
        contour_masked: torch.Tensor,
        contour_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Stage-1 generator losses.

        Returns total loss and a dict of named scalar components.
        """
        inp1 = torch.cat([phase_masked, contour_masked, mask], dim=1)
        contour_pred = self.contour_gen(inp1)
        contour_complete = contour_masked + (1.0 - mask) * contour_pred

        # BCE contour losses
        l_cnt_valid, l_cnt_masked = self.contour_loss(
            contour_pred, contour_gt, mask
        )
        l_cnt = self.contour_loss_w * (l_cnt_valid + l_cnt_masked)

        # Adversarial loss (conditioned on masked phase)
        real_input = torch.cat([phase_masked, contour_gt], dim=1)
        fake_input = torch.cat([phase_masked, contour_complete], dim=1)

        d_fake, feat_fake = self.contour_disc(fake_input)
        d_real, feat_real = self.contour_disc(real_input)

        l_adv = self.adv_w * self.adv_loss.generator_loss(d_fake)
        l_fm = self.fm_loss(feat_real, feat_fake)

        total = l_cnt + l_adv + l_fm
        info = {
            "cnt_valid": l_cnt_valid.item(),
            "cnt_masked": l_cnt_masked.item(),
            "cnt_adv": l_adv.item(),
            "cnt_fm": l_fm.item(),
        }
        return total, info

    def contour_discriminator_loss(
        self,
        phase_masked: torch.Tensor,
        contour_masked: torch.Tensor,
        contour_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Stage-1 discriminator loss."""
        with torch.no_grad():
            inp1 = torch.cat([phase_masked, contour_masked, mask], dim=1)
            contour_pred = self.contour_gen(inp1)
            contour_complete = contour_masked + (1.0 - mask) * contour_pred

        real_input = torch.cat([phase_masked, contour_gt], dim=1)
        fake_input = torch.cat([phase_masked, contour_complete], dim=1)

        real_input.requires_grad_(True)
        d_real, _ = self.contour_disc(real_input)
        d_fake, _ = self.contour_disc(fake_input.detach())

        loss_d = self.adv_loss.discriminator_loss(d_real, d_fake)

        # R1 gradient penalty
        grad_real = torch.autograd.grad(
            outputs=d_real.sum(), inputs=real_input, create_graph=True
        )[0]
        r1_penalty = 0.5 * grad_real.pow(2).view(grad_real.size(0), -1).sum(1).mean()
        return loss_d + self.r1_weight * r1_penalty

    # ------------------------------------------------------------------
    # Stage 2: Phase inpainter training step
    # ------------------------------------------------------------------

    def phase_generator_loss(
        self,
        phase_masked: torch.Tensor,
        contour_complete: torch.Tensor,
        phase_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Stage-2 generator losses.

        ``contour_complete`` should be the *detached* completed contour map
        from Stage 1 (or the ground-truth contour for Stage-2-only training).
        """
        inp2 = torch.cat([phase_masked, contour_complete.detach(), mask], dim=1)
        phase_pred = self.phase_gen(inp2)
        phase_complete = phase_masked + (1.0 - mask) * phase_pred

        # Circular phase losses
        l_ph_valid, l_ph_masked = self.phase_loss(phase_pred, phase_gt, mask)
        l_ph = self.phase_loss_w * (l_ph_valid + l_ph_masked)

        # Adversarial
        real_input = torch.cat([contour_complete, phase_gt], dim=1)
        fake_input = torch.cat([contour_complete, phase_complete], dim=1)

        d_fake, feat_fake = self.phase_disc(fake_input)
        d_real, feat_real = self.phase_disc(real_input)

        l_adv = self.adv_w * self.adv_loss.generator_loss(d_fake)
        l_fm = self.fm_loss(feat_real, feat_fake)

        total = l_ph + l_adv + l_fm
        info = {
            "ph_valid": l_ph_valid.item(),
            "ph_masked": l_ph_masked.item(),
            "ph_adv": l_adv.item(),
            "ph_fm": l_fm.item(),
        }
        return total, info

    def phase_discriminator_loss(
        self,
        phase_masked: torch.Tensor,
        contour_complete: torch.Tensor,
        phase_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Stage-2 discriminator loss."""
        with torch.no_grad():
            inp2 = torch.cat([phase_masked, contour_complete.detach(), mask], dim=1)
            phase_pred = self.phase_gen(inp2)
            phase_complete = phase_masked + (1.0 - mask) * phase_pred

        real_input = torch.cat([contour_complete, phase_gt], dim=1)
        fake_input = torch.cat([contour_complete, phase_complete], dim=1)

        real_input.requires_grad_(True)
        d_real, _ = self.phase_disc(real_input)
        d_fake, _ = self.phase_disc(fake_input.detach())

        loss_d = self.adv_loss.discriminator_loss(d_real, d_fake)

        # R1 gradient penalty
        grad_real = torch.autograd.grad(
            outputs=d_real.sum(), inputs=real_input, create_graph=True
        )[0]
        r1_penalty = 0.5 * grad_real.pow(2).view(grad_real.size(0), -1).sum(1).mean()
        return loss_d + self.r1_weight * r1_penalty
