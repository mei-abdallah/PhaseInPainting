"""
Loss functions for InSAR-EdgeConnect.

Key InSAR-specific adaptations vs. the original EdgeConnect
------------------------------------------------------------
* **Circular phase loss** — plain L1 on wrapped phase is inappropriate
  because phase is periodic (π and -π are the same value).  We measure
  the complex-exponential distance:
      L_circ = 1 - cos(φ_pred - φ_gt)
  which is equivalent to L1 on the unit-circle representation.

* **Feature-matching loss** — both the edge/contour discriminator and the
  phase discriminator contribute feature-matching terms that enforce
  perceptual coherence without requiring a separate VGG network.

* **Adversarial loss** — LSGAN (least-squares GAN) is used for both
  discriminators because it avoids vanishing gradients and produces
  sharper boundaries without the mode collapse often seen with NSGAN.

* **Contour loss** — binary cross-entropy on the predicted contour map
  (sigmoid output) plus adversarial and feature-matching terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Adversarial loss (LSGAN)
# ---------------------------------------------------------------------------

class LSGANLoss(nn.Module):
    """
    Least-squares GAN loss with one-sided label smoothing.

    Real target = 0.9 (smoothed), fake target = 0.

    Usage::

        loss_g = criterion.generator_loss(d_fake)
        loss_d = criterion.discriminator_loss(d_real, d_fake)
    """

    real_label: float = 0.9   # one-sided label smoothing
    fake_label: float = 0.0

    def generator_loss(self, d_fake: torch.Tensor) -> torch.Tensor:
        """Generator wants D(fake) → 1."""
        return 0.5 * torch.mean((d_fake - 1.0) ** 2)

    def discriminator_loss(
        self,
        d_real: torch.Tensor,
        d_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Discriminator wants D(real) → 0.9 (smoothed), D(fake) → 0."""
        return 0.5 * (
            torch.mean((d_real - self.real_label) ** 2)
            + torch.mean((d_fake - self.fake_label) ** 2)
        )


# ---------------------------------------------------------------------------
# Feature-matching loss
# ---------------------------------------------------------------------------

class FeatureMatchingLoss(nn.Module):
    """
    L1 distance between intermediate discriminator features for real and
    fake inputs, normalised by the number of feature-map elements.

    Args:
        weight: Scalar multiplier applied to the total feature-matching loss.
    """

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        feat_real: List[torch.Tensor],
        feat_fake: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=feat_real[0].device)
        for fr, ff in zip(feat_real, feat_fake):
            loss = loss + F.l1_loss(ff, fr.detach())
        return self.weight * loss / len(feat_real)


# ---------------------------------------------------------------------------
# Phase-aware (circular) loss
# ---------------------------------------------------------------------------

class CircularPhaseLoss(nn.Module):
    """
    Phase loss that respects the circular topology of wrapped phase.

    Two complementary terms:
    1. **Cosine term** — 1 - cos(φ_pred - φ_gt)  (smooth, works globally)
    2. **Complex-L1 term** — L1 on the unit-circle representation
       |e^{iφ_pred} - e^{iφ_gt}|, which emphasises large errors more
       strongly than the cosine term alone.

    Both φ_pred and φ_gt are expected in *network space* ([-1, 1], i.e.
    divided by π). They are multiplied by π internally before computing
    trigonometric quantities.

    Args:
        cosine_weight  : Weight for the cosine term.
        complex_weight : Weight for the complex-L1 term.
        mask_weight    : Extra weight applied inside the masked region.
    """

    def __init__(
        self,
        cosine_weight: float = 1.0,
        complex_weight: float = 1.0,
        mask_weight: float = 2.0,
    ):
        super().__init__()
        self.cosine_w = cosine_weight
        self.complex_w = complex_weight
        self.mask_w = mask_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred   : Predicted phase (net-space, [-1, 1]), (B, 1, H, W).
            target : Ground-truth phase (net-space, [-1, 1]), (B, 1, H, W).
            mask   : Binary mask (1=valid, 0=missing), (B, 1, H, W).

        Returns:
            loss_valid   : Loss over the *valid* (unmasked) region.
            loss_masked  : Loss over the *masked* region (reconstruction target).
        """
        phi_pred = pred * torch.pi
        phi_gt = target * torch.pi
        diff = phi_pred - phi_gt

        # --- cosine term ---
        cos_term = 1.0 - torch.cos(diff)

        # --- complex-L1 term ---
        cpx_term = (
            (torch.sin(phi_pred) - torch.sin(phi_gt)).abs()
            + (torch.cos(phi_pred) - torch.cos(phi_gt)).abs()
        )

        per_pixel = (
            self.cosine_w * cos_term
            + self.complex_w * cpx_term
        )

        # Valid-region loss
        valid_px = mask.sum().clamp(min=1.0)
        loss_valid = (per_pixel * mask).sum() / valid_px

        # Masked-region loss (reconstruction target)
        inv_mask = 1.0 - mask
        missing_px = inv_mask.sum().clamp(min=1.0)
        loss_masked = self.mask_w * (per_pixel * inv_mask).sum() / missing_px

        return loss_valid, loss_masked


# ---------------------------------------------------------------------------
# Contour / edge loss
# ---------------------------------------------------------------------------

class ContourLoss(nn.Module):
    """
    Fringe-contour loss: weighted Focal-BCE + Soft Dice.

    Addresses three failure modes of plain BCE for contour connectivity:

    1. **Focal modulation** (γ=2) — down-weights easy correct predictions
       (confident background pixels) so gradients are dominated by *uncertain
       bridge pixels* at the mask boundary — exactly where fringe lines break.

    2. **Soft Dice** — overlap-based term; rewards predicting the full spatial
       extent of a fringe line, not just per-pixel accuracy.  A single broken
       pixel in a long line causes a bigger Dice drop than BCE.

    3. **Masked-first weighting** — the masked region (reconstruction target)
       is the primary optimisation target (``masked_weight``).  The valid
       region (pixels already known and copied at inference) contributes only a
       small regularisation signal (``valid_weight``), preventing it from
       dominating the gradient budget.

    Args:
        pos_weight   : Positive-class weight applied before focal modulation.
                       Amplifies the loss on sparse fringe pixels.
        focal_gamma  : Focal exponent γ.  0 = plain weighted BCE; 2 = standard
                       focal loss.
        dice_weight  : Relative weight of Soft Dice vs. Focal-BCE per region.
        valid_weight : Multiplier for the valid-region component (regulariser).
        masked_weight: Multiplier for the masked-region component (target).
    """

    def __init__(
        self,
        pos_weight: float = 5.0,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        valid_weight: float = 0.1,
        masked_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.valid_weight = valid_weight
        self.masked_weight = masked_weight

    @staticmethod
    def _soft_dice(
        pred: torch.Tensor,
        target: torch.Tensor,
        region: torch.Tensor,
    ) -> torch.Tensor:
        """Soft Dice loss restricted to a spatial region."""
        eps = 1e-6
        p = pred * region
        t = target * region
        intersection = (p * t).sum(dim=[1, 2, 3])
        union = p.sum(dim=[1, 2, 3]) + t.sum(dim=[1, 2, 3])
        return (1.0 - (2.0 * intersection + eps) / (union + eps)).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred   : Predicted contour map (sigmoid output, [0, 1]),
                     shape (B, 1, H, W).
            target : Ground-truth contour map ([0, 1]), shape (B, 1, H, W).
            mask   : Binary mask (1=valid, 0=missing), (B, 1, H, W).

        Returns:
            loss_valid : Focal-BCE + Dice over valid region (regulariser,
                         scaled by ``valid_weight``).
            loss_masked: Focal-BCE + Dice over masked region (primary target,
                         scaled by ``masked_weight``).
        """
        # --- per-pixel weighted Focal-BCE ---
        # F.binary_cross_entropy is numerically stable for sigmoid outputs
        bce_pp = F.binary_cross_entropy(pred, target, reduction="none")

        # Positive-class weight: amplify loss on fringe pixels
        pw = 1.0 + (self.pos_weight - 1.0) * target
        bce_pp = pw * bce_pp

        # Focal modulation: (1 - p_t)^γ down-weights easy correct predictions
        p_t = torch.where(target > 0.5, pred, 1.0 - pred)
        focal_pp = ((1.0 - p_t) ** self.focal_gamma) * bce_pp

        inv_mask = 1.0 - mask

        # --- valid region (regulariser) ---
        valid_px = mask.sum().clamp(min=1.0)
        focal_valid = (focal_pp * mask).sum() / valid_px
        dice_valid = self._soft_dice(pred, target, mask)
        loss_valid = self.valid_weight * (focal_valid + self.dice_weight * dice_valid)

        # --- masked region (primary target) ---
        missing_px = inv_mask.sum().clamp(min=1.0)
        focal_masked = (focal_pp * inv_mask).sum() / missing_px
        dice_masked = self._soft_dice(pred, target, inv_mask)
        loss_masked = self.masked_weight * (focal_masked + self.dice_weight * dice_masked)

        return loss_valid, loss_masked
