"""
Neural-network building blocks for InSAR-EdgeConnect.

Contents
--------
Contour / fringe-line detector
    InSARVGGCanny     — Frozen VGG19 fringe-line detector.
                        predict(phase_norm) → numpy edge map.
                        Input: float32 numpy (H,W), normalised phase φ/π ∈ [-1,1].

Generator / discriminator
    UNetGenerator     — Shared encoder-bottleneck-decoder backbone.
    PatchDiscriminator — 70×70 PatchGAN with spectral normalisation.
    build_contour_generator, build_phase_generator, build_discriminator
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torchvision.models as tv_models
import numpy as np


# ===========================================================================
# Fringe-line detector
# ===========================================================================

class InSARVGGCanny(nn.Module):
    """
    Frozen VGG19 fringe-line detector for InSAR phase.

    Takes normalised phase φ/π ∈ [-1, 1] (single channel), expands it to
    3 channels with a frozen all-ones 1×1 conv, then runs VGG19 features
    and thresholds one channel to produce a binary edge map.

    Args:
        feature   : Number of VGG19 feature layers to use (default=4).
        threshold : Activation threshold for the binary edge map.
        layer     : Channel index of the VGG feature map to threshold.
    """

    def __init__(
        self,
        feature: int = 4,
        threshold: float = 0.5,
        layer: int = 49,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 1, bias=False)
        self.conv.weight = nn.Parameter(torch.ones_like(self.conv.weight))

        self.features = tv_models.vgg19(weights="DEFAULT").eval().features[:feature]
        self.threshold = threshold
        self.layer = layer

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, phase_norm: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) normalised phase → VGG feature map (B,C,H,W)."""
        return self.features(self.conv(phase_norm))

    def predict(self, phase_norm: np.ndarray) -> np.ndarray:
        """(H,W) normalised phase φ/π ∈ [-1,1] → binary edge map (H,W) float32."""
        t = torch.from_numpy(phase_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            feat = self(t)
        return (feat[:, self.layer: self.layer + 1] > self.threshold).float().squeeze(0).squeeze(0).numpy()


# ===========================================================================
# Shared helpers
# ===========================================================================

def _sn_conv(
    in_ch: int, out_ch: int,
    kernel: int = 4, stride: int = 2, padding: int = 1, bias: bool = False,
) -> nn.Conv2d:
    return spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=bias))


def _sn_conv_same(
    in_ch: int, out_ch: int,
    kernel: int = 3, stride: int = 1, padding: int = 1, bias: bool = False,
) -> nn.Conv2d:
    return spectral_norm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=bias))


# ===========================================================================
# Generator building blocks
# ===========================================================================

class ResBlock(nn.Module):
    """Residual block with spectral norm + Instance Norm."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(channels, channels, 3, 1,
                          padding=pad, dilation=dilation, bias=False)
            ),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            spectral_norm(
                nn.Conv2d(channels, channels, 3, 1,
                          padding=pad, dilation=dilation, bias=False)
            ),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class EncoderBlock(nn.Module):
    """Strided conv → InstanceNorm → LeakyReLU (halves spatial size)."""

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers = [_sn_conv(in_ch, out_ch)]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Bilinear upsample → conv → InstanceNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _sn_conv_same(in_ch, out_ch),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ===========================================================================
# U-Net Generator
# ===========================================================================


class UNetGenerator(nn.Module):
    """
    U-Net generator with residual bottleneck and spectral normalisation.

    Input  : (B, in_ch,  H, W)  — 3 channels for both stages
    Output : (B, out_ch, H, W)  — 1 channel

    Args:
        in_ch   : Input channels (3 = [signal, contour/phase, mask]).
        out_ch  : Output channels (1).
        base_ch : Base filter count, doubles at each encoder level.
        n_res   : Residual blocks in the bottleneck.
        out_act : 'tanh' for phase inpainting, 'sigmoid' for contours.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        base_ch: int = 64,
        n_res: int = 8,
        out_act: str = "tanh",
    ):
        super().__init__()

        # ---------- Encoder ----------
        self.enc1 = EncoderBlock(in_ch,       base_ch,     use_norm=False)  # /2
        self.enc2 = EncoderBlock(base_ch,     base_ch * 2)                  # /4
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)                  # /8
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)                  # /16
        self.enc5 = EncoderBlock(base_ch * 8, base_ch * 8)                  # /32

        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            *[ResBlock(base_ch * 8, dilation=2 ** (i % 3)) for i in range(n_res)]
        )

        # ---------- Decoder (skip connections) ----------
        # Skip channels mirror the corresponding encoder output channels.
        self.dec5 = DecoderBlock(base_ch * 8 + base_ch * 8, base_ch * 8, use_dropout=True)
        self.dec4 = DecoderBlock(base_ch * 8 + base_ch * 8, base_ch * 4, use_dropout=True)
        self.dec3 = DecoderBlock(base_ch * 4 + base_ch * 4, base_ch * 2)
        self.dec2 = DecoderBlock(base_ch * 2 + base_ch * 2, base_ch)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _sn_conv_same(base_ch + base_ch, out_ch),
        )

        if out_act == "tanh":
            self.out_act: nn.Module = nn.Tanh()
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)

        d5 = self.dec5(torch.cat([b,  e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))

        return self.out_act(out)


# ===========================================================================
# PatchGAN Discriminator
# ===========================================================================

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator with spectral normalisation.

    Classifies overlapping patches as real / fake to enforce local
    sharpness and structural coherence.

    Args:
        in_ch   : Input channels (real/fake signal + conditioning).
        base_ch : Base filter count.
        n_layers: Number of strided conv layers.
    """

    def __init__(self, in_ch: int = 2, base_ch: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [
            _sn_conv(in_ch, base_ch, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base_ch
        for _ in range(n_layers):
            out_ch = min(ch * 2, 512)
            layers += [
                _sn_conv(ch, out_ch),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch

        layers.append(_sn_conv(ch, 1, kernel=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Discriminator with intermediate feature extraction
# ---------------------------------------------------------------------------

class DiscriminatorWithFeatures(nn.Module):
    """
    Wraps a PatchDiscriminator so that it returns both the final logit map
    *and* the list of intermediate feature maps.  Feature maps are used for
    the feature-matching loss.
    """

    def __init__(self, discriminator: nn.Module):
        super().__init__()
        # Split the sequential model into individual layers
        self.layers = nn.ModuleList(discriminator.model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        features = []
        out = x
        for layer in self.layers:
            out = layer(out)
            features.append(out)
        # features[-1] is the logit map; all others are intermediate features
        return features[-1], features[:-1]

# ===========================================================================
# Named constructors
# ===========================================================================

def build_contour_generator(base_ch: int = 64, n_res: int = 8) -> UNetGenerator:
    """Stage-1 generator: predicts fringe-contour map in masked regions."""
    return UNetGenerator(in_ch=3, out_ch=1, base_ch=base_ch,
                         n_res=n_res, out_act="sigmoid")


def build_phase_generator(base_ch: int = 64, n_res: int = 8) -> UNetGenerator:
    """Stage-2 generator: inpaints phase values in masked regions."""
    return UNetGenerator(in_ch=3, out_ch=1, base_ch=base_ch,
                         n_res=n_res, out_act="tanh")


def build_discriminator(in_ch: int = 2) -> PatchDiscriminator:
    """PatchGAN discriminator for either the contour or phase branch."""
    return PatchDiscriminator(in_ch=in_ch)
