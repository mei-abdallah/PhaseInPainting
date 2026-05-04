"""
InSAR Interferogram Simulator.

Converts a synthetic DEM to a wrapped interferogram.  Optionally adds a
random linear deformation ramp to simulate earthquake-like LOS displacement:

    φ_total  = φ_topo + φ_defo
    φ_topo   = (4π / λ) · (B_⊥ · h) / (R · sin θ)
    φ_defo   = 2π · N · (cos θ_d · x̃ + sin θ_d · ỹ)

where x̃, ỹ ∈ [-0.5, 0.5] are normalised pixel coordinates, N is the
random fringe count, and θ_d is the random direction.  Wrapping the
combined phase produces the realistic mix of topographic and deformation
fringes seen in real earthquake interferograms.
"""

import numpy as np
from typing import Optional


class InSARSimulator:
    """
    Converts a synthetic DEM to a wrapped InSAR interferogram.

    The topographic phase contribution follows the standard two-pass formula:

        φ_topo = (4π / λ) · (B_⊥ · h) / (R · sin θ)

    where  λ = radar wavelength, B_⊥ = perpendicular baseline,
    h = elevation, R = range distance, θ = incidence angle.

    Only the topographic contribution is modelled; the result is wrapped to
    [-π, π].  Masking is handled externally by ``RealMaskLoader``.
    """

    def __init__(
        self,
        wavelength: float = 0.056,          # metres (C-band, Sentinel-1)
        incidence_angle: float = 38.0,      # degrees
        perp_baseline: float = 100.0,       # metres
        range_distance: float = 700_000.0,  # metres
    ):
        self.wavelength = wavelength
        self.incidence_angle = np.deg2rad(incidence_angle)
        self.perp_baseline = perp_baseline
        self.range_distance = range_distance

    # ------------------------------------------------------------------
    # Phase contributors
    # ------------------------------------------------------------------

    def dem_to_topo_phase(self, dem: np.ndarray) -> np.ndarray:
        """Compute topographic phase contribution (unwrapped, radians)."""
        return (
            (4.0 * np.pi / self.wavelength)
            * (self.perp_baseline * dem)
            / (self.range_distance * np.sin(self.incidence_angle))
        )

    @staticmethod
    def linear_deformation_phase(
        shape: tuple,
        rng: np.random.RandomState,
        min_fringes: float = 1.0,
        max_fringes: float = 3.0,
    ) -> np.ndarray:
        """
        Generate a random linear phase ramp simulating earthquake LOS displacement.

        Models the dominant signal in coseismic interferograms: a broad,
        smooth gradient going in a single direction (parallel to fault slip
        projected onto the LOS vector).  The ramp is parameterised directly
        in phase (radians) rather than physical displacement:

            φ_defo = 2π · N · (cos θ · x̃ + sin θ · ỹ)

        where x̃, ỹ ∈ [-0.5, 0.5] are normalised coordinates, N is a
        random fringe count, and θ is a random azimuth.

        Args:
            shape       : (H, W) of the output array.
            rng         : NumPy RandomState for reproducibility.
            min_fringes : Minimum number of fringe cycles across the image.
            max_fringes : Maximum number of fringe cycles across the image.

        Returns:
            Unwrapped deformation phase, float32, shape (H, W), in radians.
        """
        H, W = shape
        n_fringes = rng.uniform(min_fringes, max_fringes)
        direction = rng.uniform(0.0, 2.0 * np.pi)

        yy = np.linspace(-0.5, 0.5, H)[:, None]   # (H, 1)
        xx = np.linspace(-0.5, 0.5, W)[None, :]   # (1, W)

        ramp = 2.0 * np.pi * n_fringes * (
            np.cos(direction) * xx + np.sin(direction) * yy
        )
        return ramp.astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @staticmethod
    def wrap_phase(phase: np.ndarray) -> np.ndarray:
        """Wrap any phase array to [-π, π]."""
        return np.angle(np.exp(1j * phase))

    def generate_interferogram(
        self,
        dem: np.ndarray,
        rng: Optional[np.random.RandomState] = None,
        deformation: bool = True,
    ) -> np.ndarray:
        """
        Generate a wrapped interferogram from a DEM.

        Args:
            dem         : 2-D elevation array in metres.
            rng         : RandomState for the deformation ramp.  If None,
                          no deformation is added.
            deformation : If True and rng is provided, add a random linear
                          deformation ramp before wrapping.

        Returns:
            wrapped_phase : float32, shape (H, W), values in [-π, π].
        """
        phase = self.dem_to_topo_phase(dem)
        if deformation and rng is not None:
            phase = phase + self.linear_deformation_phase(dem.shape, rng)
        wrapped = self.wrap_phase(phase)
        return wrapped.astype(np.float32)
