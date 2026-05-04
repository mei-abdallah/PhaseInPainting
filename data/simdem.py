"""
DEM Simulator for Topographic Phase Generation.

Generates synthetic Digital Elevation Models using fractal-based methods
(fBm / diamond-square) with optional mountain-peak features.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter


class DEMSimulator:
    """
    Generates synthetic DEMs and calculates topographic phase.
    
    Creates realistic terrain using fractal-based methods (fBm - fractional 
    Brownian motion) combined with coherent structures to simulate mountains,
    valleys, and other topographic features.
    """
    
    def __init__(
        self,
        shape: Tuple[int, int] = (512, 512),
        pixel_size: float = 90.0,  # meters
        max_elevation: Optional[float] = None,
        min_elevation: float = 0.0,
        roughness: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize DEM simulator.

        Args:
            shape: (height, width) of output DEM in pixels
            pixel_size: pixel size in meters
            max_elevation: maximum elevation in meters.  If None (default),
                auto-scaled from pixel_size × max(shape): larger images cover
                a larger physical area and therefore exhibit a greater total
                elevation range.  Calibrated so that 256×256 at 90 m/px
                gives ≈ 2000 m, 512×512 gives ≈ 4000 m.
            min_elevation: minimum elevation in meters
            roughness: terrain roughness (0-1), higher = rougher
            seed: random seed for reproducibility
        """
        self.shape = shape
        self.pixel_size = pixel_size
        self.max_elevation = (
            max_elevation if max_elevation is not None
            else pixel_size * max(shape) * 0.043
        )
        self.min_elevation = min_elevation
        self.roughness = roughness
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_fractal_dem(self, hurst: float = 0.9) -> np.ndarray:
        """
        Generate DEM using spectral synthesis (fBm).
        
        Uses the Fourier filtering method to generate fractal terrain
        with controllable roughness through the Hurst exponent.
        
        Args:
            hurst: Hurst exponent (0-1). Higher values = smoother terrain
            
        Returns:
            2D array of elevation values in meters
        """
        ny, nx = self.shape
        
        # Generate random phase
        phase = np.random.uniform(0, 2*np.pi, (ny, nx))
        
        # Create frequency grids
        fx = np.fft.fftfreq(nx)
        fy = np.fft.fftfreq(ny)
        FX, FY = np.meshgrid(fx, fy)
        
        # Frequency magnitude (avoiding division by zero)
        freq_mag = np.sqrt(FX**2 + FY**2)
        freq_mag[0, 0] = 1.0
        
        # Power law spectrum for fBm
        # P(f) ~ f^(-beta) where beta = 2H + 1 for 2D
        beta = 2 * hurst + 1
        amplitude = freq_mag ** (-beta / 2)
        amplitude[0, 0] = 0  # No DC component
        
        # Generate complex spectrum
        spectrum = amplitude * np.exp(1j * phase)
        
        # Inverse FFT to get terrain
        dem = np.real(np.fft.ifft2(spectrum))
        
        # Normalize to elevation range
        dem = (dem - dem.min()) / (dem.max() - dem.min())
        dem = dem * (self.max_elevation - self.min_elevation) + self.min_elevation
        
        return dem
    
    def generate_diamond_square_dem(self) -> np.ndarray:
        """
        Generate DEM using diamond-square algorithm.
        
        Alternative method for terrain generation that creates 
        more natural-looking terrain with self-similar properties.
        
        Returns:
            2D array of elevation values in meters
        """
        # Ensure size is 2^n + 1
        size = max(self.shape)
        n = int(np.ceil(np.log2(size - 1)))
        grid_size = 2**n + 1
        
        dem = np.zeros((grid_size, grid_size))
        
        # Initialize corners
        dem[0, 0] = np.random.uniform(0, 1)
        dem[0, -1] = np.random.uniform(0, 1)
        dem[-1, 0] = np.random.uniform(0, 1)
        dem[-1, -1] = np.random.uniform(0, 1)
        
        step_size = grid_size - 1
        scale = self.roughness
        
        while step_size > 1:
            half = step_size // 2
            
            # Diamond step
            for y in range(half, grid_size - 1, step_size):
                for x in range(half, grid_size - 1, step_size):
                    avg = (dem[y - half, x - half] + 
                           dem[y - half, x + half] +
                           dem[y + half, x - half] + 
                           dem[y + half, x + half]) / 4.0
                    dem[y, x] = avg + np.random.uniform(-scale, scale)
            
            # Square step
            for y in range(0, grid_size, half):
                for x in range((y + half) % step_size, grid_size, step_size):
                    count = 0
                    total = 0
                    
                    if y >= half:
                        total += dem[y - half, x]
                        count += 1
                    if y + half < grid_size:
                        total += dem[y + half, x]
                        count += 1
                    if x >= half:
                        total += dem[y, x - half]
                        count += 1
                    if x + half < grid_size:
                        total += dem[y, x + half]
                        count += 1
                        
                    dem[y, x] = total / count + np.random.uniform(-scale, scale)
            
            step_size = half
            scale *= 0.5
        
        # Resize to desired shape
        from scipy.ndimage import zoom
        zoom_factors = (self.shape[0] / grid_size, self.shape[1] / grid_size)
        dem = zoom(dem, zoom_factors, order=3)
        
        # Normalize to elevation range
        dem = (dem - dem.min()) / (dem.max() - dem.min())
        dem = dem * (self.max_elevation - self.min_elevation) + self.min_elevation
        
        return dem
    
    def add_mountain_features(
        self,
        dem: np.ndarray,
        num_peaks: int = 3,
        peak_height_range: Tuple[float, float] = (500, 1500),
        peak_width_range: Tuple[float, float] = (50, 150)
    ) -> np.ndarray:
        """
        Add mountain peaks/ridges to existing DEM.
        
        Args:
            dem: Base DEM array
            num_peaks: Number of peaks to add
            peak_height_range: (min, max) height of peaks in meters
            peak_width_range: (min, max) width of peaks in pixels
            
        Returns:
            DEM with added mountain features
        """
        ny, nx = dem.shape
        result = dem.copy()
        
        for _ in range(num_peaks):
            # Random peak location
            cy = np.random.randint(0, ny)
            cx = np.random.randint(0, nx)
            
            # Random height and width
            height = np.random.uniform(*peak_height_range)
            width = np.random.uniform(*peak_width_range)
            
            # Create Gaussian peak
            y, x = np.ogrid[:ny, :nx]
            peak = height * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * width**2))
            
            result += peak
        
        return result
    
    def generate_dem(
        self,
        method: str = 'fractal',
        add_features: bool = True,
        smooth_sigma: float = 6.0,
    ) -> np.ndarray:
        """
        Generate a synthetic DEM.

        Args:
            method: 'fractal' (spectral fBm) or 'diamond_square'
            add_features: randomly add mountain peaks (50 % chance)
            smooth_sigma: Gaussian smoothing sigma in pixels for realism

        Returns:
            2D DEM array in metres
        """
        if method == 'fractal':
            dem = self.generate_fractal_dem()
        else:
            dem = self.generate_diamond_square_dem()

        if add_features and np.random.random() > 0.5:
            dem = self.add_mountain_features(dem)

        if smooth_sigma > 0:
            dem = gaussian_filter(dem, sigma=smooth_sigma)

        return dem
