"""
Heatmap-based Landmark Detection Methods
Comprehensive tutorial on different heatmap regression approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import cv2


class HeatmapLossFunctions:
    """Collection of loss functions for heatmap regression"""

    @staticmethod
    def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error (L2) Loss

        Args:
            pred: Predicted heatmaps [B, C, H, W]
            target: Ground truth heatmaps [B, C, H, W]

        Returns:
            MSE loss
        """
        return F.mse_loss(pred, target)

    @staticmethod
    def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Smooth L1 Loss (Huber Loss)

        Better than MSE for heatmap regression according to research

        Args:
            pred: Predicted heatmaps
            target: Ground truth heatmaps

        Returns:
            Smooth L1 loss
        """
        return F.smooth_l1_loss(pred, target)

    @staticmethod
    def wing_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        omega: float = 10.0,
        epsilon: float = 2.0
    ) -> torch.Tensor:
        """
        Wing Loss - Designed for direct coordinate regression
        WARNING: Not recommended for heatmap regression (causes divergence)

        Args:
            pred: Predictions
            target: Ground truth
            omega: Width parameter
            epsilon: Curvature parameter

        Returns:
            Wing loss
        """
        delta = (pred - target).abs()

        # Threshold calculation
        C = omega - omega * torch.log(torch.tensor(1.0 + omega / epsilon))

        loss = torch.where(
            delta < omega,
            omega * torch.log(1.0 + delta / epsilon),
            delta - C
        )

        return loss.mean()

    @staticmethod
    def adaptive_wing_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1
    ) -> torch.Tensor:
        """
        Adaptive Wing Loss - State-of-the-art for heatmap regression

        Key innovation: Adapts behavior based on ground truth pixel values
        - Foreground pixels (target ≈ 1): Sensitive to small errors
        - Background pixels (target ≈ 0): Tolerates small errors

        Args:
            pred: Predicted heatmaps [B, C, H, W]
            target: Ground truth heatmaps [B, C, H, W]
            omega: Controls influence of small errors
            theta: Threshold between regions
            epsilon: Scaling factor
            alpha: Adaptation power term

        Returns:
            Adaptive Wing loss
        """
        delta = (pred - target).abs()

        # Calculate A for continuity
        A = omega * (1.0 / (1.0 + torch.pow(theta / epsilon, alpha - target))) * \
            (alpha - target) * torch.pow(theta / epsilon, alpha - target - 1.0) / epsilon

        # Calculate C for smoothness
        C = theta * A - omega * torch.log(1.0 + torch.pow(theta / epsilon, alpha - target))

        # Adaptive wing loss
        loss = torch.where(
            delta < theta,
            omega * torch.log(1.0 + torch.pow(delta / epsilon, alpha - target)),
            A * delta - C
        )

        return loss.mean()


class GaussianHeatmapGenerator:
    """Generate various types of Gaussian heatmaps"""

    def __init__(self, sigma: float = 2.0, heatmap_size: Tuple[int, int] = (64, 64)):
        """
        Initialize heatmap generator

        Args:
            sigma: Standard deviation of Gaussian
            heatmap_size: Output heatmap dimensions
        """
        self.sigma = sigma
        self.heatmap_size = heatmap_size

    def generate_isotropic_gaussian(
        self,
        center_x: float,
        center_y: float
    ) -> np.ndarray:
        """
        Generate standard isotropic Gaussian heatmap

        Args:
            center_x: X coordinate of center
            center_y: Y coordinate of center

        Returns:
            Gaussian heatmap [H, W]
        """
        # Create coordinate grids
        x = np.arange(0, self.heatmap_size[1], 1, dtype=np.float32)
        y = np.arange(0, self.heatmap_size[0], 1, dtype=np.float32)
        x, y = np.meshgrid(x, y)

        # Calculate Gaussian
        gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * self.sigma**2))

        # Normalize
        if gaussian.max() > 0:
            gaussian = gaussian / gaussian.max()

        return gaussian

    def generate_anisotropic_gaussian(
        self,
        center_x: float,
        center_y: float,
        sigma_x: float,
        sigma_y: float,
        theta: float = 0.0
    ) -> np.ndarray:
        """
        Generate anisotropic Gaussian with rotation

        Useful for elongated landmarks or directional features

        Args:
            center_x: X coordinate
            center_y: Y coordinate
            sigma_x: Std dev in x direction
            sigma_y: Std dev in y direction
            theta: Rotation angle in radians

        Returns:
            Anisotropic Gaussian heatmap
        """
        x = np.arange(0, self.heatmap_size[1], 1, dtype=np.float32)
        y = np.arange(0, self.heatmap_size[0], 1, dtype=np.float32)
        x, y = np.meshgrid(x, y)

        # Translate to center
        x_shifted = x - center_x
        y_shifted = y - center_y

        # Rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Rotate coordinates
        x_rot = cos_theta * x_shifted + sin_theta * y_shifted
        y_rot = -sin_theta * x_shifted + cos_theta * y_shifted

        # Anisotropic Gaussian
        gaussian = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))

        # Normalize
        if gaussian.max() > 0:
            gaussian = gaussian / gaussian.max()

        return gaussian

    def visualize_sigma_effects(self, center: Tuple[float, float] = None):
        """
        Visualize effect of different sigma values

        Args:
            center: Center point for Gaussian
        """
        if center is None:
            center = (self.heatmap_size[1] // 2, self.heatmap_size[0] // 2)

        sigma_values = [1.0, 2.0, 3.0, 5.0]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for idx, sigma in enumerate(sigma_values):
            self.sigma = sigma
            heatmap = self.generate_isotropic_gaussian(center[0], center[1])

            axes[idx].imshow(heatmap, cmap='jet')
            axes[idx].set_title(f'σ = {sigma}')
            axes[idx].axis('off')

        plt.suptitle('Effect of Sigma on Gaussian Heatmap')
        plt.tight_layout()
        plt.savefig('gaussian_sigma_comparison.png', dpi=150)
        print("Saved: gaussian_sigma_comparison.png")


class SubPixelCoordinateExtraction:
    """Various methods for extracting coordinates from heatmaps"""

    @staticmethod
    def argmax(heatmap: np.ndarray) -> Tuple[float, float]:
        """
        Simple argmax - integer precision only

        Args:
            heatmap: Input heatmap [H, W]

        Returns:
            (x, y) coordinates
        """
        idx = heatmap.argmax()
        y, x = np.unravel_index(idx, heatmap.shape)
        return (float(x), float(y))

    @staticmethod
    def soft_argmax_local(
        heatmap: np.ndarray,
        temperature: float = 10.0,
        patch_size: int = 5
    ) -> Tuple[float, float]:
        """
        Local soft-argmax with temperature scaling

        Args:
            heatmap: Input heatmap [H, W]
            temperature: Temperature parameter (higher = sharper)
            patch_size: Size of local patch around maximum

        Returns:
            (x, y) coordinates with sub-pixel precision
        """
        # Find maximum
        max_idx = heatmap.argmax()
        max_y, max_x = np.unravel_index(max_idx, heatmap.shape)

        # Extract local patch
        half_size = patch_size // 2
        y_min = max(0, max_y - half_size)
        y_max = min(heatmap.shape[0], max_y + half_size + 1)
        x_min = max(0, max_x - half_size)
        x_max = min(heatmap.shape[1], max_x + half_size + 1)

        patch = heatmap[y_min:y_max, x_min:x_max]

        # Apply temperature-scaled softmax
        patch_flat = patch.flatten()
        softmax_vals = np.exp(temperature * patch_flat)
        softmax_vals = softmax_vals / (softmax_vals.sum() + 1e-8)

        # Coordinate grids
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]

        # Weighted average
        x_coord = np.sum(x_coords.flatten() * softmax_vals)
        y_coord = np.sum(y_coords.flatten() * softmax_vals)

        return (x_coord, y_coord)

    @staticmethod
    def soft_argmax_global(heatmap: torch.Tensor) -> Tuple[float, float]:
        """
        Global soft-argmax (differentiable)

        Args:
            heatmap: Input heatmap as torch.Tensor [H, W]

        Returns:
            (x, y) coordinates
        """
        H, W = heatmap.shape

        # Create coordinate grids
        x_coords = torch.arange(W, dtype=torch.float32).view(1, -1).expand(H, -1)
        y_coords = torch.arange(H, dtype=torch.float32).view(-1, 1).expand(-1, W)

        # Flatten and apply softmax
        heatmap_flat = heatmap.view(-1)
        probs = F.softmax(heatmap_flat, dim=0).view(H, W)

        # Expected value
        x_coord = torch.sum(x_coords * probs)
        y_coord = torch.sum(y_coords * probs)

        return (x_coord.item(), y_coord.item())

    @staticmethod
    def center_of_mass(heatmap: np.ndarray, threshold: float = 0.1) -> Tuple[float, float]:
        """
        Center of mass method

        Args:
            heatmap: Input heatmap [H, W]
            threshold: Relative threshold (0-1)

        Returns:
            (x, y) coordinates
        """
        # Threshold heatmap
        threshold_val = threshold * heatmap.max()
        mask = heatmap > threshold_val

        if mask.sum() == 0:
            # Fallback to argmax
            return SubPixelCoordinateExtraction.argmax(heatmap)

        # Calculate center of mass
        y_coords, x_coords = np.where(mask)
        weights = heatmap[mask]

        x_com = np.sum(x_coords * weights) / weights.sum()
        y_com = np.sum(y_coords * weights) / weights.sum()

        return (x_com, y_com)

    @staticmethod
    def gaussian_fitting(heatmap: np.ndarray) -> Tuple[float, float]:
        """
        Fit 2D Gaussian to heatmap peak

        Args:
            heatmap: Input heatmap [H, W]

        Returns:
            (x, y) coordinates of Gaussian center
        """
        from scipy.optimize import curve_fit

        # Find approximate center via argmax
        max_idx = heatmap.argmax()
        y0, x0 = np.unravel_index(max_idx, heatmap.shape)

        # Extract region around peak
        size = 11
        half = size // 2
        y_min = max(0, y0 - half)
        y_max = min(heatmap.shape[0], y0 + half + 1)
        x_min = max(0, x0 - half)
        x_max = min(heatmap.shape[1], x0 + half + 1)

        patch = heatmap[y_min:y_max, x_min:x_max]

        # Create coordinate grids
        x = np.arange(x_min, x_max)
        y = np.arange(y_min, y_max)
        x, y = np.meshgrid(x, y)

        # Define 2D Gaussian function
        def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y):
            x, y = xy
            g = amplitude * np.exp(-(((x-xo)**2)/(2*sigma_x**2) + ((y-yo)**2)/(2*sigma_y**2)))
            return g.ravel()

        # Initial guess
        initial_guess = (heatmap.max(), x0, y0, 2.0, 2.0)

        try:
            # Fit Gaussian
            popt, _ = curve_fit(
                gaussian_2d,
                (x, y),
                patch.ravel(),
                p0=initial_guess,
                maxfev=1000
            )

            return (popt[1], popt[2])  # (xo, yo)
        except:
            # Fallback to soft-argmax
            return SubPixelCoordinateExtraction.soft_argmax_local(heatmap)


def compare_extraction_methods():
    """Compare different coordinate extraction methods"""
    print("="*60)
    print("Coordinate Extraction Methods Comparison")
    print("="*60)

    # Generate test heatmap
    generator = GaussianHeatmapGenerator(sigma=2.5, heatmap_size=(64, 64))
    true_x, true_y = 32.7, 31.3  # Sub-pixel ground truth

    heatmap = generator.generate_isotropic_gaussian(true_x, true_y)

    # Add some noise
    noise = np.random.normal(0, 0.01, heatmap.shape)
    heatmap = np.clip(heatmap + noise, 0, 1)

    extractor = SubPixelCoordinateExtraction()

    # Test all methods
    methods = {
        'Argmax': extractor.argmax,
        'Soft-Argmax (Local)': extractor.soft_argmax_local,
        'Center of Mass': extractor.center_of_mass,
        'Gaussian Fitting': extractor.gaussian_fitting
    }

    results = []

    print(f"\nGround Truth: x={true_x:.2f}, y={true_y:.2f}\n")

    for method_name, method_func in methods.items():
        try:
            x_pred, y_pred = method_func(heatmap)
            error = np.sqrt((x_pred - true_x)**2 + (y_pred - true_y)**2)

            results.append({
                'method': method_name,
                'x': x_pred,
                'y': y_pred,
                'error': error
            })

            print(f"{method_name:25s}: x={x_pred:6.2f}, y={y_pred:6.2f} | Error: {error:.3f} pixels")
        except Exception as e:
            print(f"{method_name:25s}: Failed ({e})")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap
    axes[0].imshow(heatmap, cmap='jet')
    axes[0].plot(true_x, true_y, 'w*', markersize=15, label='Ground Truth')

    for result in results:
        axes[0].plot(result['x'], result['y'], 'o', label=result['method'], markersize=8)

    axes[0].set_title('Heatmap with Predictions')
    axes[0].legend(fontsize=8)
    axes[0].axis('off')

    # Error comparison
    errors = [r['error'] for r in results]
    method_names = [r['method'] for r in results]

    axes[1].barh(method_names, errors)
    axes[1].set_xlabel('Error (pixels)')
    axes[1].set_title('Coordinate Extraction Error')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('coordinate_extraction_comparison.png', dpi=150)
    print("\n✓ Saved: coordinate_extraction_comparison.png")

    print("\n" + "="*60)


def demo_loss_functions():
    """Demonstrate different loss functions"""
    print("\n" + "="*60)
    print("Loss Functions Comparison")
    print("="*60)

    # Create sample predictions and targets
    batch_size = 4
    num_landmarks = 3
    heatmap_size = 64

    torch.manual_seed(42)

    # Simulated predictions (slightly noisy)
    target = torch.rand(batch_size, num_landmarks, heatmap_size, heatmap_size)
    pred = target + torch.randn_like(target) * 0.1
    pred = torch.clamp(pred, 0, 1)

    loss_funcs = HeatmapLossFunctions()

    # Calculate all losses
    losses = {
        'MSE': loss_funcs.mse_loss(pred, target),
        'Smooth L1': loss_funcs.smooth_l1_loss(pred, target),
        'Adaptive Wing': loss_funcs.adaptive_wing_loss(pred, target)
    }

    print("\nLoss Values:")
    for name, value in losses.items():
        print(f"  {name:20s}: {value.item():.6f}")

    print("\n✓ Adaptive Wing Loss is recommended for heatmap regression")
    print("  - Sensitive to small errors on foreground pixels")
    print("  - Tolerates small errors on background pixels")
    print("  - Smooth gradients, stable training")

    print("\n" + "="*60)


def main():
    """Main demo"""
    # Visualize sigma effects
    generator = GaussianHeatmapGenerator()
    generator.visualize_sigma_effects()

    # Compare extraction methods
    compare_extraction_methods()

    # Demo loss functions
    demo_loss_functions()

    print("\n🎉 All demos completed!")


if __name__ == "__main__":
    main()
