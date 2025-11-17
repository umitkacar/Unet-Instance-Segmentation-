"""
Medical Landmark Detection using U-Net and Heatmap Regression
Application: Cephalometric landmark detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """Generate Gaussian heatmaps for landmark annotations"""

    def __init__(self, sigma: float = 2.0, heatmap_size: Tuple[int, int] = (64, 64)):
        """
        Initialize heatmap generator

        Args:
            sigma: Standard deviation for Gaussian distribution
            heatmap_size: Size of output heatmap (H, W)
        """
        self.sigma = sigma
        self.heatmap_size = heatmap_size

    def generate_heatmap(
        self,
        landmarks: List[Tuple[float, float]],
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generate multi-channel Gaussian heatmap

        Args:
            landmarks: List of (x, y) coordinates (normalized 0-1)
            image_size: Original image size (H, W)

        Returns:
            Heatmap array of shape [num_landmarks, H, W]
        """
        num_landmarks = len(landmarks)
        heatmaps = np.zeros((num_landmarks, *self.heatmap_size), dtype=np.float32)

        h_ratio = self.heatmap_size[0] / image_size[0]
        w_ratio = self.heatmap_size[1] / image_size[1]

        for idx, (x, y) in enumerate(landmarks):
            # Convert to heatmap coordinates
            hm_x = int(x * image_size[1] * w_ratio)
            hm_y = int(y * image_size[0] * h_ratio)

            # Generate 2D Gaussian
            heatmap = self._generate_2d_gaussian(hm_x, hm_y)
            heatmaps[idx] = heatmap

        return heatmaps

    def _generate_2d_gaussian(self, center_x: int, center_y: int) -> np.ndarray:
        """
        Generate 2D Gaussian distribution

        Args:
            center_x: X coordinate of Gaussian center
            center_y: Y coordinate of Gaussian center

        Returns:
            2D Gaussian heatmap
        """
        # Create coordinate grids
        x = np.arange(0, self.heatmap_size[1], 1, dtype=np.float32)
        y = np.arange(0, self.heatmap_size[0], 1, dtype=np.float32)
        x, y = np.meshgrid(x, y)

        # Calculate Gaussian
        gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * self.sigma**2))

        # Normalize to [0, 1]
        gaussian = gaussian / (gaussian.max() + 1e-8)

        return gaussian


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression
    Paper: "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    """

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1
    ):
        """
        Initialize Adaptive Wing Loss

        Args:
            omega: Controls influence of small errors
            theta: Threshold between two loss regions
            epsilon: Scaling factor
            alpha: Power term for adaptation
        """
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Adaptive Wing Loss

        Args:
            pred: Predicted heatmaps [B, C, H, W]
            target: Ground truth heatmaps [B, C, H, W]

        Returns:
            Loss value
        """
        delta = (pred - target).abs()

        # Calculate A and C for continuity
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        # Adaptive wing loss
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )

        return loss.mean()


class LandmarkUNet(nn.Module):
    """U-Net for landmark detection via heatmap regression"""

    def __init__(
        self,
        num_landmarks: int,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet"
    ):
        """
        Initialize Landmark U-Net

        Args:
            num_landmarks: Number of landmarks to detect
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights
        """
        super().__init__()

        self.num_landmarks = num_landmarks

        # U-Net backbone
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,  # Grayscale medical images
            classes=num_landmarks,  # One channel per landmark
            activation=None  # We'll apply sigmoid manually
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input image [B, 1, H, W]

        Returns:
            Heatmaps [B, num_landmarks, H, W]
        """
        heatmaps = self.unet(x)
        heatmaps = torch.sigmoid(heatmaps)  # Ensure [0, 1] range
        return heatmaps


class SubPixelLandmarkExtractor:
    """Extract landmark coordinates from heatmaps with sub-pixel accuracy"""

    def __init__(self, method: str = 'soft_argmax', temperature: float = 10.0):
        """
        Initialize landmark extractor

        Args:
            method: 'argmax', 'soft_argmax', or 'center_of_mass'
            temperature: Temperature parameter for soft-argmax
        """
        self.method = method
        self.temperature = temperature

    def extract_landmarks(self, heatmaps: np.ndarray) -> List[Tuple[float, float]]:
        """
        Extract landmark coordinates from heatmaps

        Args:
            heatmaps: Heatmap array [num_landmarks, H, W]

        Returns:
            List of (x, y) coordinates
        """
        landmarks = []

        for heatmap in heatmaps:
            if self.method == 'argmax':
                y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
                landmarks.append((float(x), float(y)))

            elif self.method == 'soft_argmax':
                x, y = self._soft_argmax(heatmap)
                landmarks.append((x, y))

            elif self.method == 'center_of_mass':
                x, y = self._center_of_mass(heatmap)
                landmarks.append((x, y))

        return landmarks

    def _soft_argmax(self, heatmap: np.ndarray) -> Tuple[float, float]:
        """
        Soft-argmax with temperature scaling

        Args:
            heatmap: Single heatmap [H, W]

        Returns:
            (x, y) coordinates with sub-pixel precision
        """
        # Find maximum location
        max_idx = heatmap.argmax()
        max_y, max_x = np.unravel_index(max_idx, heatmap.shape)

        # Extract local patch (e.g., 5x5 around maximum)
        patch_size = 5
        half_size = patch_size // 2

        y_min = max(0, max_y - half_size)
        y_max = min(heatmap.shape[0], max_y + half_size + 1)
        x_min = max(0, max_x - half_size)
        x_max = min(heatmap.shape[1], max_x + half_size + 1)

        patch = heatmap[y_min:y_max, x_min:x_max]

        # Apply temperature-scaled softmax
        patch_flat = patch.flatten()
        softmax_vals = np.exp(self.temperature * patch_flat)
        softmax_vals = softmax_vals / (softmax_vals.sum() + 1e-8)

        # Create coordinate grids for the patch
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]

        # Compute weighted average
        x_coord = np.sum(x_coords.flatten() * softmax_vals)
        y_coord = np.sum(y_coords.flatten() * softmax_vals)

        return (x_coord, y_coord)

    def _center_of_mass(self, heatmap: np.ndarray) -> Tuple[float, float]:
        """
        Center of mass method

        Args:
            heatmap: Single heatmap [H, W]

        Returns:
            (x, y) coordinates
        """
        # Threshold to remove noise
        threshold = 0.1 * heatmap.max()
        mask = heatmap > threshold

        if mask.sum() == 0:
            # Fallback to argmax
            y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
            return (float(x), float(y))

        # Calculate center of mass
        y_coords, x_coords = np.where(mask)
        weights = heatmap[mask]

        x_com = np.sum(x_coords * weights) / weights.sum()
        y_com = np.sum(y_coords * weights) / weights.sum()

        return (x_com, y_com)


class MedicalLandmarkDetector:
    """Complete pipeline for medical landmark detection"""

    def __init__(
        self,
        model_path: str,
        num_landmarks: int = 19,
        device: str = 'cuda'
    ):
        """
        Initialize medical landmark detector

        Args:
            model_path: Path to trained model weights
            num_landmarks: Number of landmarks
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_landmarks = num_landmarks

        # Load model
        self.model = LandmarkUNet(num_landmarks=num_landmarks)

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Landmark extractor
        self.extractor = SubPixelLandmarkExtractor(method='soft_argmax')

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess medical image

        Args:
            image: Input image (grayscale or RGB)

        Returns:
            Preprocessed tensor [1, 1, H, W]
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to tensor
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        return tensor

    def detect(self, image: np.ndarray) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Detect landmarks in medical image

        Args:
            image: Input medical image

        Returns:
            Tuple of (landmarks, heatmaps)
        """
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)

        # Inference
        with torch.no_grad():
            heatmaps = self.model(input_tensor)

        # Convert to numpy
        heatmaps_np = heatmaps.squeeze(0).cpu().numpy()

        # Extract landmarks
        landmarks = self.extractor.extract_landmarks(heatmaps_np)

        return landmarks, heatmaps_np

    def visualize(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[float, float]],
        heatmaps: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize detection results

        Args:
            image: Original image
            landmarks: Detected landmarks
            heatmaps: Optional heatmaps to display
            save_path: Path to save visualization
        """
        if heatmaps is not None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Original image with landmarks
            if len(image.shape) == 2:
                axes[0, 0].imshow(image, cmap='gray')
            else:
                axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            for idx, (x, y) in enumerate(landmarks):
                axes[0, 0].plot(x, y, 'r+', markersize=10, markeredgewidth=2)
                axes[0, 0].text(x+2, y-2, str(idx+1), color='yellow', fontsize=8)
            axes[0, 0].set_title('Detected Landmarks')
            axes[0, 0].axis('off')

            # Show first 5 heatmaps
            for i in range(min(5, len(heatmaps))):
                row = (i + 1) // 3
                col = (i + 1) % 3
                axes[row, col].imshow(heatmaps[i], cmap='jet')
                axes[row, col].set_title(f'Landmark {i+1} Heatmap')
                axes[row, col].axis('off')

            # Hide unused subplots
            for i in range(min(5, len(heatmaps)) + 1, 6):
                row = i // 3
                col = i % 3
                axes[row, col].axis('off')
        else:
            fig, ax = plt.subplots(figsize=(10, 10))

            if len(image.shape) == 2:
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            for idx, (x, y) in enumerate(landmarks):
                ax.plot(x, y, 'r+', markersize=10, markeredgewidth=2)
                ax.text(x+2, y-2, str(idx+1), color='yellow', fontsize=10)
            ax.set_title('Medical Landmark Detection')
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

    def calculate_metrics(
        self,
        pred_landmarks: List[Tuple[float, float]],
        gt_landmarks: List[Tuple[float, float]],
        image_size: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Calculate landmark detection metrics

        Args:
            pred_landmarks: Predicted landmarks
            gt_landmarks: Ground truth landmarks
            image_size: Image size for normalization

        Returns:
            Dictionary of metrics
        """
        pred = np.array(pred_landmarks)
        gt = np.array(gt_landmarks)

        # Mean Radial Error (MRE)
        distances = np.linalg.norm(pred - gt, axis=1)
        mre = np.mean(distances)

        # Success Detection Rate (SDR) at different thresholds
        sdr_2mm = (distances <= 2.0).mean() * 100
        sdr_2_5mm = (distances <= 2.5).mean() * 100
        sdr_3mm = (distances <= 3.0).mean() * 100
        sdr_4mm = (distances <= 4.0).mean() * 100

        # Normalized Mean Error (NME)
        # Normalize by image diagonal
        diagonal = np.sqrt(image_size[0]**2 + image_size[1]**2)
        nme = (mre / diagonal) * 100

        return {
            'mre_mm': mre,
            'mre_std': np.std(distances),
            'sdr_2mm': sdr_2mm,
            'sdr_2.5mm': sdr_2_5mm,
            'sdr_3mm': sdr_3mm,
            'sdr_4mm': sdr_4mm,
            'nme_percent': nme
        }


def demo():
    """Demo medical landmark detection"""
    print("="*60)
    print("Medical Landmark Detection Demo")
    print("="*60)

    # Create demo cephalometric image
    print("\n1. Creating demo medical image...")
    demo_image = np.random.randint(100, 200, (512, 512), dtype=np.uint8)

    # Add some anatomical-like structures
    cv2.circle(demo_image, (256, 200), 80, 150, -1)  # Head
    cv2.circle(demo_image, (256, 350), 40, 140, -1)  # Jaw

    print("   Image size:", demo_image.shape)

    # Initialize detector (with random weights for demo)
    print("\n2. Initializing detector...")
    detector = MedicalLandmarkDetector(
        model_path=None,  # Use random weights for demo
        num_landmarks=19,  # Standard cephalometric landmarks
        device='cuda'
    )

    # Detect landmarks
    print("\n3. Detecting landmarks...")
    landmarks, heatmaps = detector.detect(demo_image)
    print(f"   Detected {len(landmarks)} landmarks")

    # Visualize
    print("\n4. Visualizing results...")
    detector.visualize(
        demo_image,
        landmarks,
        heatmaps,
        save_path='medical_landmark_detection.png'
    )

    # Demo metrics calculation
    print("\n5. Example metrics calculation:")
    # Create fake ground truth for demo
    gt_landmarks = [(l[0] + np.random.randn()*2, l[1] + np.random.randn()*2)
                    for l in landmarks]

    metrics = detector.calculate_metrics(
        landmarks,
        gt_landmarks,
        demo_image.shape
    )

    for metric, value in metrics.items():
        print(f"   {metric}: {value:.2f}")

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    demo()
