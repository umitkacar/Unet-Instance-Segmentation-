"""
Medical Image Segmentation with U-Net
Specialized for biomedical imaging applications
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


class MedicalUNet:
    """U-Net model specialized for medical image segmentation"""

    def __init__(
        self,
        task_type='cell_nuclei',
        encoder='efficientnet-b4',
        device='cuda'
    ):
        """
        Initialize medical segmentation model

        Args:
            task_type: Type of medical task (cell_nuclei, organ, tumor)
            encoder: Backbone encoder
            device: Device to run on
        """
        self.task_type = task_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Create model based on task
        self.model = self._create_model(encoder)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _create_model(self, encoder):
        """Create task-specific model"""
        if self.task_type == 'cell_nuclei':
            # For cell nuclei segmentation
            model = smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
                activation='sigmoid'
            )
        elif self.task_type == 'tumor':
            # For tumor segmentation
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=1,  # Often grayscale CT/MRI
                classes=1,
                activation='sigmoid'
            )
        else:
            # Default U-Net
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
                activation='sigmoid'
            )

        return model

    def preprocess_medical_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess medical image

        Args:
            image: Input medical image

        Returns:
            Preprocessed tensor
        """
        # Normalize based on modality
        if self.task_type == 'tumor':
            # CT/MRI normalization (often requires different approach)
            image = self._normalize_ct_mri(image)
        else:
            # Standard normalization for histology
            image = image.astype(np.float32) / 255.0

        # Convert to tensor
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.unsqueeze(0)

        return image

    def _normalize_ct_mri(self, image: np.ndarray) -> np.ndarray:
        """Normalize CT/MRI images using windowing"""
        # Example Hounsfield Unit windowing for CT
        # Adjust window based on organ/tissue
        window_center = 40  # Soft tissue
        window_width = 400

        min_value = window_center - window_width / 2
        max_value = window_center + window_width / 2

        image = np.clip(image, min_value, max_value)
        image = (image - min_value) / (max_value - min_value)

        return image

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Perform segmentation

        Args:
            image: Input image

        Returns:
            Segmentation mask
        """
        input_tensor = self.preprocess_medical_image(image).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

        return mask

    def post_process_cells(self, mask: np.ndarray) -> np.ndarray:
        """
        Post-process cell segmentation masks

        Args:
            mask: Binary segmentation mask

        Returns:
            Cleaned mask
        """
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Remove small objects
        from skimage import morphology
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=50)

        return mask.astype(np.uint8)

    def calculate_medical_metrics(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate medical imaging metrics

        Args:
            pred_mask: Predicted mask
            gt_mask: Ground truth mask

        Returns:
            Dictionary of metrics
        """
        # Dice Coefficient (F1-Score)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        dice = (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)

        # IoU (Jaccard Index)
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / (union + 1e-8)

        # Sensitivity (Recall)
        sensitivity = intersection / (gt_mask.sum() + 1e-8)

        # Specificity
        tn = np.logical_and(~pred_mask, ~gt_mask).sum()
        fp = np.logical_and(pred_mask, ~gt_mask).sum()
        specificity = tn / (tn + fp + 1e-8)

        # Precision
        precision = intersection / (pred_mask.sum() + 1e-8)

        # Hausdorff Distance
        from scipy.spatial.distance import directed_hausdorff
        pred_coords = np.argwhere(pred_mask)
        gt_coords = np.argwhere(gt_mask)

        if len(pred_coords) > 0 and len(gt_coords) > 0:
            hausdorff = max(
                directed_hausdorff(pred_coords, gt_coords)[0],
                directed_hausdorff(gt_coords, pred_coords)[0]
            )
        else:
            hausdorff = float('inf')

        metrics = {
            'dice': dice,
            'iou': iou,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'hausdorff_distance': hausdorff
        }

        return metrics

    def visualize_medical_results(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        ground_truth: np.ndarray = None,
        save_path: str = None
    ):
        """
        Visualize medical segmentation results

        Args:
            image: Original image
            mask: Predicted mask
            ground_truth: Ground truth mask (optional)
            save_path: Path to save figure
        """
        n_plots = 3 if ground_truth is not None else 2
        fig, axes = plt.subplots(1, n_plots + 1, figsize=(6 * (n_plots + 1), 6))

        # Original image
        if len(image.shape) == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')

        # Prediction
        axes[1].imshow(mask, cmap='jet', alpha=0.7)
        axes[1].set_title('Prediction', fontsize=14)
        axes[1].axis('off')

        # Overlay
        if len(image.shape) == 2:
            overlay = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            overlay = image.copy()

        mask_colored = np.zeros_like(overlay)
        mask_colored[mask > 0] = [255, 0, 0]  # Red for segmentation
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14)
        axes[2].axis('off')

        # Ground truth comparison
        if ground_truth is not None:
            # Calculate metrics
            metrics = self.calculate_medical_metrics(mask, ground_truth)

            # Comparison visualization
            comparison = np.zeros((*mask.shape, 3))
            comparison[np.logical_and(mask, ground_truth)] = [0, 255, 0]  # TP: Green
            comparison[np.logical_and(mask, ~ground_truth)] = [255, 0, 0]  # FP: Red
            comparison[np.logical_and(~mask, ground_truth)] = [0, 0, 255]  # FN: Blue

            axes[3].imshow(comparison.astype(np.uint8))
            title = f"Comparison\\nDice: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f}"
            axes[3].set_title(title, fontsize=14)
            axes[3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()


def demo_cell_nuclei_segmentation():
    """Demo for cell nuclei segmentation"""
    print("\\n" + "="*60)
    print("Cell Nuclei Segmentation Demo")
    print("="*60)

    # Initialize model
    model = MedicalUNet(task_type='cell_nuclei', encoder='efficientnet-b4')

    # Create demo histology image
    demo_image = np.random.randint(150, 255, (512, 512, 3), dtype=np.uint8)

    # Add some circular "nuclei"
    for _ in range(20):
        center = (np.random.randint(50, 462), np.random.randint(50, 462))
        radius = np.random.randint(10, 30)
        color = tuple(np.random.randint(100, 200, 3).tolist())
        cv2.circle(demo_image, center, radius, color, -1)

    print("\\nRunning cell nuclei segmentation...")
    mask = model.segment(demo_image)
    mask = model.post_process_cells(mask)

    print(f"Segmented area: {mask.sum()} pixels")
    print(f"Coverage: {mask.sum() / mask.size * 100:.2f}%")

    model.visualize_medical_results(
        demo_image,
        mask,
        save_path='cell_nuclei_segmentation.png'
    )


def demo_tumor_segmentation():
    """Demo for tumor segmentation"""
    print("\\n" + "="*60)
    print("Tumor Segmentation Demo (CT/MRI)")
    print("="*60)

    # Initialize model
    model = MedicalUNet(task_type='tumor', encoder='resnet50')

    # Create demo CT scan
    demo_ct = np.random.randint(0, 100, (512, 512), dtype=np.uint8)

    # Add "tumor" region
    tumor_mask = np.zeros_like(demo_ct)
    cv2.ellipse(tumor_mask, (256, 256), (80, 60), 30, 0, 360, 255, -1)
    demo_ct = np.where(tumor_mask > 0, 150, demo_ct)

    print("\\nRunning tumor segmentation...")
    mask = model.segment(demo_ct)

    # Create ground truth for metrics
    gt_mask = (tumor_mask > 0).astype(np.uint8)

    # Calculate metrics
    metrics = model.calculate_medical_metrics(mask, gt_mask)

    print("\\nMetrics:")
    for metric_name, value in metrics.items():
        if metric_name != 'hausdorff_distance':
            print(f"  {metric_name.capitalize()}: {value:.4f}")
        else:
            print(f"  {metric_name.replace('_', ' ').title()}: {value:.2f} pixels")

    model.visualize_medical_results(
        demo_ct,
        mask,
        gt_mask,
        save_path='tumor_segmentation.png'
    )


def main():
    """Main function"""
    print("Medical Image Segmentation with U-Net")

    # Run demos
    demo_cell_nuclei_segmentation()
    demo_tumor_segmentation()

    print("\\n" + "="*60)
    print("All demos completed!")
    print("="*60)


if __name__ == "__main__":
    main()
