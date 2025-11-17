"""
Instance Segmentation using U-Net + Watershed Algorithm
"""

import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import matplotlib.pyplot as plt


class InstanceSegmentationPipeline:
    """Complete pipeline for instance segmentation using U-Net + Watershed"""

    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize the instance segmentation pipeline

        Args:
            model_path: Path to trained U-Net model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Create or load model
        self.model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=None,  # Will load from checkpoint
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )

        if model_path:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path):
        """Load trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")

    def preprocess_image(self, image):
        """
        Preprocess image for model input

        Args:
            image: numpy array [H, W, 3] or [H, W]

        Returns:
            tensor: Preprocessed image tensor [1, 3, H, W]
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.unsqueeze(0)  # Add batch dimension

        return image

    def semantic_segmentation(self, image):
        """
        Run semantic segmentation using U-Net

        Args:
            image: Input image [H, W, 3]

        Returns:
            mask: Binary segmentation mask [H, W]
        """
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Post-process
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

        return mask

    def watershed_segmentation(self, semantic_mask, min_distance=10):
        """
        Apply watershed algorithm to separate instances

        Args:
            semantic_mask: Binary semantic segmentation mask
            min_distance: Minimum distance between instance peaks

        Returns:
            instance_mask: Instance segmentation mask with unique labels
        """
        # Distance transform
        distance = ndimage.distance_transform_edt(semantic_mask)

        # Find local maxima (peaks) as markers
        local_max = morphology.local_maxima(distance)
        markers = measure.label(local_max)

        # Alternative: Use peak_local_max for more control
        from skimage.feature import peak_local_max
        coordinates = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=semantic_mask
        )

        # Create markers from coordinates
        markers = np.zeros_like(semantic_mask, dtype=int)
        for i, coord in enumerate(coordinates, start=1):
            markers[tuple(coord)] = i

        # Apply watershed
        instance_mask = morphology.watershed(
            -distance,
            markers,
            mask=semantic_mask
        )

        return instance_mask

    def refine_instances(self, instance_mask, min_size=100):
        """
        Refine instance masks by removing small objects

        Args:
            instance_mask: Instance segmentation mask
            min_size: Minimum instance size in pixels

        Returns:
            refined_mask: Refined instance mask
        """
        # Remove small objects
        refined_mask = morphology.remove_small_objects(
            instance_mask.astype(bool),
            min_size=min_size
        )

        # Re-label instances
        refined_mask = measure.label(refined_mask)

        return refined_mask

    def extract_instance_properties(self, instance_mask):
        """
        Extract properties of each instance

        Args:
            instance_mask: Instance segmentation mask

        Returns:
            properties: List of dictionaries with instance properties
        """
        regions = measure.regionprops(instance_mask)

        properties = []
        for region in regions:
            prop = {
                'label': region.label,
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox,  # (min_row, min_col, max_row, max_col)
                'perimeter': region.perimeter,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity,
            }
            properties.append(prop)

        return properties

    def predict(self, image, min_distance=10, min_size=100):
        """
        Complete instance segmentation pipeline

        Args:
            image: Input image [H, W, 3]
            min_distance: Minimum distance between instance peaks
            min_size: Minimum instance size

        Returns:
            instance_mask: Instance segmentation mask
            properties: List of instance properties
        """
        # Step 1: Semantic segmentation
        semantic_mask = self.semantic_segmentation(image)

        # Step 2: Watershed instance segmentation
        instance_mask = self.watershed_segmentation(semantic_mask, min_distance)

        # Step 3: Refine instances
        instance_mask = self.refine_instances(instance_mask, min_size)

        # Step 4: Extract properties
        properties = self.extract_instance_properties(instance_mask)

        return instance_mask, properties

    def visualize_results(self, image, instance_mask, properties=None, save_path=None):
        """
        Visualize instance segmentation results

        Args:
            image: Original image
            instance_mask: Instance segmentation mask
            properties: List of instance properties (optional)
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')

        # Instance mask with random colors
        colored_mask = self.colorize_instance_mask(instance_mask)
        axes[1].imshow(colored_mask)
        axes[1].set_title(f'Instance Segmentation ({instance_mask.max()} instances)', fontsize=14)
        axes[1].axis('off')

        # Overlay
        overlay = image.copy()
        overlay = (overlay * 0.6 + colored_mask * 0.4).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay', fontsize=14)
        axes[2].axis('off')

        # Add bounding boxes and labels if properties provided
        if properties:
            for prop in properties:
                min_row, min_col, max_row, max_col = prop['bbox']
                rect = plt.Rectangle(
                    (min_col, min_row),
                    max_col - min_col,
                    max_row - min_row,
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                axes[2].add_patch(rect)

                # Add label
                cy, cx = prop['centroid']
                axes[2].text(
                    cx, cy,
                    f"{prop['label']}",
                    color='white',
                    fontsize=10,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7)
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

    @staticmethod
    def colorize_instance_mask(instance_mask):
        """
        Convert instance mask to colored image

        Args:
            instance_mask: Instance mask with unique labels

        Returns:
            colored_mask: RGB colored mask
        """
        num_instances = instance_mask.max()
        colored_mask = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)

        # Generate random colors for each instance
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_instances + 1, 3))
        colors[0] = [0, 0, 0]  # Background is black

        for i in range(num_instances + 1):
            colored_mask[instance_mask == i] = colors[i]

        return colored_mask


def main():
    """Demo instance segmentation pipeline"""
    print("="*60)
    print("Instance Segmentation using U-Net + Watershed")
    print("="*60)

    # Create pipeline
    print("\n1. Initializing pipeline...")
    pipeline = InstanceSegmentationPipeline(
        model_path=None,  # Use None for demo (random weights)
        device='cuda'
    )

    # Create demo image (in practice, load from file)
    print("\n2. Creating demo image...")
    demo_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Run instance segmentation
    print("\n3. Running instance segmentation...")
    instance_mask, properties = pipeline.predict(
        demo_image,
        min_distance=20,
        min_size=100
    )

    # Print results
    print(f"\n4. Results:")
    print(f"   Number of instances detected: {len(properties)}")
    print(f"\n   Instance properties:")
    for i, prop in enumerate(properties[:5], 1):  # Show first 5
        print(f"   Instance {i}:")
        print(f"      - Area: {prop['area']} pixels")
        print(f"      - Centroid: ({prop['centroid'][0]:.1f}, {prop['centroid'][1]:.1f})")
        print(f"      - BBox: {prop['bbox']}")

    if len(properties) > 5:
        print(f"   ... and {len(properties) - 5} more instances")

    # Visualize
    print("\n5. Generating visualization...")
    pipeline.visualize_results(
        demo_image,
        instance_mask,
        properties,
        save_path='instance_segmentation_result.png'
    )

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
