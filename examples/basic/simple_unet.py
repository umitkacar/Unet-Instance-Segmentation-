"""
Simple U-Net Model Creation and Usage Example
"""

import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_unet_model(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation='sigmoid'
):
    """
    Create a U-Net model with specified parameters

    Args:
        encoder_name: Backbone encoder (resnet34, efficientnet-b4, etc.)
        encoder_weights: Pretrained weights (imagenet, None)
        in_channels: Number of input channels
        classes: Number of output classes
        activation: Output activation function

    Returns:
        model: U-Net model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )
    return model


def get_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


def inference_example(model, image_path=None):
    """
    Run inference on a sample image

    Args:
        model: Trained U-Net model
        image_path: Path to input image (if None, creates random tensor)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Create sample input
    if image_path:
        # Load real image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
    else:
        # Create random tensor
        image = torch.rand(3, 512, 512)

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(image)

    print(f"Input shape: {image.shape}")
    print(f"Output shape: {output.shape}")

    return output


def visualize_prediction(image, prediction, ground_truth=None):
    """
    Visualize input image, prediction, and optionally ground truth

    Args:
        image: Input image tensor [C, H, W]
        prediction: Model prediction [1, H, W]
        ground_truth: Ground truth mask [1, H, W] (optional)
    """
    fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))

    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().permute(1, 2, 0).numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().squeeze().numpy()
    if ground_truth is not None and isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().squeeze().numpy()

    # Plot image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Plot prediction
    axes[1].imshow(prediction, cmap='jet')
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Plot ground truth if available
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap='jet')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'prediction_visualization.png'")


def main():
    """Main function to demonstrate U-Net usage"""
    print("="*60)
    print("U-Net Instance Segmentation - Basic Example")
    print("="*60)

    # Create model
    print("\n1. Creating U-Net model...")
    model = create_unet_model(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    # Get model info
    print("\n2. Model Information:")
    get_model_info(model)

    # Run inference
    print("\n3. Running inference on random image...")
    output = inference_example(model)

    print("\n4. Inference successful!")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

    # List available encoders
    print("\n5. Available Encoders:")
    available_encoders = [
        "resnet18", "resnet34", "resnet50", "resnet101",
        "efficientnet-b0", "efficientnet-b4", "efficientnet-b7",
        "densenet121", "densenet169", "densenet201",
        "vgg16", "vgg19",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"
    ]
    for i, encoder in enumerate(available_encoders, 1):
        print(f"   {i:2d}. {encoder}")

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
