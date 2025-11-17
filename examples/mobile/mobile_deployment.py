"""
Mobile Deployment Example for U-Net
Export model to ONNX, TorchScript, and optimize for mobile devices
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import onnx
import onnxruntime as ort
import numpy as np
from torch.utils.mobile_optimizer import optimize_for_mobile
import time


class MobileUNet:
    """Lightweight U-Net optimized for mobile deployment"""

    def __init__(self, encoder='mobilenet_v2'):
        """
        Initialize mobile-friendly U-Net

        Args:
            encoder: Lightweight encoder (mobilenet_v2, efficientnet-b0)
        """
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        self.model.eval()

    def export_to_onnx(
        self,
        save_path='unet_mobile.onnx',
        input_size=(1, 3, 512, 512),
        opset_version=11
    ):
        """
        Export model to ONNX format

        Args:
            save_path: Path to save ONNX model
            input_size: Input tensor size (B, C, H, W)
            opset_version: ONNX opset version
        """
        print(f"\\nExporting model to ONNX...")

        # Create dummy input
        dummy_input = torch.randn(*input_size)

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )

        # Verify ONNX model
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)

        # Get model size
        import os
        size_mb = os.path.getsize(save_path) / (1024 * 1024)

        print(f"✓ Model exported to {save_path}")
        print(f"  Model size: {size_mb:.2f} MB")
        print(f"  ONNX opset version: {opset_version}")

        return save_path

    def export_to_torchscript(
        self,
        save_path='unet_mobile.pt',
        input_size=(1, 3, 512, 512)
    ):
        """
        Export model to TorchScript format

        Args:
            save_path: Path to save TorchScript model
            input_size: Input tensor size
        """
        print(f"\\nExporting model to TorchScript...")

        # Create example input
        example_input = torch.randn(*input_size)

        # Trace the model
        traced_model = torch.jit.trace(self.model, example_input)

        # Optimize for mobile
        optimized_model = optimize_for_mobile(traced_model)

        # Save
        optimized_model._save_for_lite_interpreter(save_path)

        # Get model size
        import os
        size_mb = os.path.getsize(save_path) / (1024 * 1024)

        print(f"✓ Model exported to {save_path}")
        print(f"  Model size: {size_mb:.2f} MB")
        print(f"  Optimized for mobile: Yes")

        return save_path

    def quantize_model(self, save_path='unet_mobile_quantized.pt'):
        """
        Quantize model to reduce size and improve inference speed

        Args:
            save_path: Path to save quantized model
        """
        print(f"\\nQuantizing model...")

        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8
        )

        # Save quantized model
        torch.save(quantized_model.state_dict(), save_path)

        # Compare sizes
        import os
        original_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)
        quantized_size = os.path.getsize(save_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"✓ Model quantized and saved to {save_path}")
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {reduction:.1f}%")

        return save_path


class ONNXInference:
    """ONNX model inference for deployment"""

    def __init__(self, model_path):
        """
        Initialize ONNX runtime session

        Args:
            model_path: Path to ONNX model
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"\\nONNX Runtime initialized")
        print(f"  Input name: {self.input_name}")
        print(f"  Output name: {self.output_name}")

    def preprocess(self, image):
        """
        Preprocess image for inference

        Args:
            image: Input image [H, W, 3]

        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Transpose to [C, H, W]
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, image):
        """
        Run inference

        Args:
            image: Input image [H, W, 3]

        Returns:
            Prediction mask [H, W]
        """
        # Preprocess
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        # Post-process
        mask = outputs[0].squeeze()
        mask = (mask > 0.5).astype(np.uint8)

        return mask

    def benchmark(self, input_size=(512, 512), num_iterations=100):
        """
        Benchmark inference speed

        Args:
            input_size: Input image size (H, W)
            num_iterations: Number of inference runs

        Returns:
            Average inference time in milliseconds
        """
        print(f"\\nBenchmarking ONNX inference...")
        print(f"  Input size: {input_size}")
        print(f"  Iterations: {num_iterations}")

        # Create random input
        dummy_image = np.random.randint(
            0, 255, (*input_size, 3), dtype=np.uint8
        )

        # Warmup
        for _ in range(10):
            _ = self.predict(dummy_image)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = self.predict(dummy_image)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time

        print(f"\\n  Results:")
        print(f"    Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"    FPS: {fps:.2f}")
        print(f"    Min time: {np.min(times):.2f} ms")
        print(f"    Max time: {np.max(times):.2f} ms")

        return avg_time


def compare_model_formats():
    """Compare different model export formats"""
    print("="*60)
    print("Mobile Deployment Format Comparison")
    print("="*60)

    # Create mobile model
    print("\\n1. Creating lightweight U-Net...")
    mobile_unet = MobileUNet(encoder='mobilenet_v2')

    # Count parameters
    total_params = sum(p.numel() for p in mobile_unet.model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size (float32): {total_params * 4 / 1024 / 1024:.2f} MB")

    # Export to ONNX
    print("\\n2. Exporting to ONNX...")
    onnx_path = mobile_unet.export_to_onnx()

    # Export to TorchScript
    print("\\n3. Exporting to TorchScript...")
    torchscript_path = mobile_unet.export_to_torchscript()

    # Quantize model
    print("\\n4. Quantizing model...")
    quantized_path = mobile_unet.quantize_model()

    # Benchmark ONNX
    print("\\n5. Benchmarking ONNX inference...")
    onnx_inference = ONNXInference(onnx_path)
    avg_time = onnx_inference.benchmark(input_size=(512, 512), num_iterations=50)

    print("\\n" + "="*60)
    print("Export completed successfully!")
    print("="*60)

    print("\\nSummary:")
    print(f"  ONNX model: {onnx_path}")
    print(f"  TorchScript model: {torchscript_path}")
    print(f"  Quantized model: {quantized_path}")
    print(f"  Average inference time: {avg_time:.2f} ms")


def deployment_guide():
    """Print deployment guide for different platforms"""
    guide = """

📱 Mobile Deployment Guide
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  iOS Deployment (Core ML)
   - Convert ONNX → Core ML using coremltools
   - Example:
     ```python
     import coremltools as ct
     mlmodel = ct.converters.onnx.convert(model='unet_mobile.onnx')
     mlmodel.save('UNet.mlmodel')
     ```
   - Integrate into Xcode project
   - Use Vision framework for image preprocessing

2️⃣  Android Deployment (TensorFlow Lite)
   - Convert PyTorch → ONNX → TF → TFLite
   - Or use PyTorch Mobile directly
   - Example:
     ```python
     import tensorflow as tf
     converter = tf.lite.TFLiteConverter.from_saved_model('model')
     tflite_model = converter.convert()
     ```
   - Integrate using TFLite Interpreter

3️⃣  Web Deployment (ONNX.js or TensorFlow.js)
   - Use ONNX Runtime Web
   - Example:
     ```javascript
     const session = await ort.InferenceSession.create('unet_mobile.onnx');
     const results = await session.run(feeds);
     ```

4️⃣  Edge Devices (NVIDIA Jetson, Raspberry Pi)
   - Use TensorRT for NVIDIA devices
   - Use ONNX Runtime for Raspberry Pi
   - Optimize with INT8 quantization

5️⃣  Recommended Optimizations
   ✓ Use MobileNetV2/V3 or EfficientNet-B0 encoders
   ✓ Reduce input resolution (256x256 or 384x384)
   ✓ Apply post-training quantization
   ✓ Use pruning to reduce model size
   ✓ Enable GPU acceleration when available

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(guide)


def main():
    """Main function"""
    compare_model_formats()
    deployment_guide()


if __name__ == "__main__":
    main()
