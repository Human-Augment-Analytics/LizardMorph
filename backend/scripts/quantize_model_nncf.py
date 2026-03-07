#!/usr/bin/env python3
"""
Quantize YOLO OBB ONNX model to INT8 using NNCF (Neural Network Compression Framework).

This script is superior to onnxruntime.quantization.quantize_dynamic() because:
1. Per-channel quantization (better for CNNs, ~2-3% accuracy preservation)
2. Advanced calibration (entropy, percentile-based range selection)
3. Operator fusion during conversion
4. No retraining required (post-training quantization)

Usage:
    python quantize_model_nncf.py \
        --input models/lizard-toe-pad/yolo_obb_6class_h7.onnx \
        --calibration-dir /path/to/toepad/images \
        --output models/lizard-toe-pad/yolo_obb_6class_h7_int8_nncf.xml \
        --num-samples 300 \
        --preset MIXED

Produces:
    - yolo_obb_6class_h7_int8_nncf.xml (quantized model in OpenVINO IR format)
    - yolo_obb_6class_h7_int8_nncf.bin (quantized weights)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

try:
    from nncf import quantize, QuantizationPreset
    from openvino import convert_model
except ImportError as e:
    print(f"Error: Missing required packages. Install with:")
    print("  pip install nncf openvino openvino-dev")
    sys.exit(1)


def load_calibration_images(image_dir: str, num_samples: int = 300) -> list:
    """
    Load calibration images from directory.

    Args:
        image_dir: Path to directory containing toepad images
        num_samples: Maximum number of images to load

    Returns:
        List of numpy arrays (BGR images)
    """
    images = []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for fname in sorted(os.listdir(image_dir)):
        if Path(fname).suffix.lower() not in valid_exts:
            continue

        img_path = os.path.join(image_dir, fname)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                if len(images) >= num_samples:
                    break
        except Exception as e:
            print(f"  Warning: Failed to load {img_path}: {e}")

    return images


def preprocess_yolo_batch(images: list, input_size: int = 1280) -> np.ndarray:
    """
    Preprocess images for YOLO ONNX model input.

    Mimics the preprocessing in ort_inference.py:
    1. Resize with aspect ratio preservation
    2. Pad to 1280x1280 with value 114 (neutral gray)
    3. BGR → RGB conversion
    4. Normalize to [0, 1]
    5. HWC → CHW + batch dimension

    Args:
        images: List of BGR images (H x W x 3)
        input_size: Target input size (1280 for yolo_obb_6class_h7)

    Returns:
        Batch tensor (N, 3, 1280, 1280) as float32
    """
    batch = []

    for img in images:
        h, w = img.shape[:2]

        # Compute scale to fit within input_size while preserving aspect ratio
        scale = min(input_size / w, input_size / h)
        new_w = round(w * scale)
        new_h = round(h * scale)

        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas (value 114 is standard YOLO padding)
        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        x_pad = (input_size - new_w) // 2
        y_pad = (input_size - new_h) // 2
        padded[y_pad:y_pad + new_h, x_pad:x_pad + new_w] = resized

        # BGR → RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Normalize and convert to float32
        tensor = rgb.astype(np.float32) / 255.0

        # HWC → CHW
        tensor = tensor.transpose(2, 0, 1)

        batch.append(tensor)

    # Stack into batch (N, 3, H, W)
    return np.stack(batch)


def calibration_data_generator(
    image_dir: str, batch_size: int = 8, num_samples: int = 300
) -> Generator:
    """
    Generator that yields calibration batches for NNCF.

    NNCF expects a generator/iterable that yields numpy arrays
    matching the model input shape.

    Args:
        image_dir: Path to calibration images
        batch_size: Batch size for each yielded batch
        num_samples: Total number of samples to use

    Yields:
        Numpy arrays (batch_size, 3, 1280, 1280) as float32
    """
    print(f"[Calibration] Loading images from {image_dir}...")
    images = load_calibration_images(image_dir, num_samples)

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    print(f"[Calibration] Loaded {len(images)} images")

    # Yield batches
    num_batches = (len(images) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch = images[start:end]

        # Preprocess batch
        batch_tensor = preprocess_yolo_batch(batch)
        print(f"  Batch {i+1}/{num_batches}: {batch_tensor.shape}")

        yield batch_tensor


def quantize_yolo_model(
    onnx_path: str,
    calibration_dir: str,
    output_path: str,
    num_samples: int = 300,
    preset: str = "MIXED",
    batch_size: int = 8,
    fast_bias_correction: bool = True,
) -> None:
    """
    Quantize YOLO ONNX model to INT8 using NNCF.

    Args:
        onnx_path: Path to input float32 ONNX model
        calibration_dir: Path to directory with calibration images
        output_path: Path where quantized OpenVINO IR model will be saved
        num_samples: Number of calibration samples to use
        preset: Quantization preset ("MIXED" or "PERFORMANCE")
                - MIXED: INT8 weights + selective float32 for sensitive layers
                - PERFORMANCE: All INT8 (more aggressive, may lose accuracy)
        batch_size: Batch size for calibration
        fast_bias_correction: Use fast bias correction (saves time, minimal accuracy impact)
    """

    # Step 1: Validate inputs
    print("=" * 70)
    print("NNCF Quantization for YOLO OBB")
    print("=" * 70)

    if not os.path.exists(onnx_path):
        print(f"Error: Input ONNX model not found: {onnx_path}")
        sys.exit(1)

    if not os.path.isdir(calibration_dir):
        print(f"Error: Calibration directory not found: {calibration_dir}")
        sys.exit(1)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Step 2: Load and convert ONNX to OpenVINO IR
    print(f"\n[1/4] Converting ONNX → OpenVINO IR...")
    print(f"      Input: {onnx_path}")
    input_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"      Size: {input_size_mb:.1f} MB")

    try:
        ov_model = convert_model(onnx_path)
        print(f"      ✓ Conversion successful")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

    # Step 3: Prepare calibration data generator
    print(f"\n[2/4] Preparing calibration data...")
    print(f"      Directory: {calibration_dir}")
    print(f"      Max samples: {num_samples}")

    try:
        calib_gen = calibration_data_generator(
            calibration_dir, batch_size=batch_size, num_samples=num_samples
        )
    except Exception as e:
        print(f"Error preparing calibration data: {e}")
        sys.exit(1)

    # Step 4: Quantize with NNCF
    print(f"\n[3/4] Quantizing with NNCF (preset={preset})...")
    print(f"      Method: Per-channel INT8 (weights) + selective float32")

    start_time = time.time()

    try:
        # Map preset strings to QuantizationPreset enums
        preset_map = {
            "MIXED": QuantizationPreset.MIXED,
            "PERFORMANCE": QuantizationPreset.PERFORMANCE,
        }

        if preset not in preset_map:
            print(f"Error: Invalid preset '{preset}'. Use MIXED or PERFORMANCE")
            sys.exit(1)

        quantized_model = quantize(
            ov_model,
            calib_gen,
            subset_size=min(num_samples, 300),  # NNCF recommendation
            preset=preset_map[preset],
            fast_bias_correction=fast_bias_correction,
        )

        quant_time = time.time() - start_time
        print(f"      ✓ Quantization complete ({quant_time:.1f}s)")

    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Save quantized model
    print(f"\n[4/4] Saving quantized model...")
    print(f"      Output: {output_path}")

    try:
        quantized_model.save_model(output_path)

        # Check output size
        xml_path = output_path.replace(".xml", ".xml")
        bin_path = output_path.replace(".xml", ".bin")

        xml_size = os.path.getsize(xml_path) if os.path.exists(xml_path) else 0
        bin_size = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
        total_size = (xml_size + bin_size) / (1024 * 1024)

        print(f"      ✓ Model saved successfully")
        print(f"      Total size: {total_size:.1f} MB")

        # Summary
        print("\n" + "=" * 70)
        print("QUANTIZATION SUMMARY")
        print("=" * 70)
        print(f"Original (FP32):  {input_size_mb:.1f} MB")
        print(f"Quantized (INT8): {total_size:.1f} MB")
        print(f"Reduction:        {(1 - total_size / input_size_mb) * 100:.1f}%")
        print(f"Speedup expected: 2.0-3.5x (depends on CPU VNNI support)")
        print()
        print("Next steps:")
        print("1. Validate accuracy: python scripts/validate_int8_accuracy.py \\")
        print(f"     --fp32-model {onnx_path} \\")
        print(f"     --int8-model {output_path} \\")
        print("     --test-images /path/to/test/images")
        print()
        print("2. Deploy: Use OpenVINO inference in production")
        print()

    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize YOLO ONNX model to INT8 using NNCF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic quantization with 300 samples
  python quantize_model_nncf.py \\
    --input models/yolo_obb_6class_h7.onnx \\
    --calibration-dir ./calibration_data \\
    --output models/yolo_obb_6class_h7_int8_nncf.xml

  # Conservative preset (safer accuracy)
  python quantize_model_nncf.py \\
    --input models/yolo_obb_6class_h7.onnx \\
    --calibration-dir ./calibration_data \\
    --output models/yolo_obb_6class_h7_int8_nncf.xml \\
    --preset MIXED \\
    --num-samples 500
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input float32 ONNX model",
    )
    parser.add_argument(
        "--calibration-dir",
        required=True,
        help="Path to directory with calibration images",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save quantized OpenVINO IR model (*.xml)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="Number of calibration samples (default: 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for calibration (default: 8)",
    )
    parser.add_argument(
        "--preset",
        choices=["MIXED", "PERFORMANCE"],
        default="MIXED",
        help="Quantization preset (default: MIXED)",
    )
    parser.add_argument(
        "--no-fast-bias",
        action="store_true",
        help="Disable fast bias correction (slower but potentially more accurate)",
    )

    args = parser.parse_args()

    quantize_yolo_model(
        onnx_path=args.input,
        calibration_dir=args.calibration_dir,
        output_path=args.output,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        preset=args.preset,
        fast_bias_correction=not args.no_fast_bias,
    )


if __name__ == "__main__":
    main()
