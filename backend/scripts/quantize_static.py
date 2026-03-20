#!/usr/bin/env python3
"""
Static INT8 Quantization for YOLO OBB Model using ONNX Runtime.

This script implements quantize_static (post-training quantization) which quantizes
BOTH weights AND activations, contrasting with quantize_dynamic which only quantizes weights.

For Conv2d-heavy models like YOLOv8 (122 Conv2d layers), static quantization provides:
- Better performance on Intel CPUs with AVX-512 VNNI (Vector Neural Network Instructions)
- Activation quantization reduces memory bandwidth (critical for inference latency)
- Per-channel weight quantization maintains accuracy compared to per-tensor

Key references:
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/quantization/
- Intel CPU optimization: AVX-512 VNNI for INT8 matrix operations
- QDQ (Quantize-Dequantize) format: More compatible with diverse hardware
- QOperator format: More optimized but less portable
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

try:
    from onnxruntime.quantization import (
        quantize_static,
        QuantFormat,
        QuantType,
        CalibrationMethod,
    )
except ImportError:
    print("Error: onnxruntime not installed. Install with: pip install onnxruntime")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Model constants (match ort_inference.py)
INPUT_SIZE = 1280
NUM_ANCHORS = 33600
NUM_CLASSES = 6


class YoloCalibrationDataReader:
    """
    Calibration data reader for static quantization.

    Static quantization requires a small dataset (~100-500 representative images)
    to calibrate the quantization scale/zero-point for activations.

    The calibration dataset should be diverse but small:
    - Representative of actual inference images
    - Diverse in size, lighting, content
    - 100-500 images is typically sufficient for computer vision
    - More samples = better calibration but longer process

    For YOLO, we use the same preprocessing as inference to ensure
    activations are calibrated for the actual input distribution.
    """

    def __init__(self, image_paths: List[str], input_name: str = "images"):
        """
        Args:
            image_paths: List of paths to calibration images.
            input_name: Name of model's input node (default: "images" for YOLOv8).
        """
        self.image_paths = image_paths
        self.input_name = input_name
        self.data_index = 0
        logger.info(f"Calibration dataset: {len(image_paths)} images")

    def get_next(self) -> dict:
        """Yield next batch of preprocessed calibration images.

        Called by quantize_static to get calibration batches.
        Yields None when calibration is complete.
        """
        if self.data_index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.data_index]
        self.data_index += 1

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                logger.warning(f"Failed to load: {img_path}")
                return self.get_next()

            # Preprocess: same as inference
            tensor = self._preprocess(img_bgr)

            return {self.input_name: tensor}
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return self.get_next()

    @staticmethod
    def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
        """Resize with aspect ratio preservation, pad to INPUT_SIZE x INPUT_SIZE.

        Must match ort_inference._preprocess exactly for valid calibration.
        """
        h, w = img_bgr.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w = round(w * scale)
        new_h = round(h * scale)

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas with value 114
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        x_pad = (INPUT_SIZE - new_w) // 2
        y_pad = (INPUT_SIZE - new_h) // 2
        canvas[y_pad : y_pad + new_h, x_pad : x_pad + new_w] = resized

        # BGR -> RGB, HWC -> CHW, normalize to [0,1], add batch dim
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = chw[np.newaxis, ...]  # [1, 3, H, W]

        return tensor

    def rewind(self):
        """Reset calibration dataset pointer."""
        self.data_index = 0


def quantize_yolo_static(
    input_model: str,
    output_model: str,
    calibration_images: List[str],
    quant_format: str = "QDQ",
    per_channel: bool = True,
    reduce_range: bool = False,
    calibration_method: str = "MinMax",
) -> None:
    """
    Perform static INT8 quantization on YOLO ONNX model.

    Args:
        input_model: Path to float32 ONNX model.
        output_model: Path to output INT8 quantized model.
        calibration_images: List of calibration image paths.
        quant_format: "QDQ" (quantize-dequantize) or "QOperator".
            - QDQ: Better portability, works on more hardware/runtimes
            - QOperator: More optimized but less portable (Intel-optimized)
        per_channel: If True, quantize weights per-channel (more accurate).
                     If False, quantize per-tensor (less accurate, faster).
        reduce_range: If True, use 7-bit quantization (less common, for safety).
        calibration_method: "MinMax" (default, fast) or "Entropy" (slower, more accurate).

    Key differences from dynamic quantization:
    1. Activations are quantized (dynamic only quantizes weights)
    2. Requires calibration dataset to determine activation ranges
    3. Much better performance on inference (activations are pre-scaled)
    4. More sensitive to calibration data quality

    Performance implications on Intel CPU:
    - AVX-512 VNNI: 4x speedup for INT8 matrix ops vs float32
    - Per-channel: ~10% accuracy improvement vs per-tensor
    - Activation quantization: 30-50% memory bandwidth reduction
    """
    if not os.path.exists(input_model):
        raise FileNotFoundError(f"Input model not found: {input_model}")

    if not calibration_images:
        raise ValueError("Must provide at least one calibration image")

    # Validate calibration images exist
    valid_images = []
    for img_path in calibration_images:
        if os.path.exists(img_path):
            valid_images.append(img_path)
        else:
            logger.warning(f"Calibration image not found (skipping): {img_path}")

    if not valid_images:
        raise ValueError("No valid calibration images found")

    logger.info(f"Input model: {input_model}")
    logger.info(f"Output model: {output_model}")
    logger.info(f"Quantization format: {quant_format}")
    logger.info(f"Per-channel weights: {per_channel}")
    logger.info(f"Calibration method: {calibration_method}")
    logger.info(f"Reduce range (7-bit): {reduce_range}")
    logger.info(f"Calibration images: {len(valid_images)}")

    # Create calibration data reader
    calibration_data_reader = YoloCalibrationDataReader(valid_images)

    # Map string args to enum values
    quant_format_enum = QuantFormat.QDQ if quant_format == "QDQ" else QuantFormat.QOperator
    calib_method_enum = (
        CalibrationMethod.Entropy
        if calibration_method == "Entropy"
        else CalibrationMethod.MinMax
    )

    logger.info("Starting static quantization...")
    t0 = time.time()

    try:
        quantize_static(
            model_input=input_model,
            model_output=output_model,
            calibration_data_reader=calibration_data_reader,
            # Weight quantization
            quant_format=quant_format_enum,
            per_channel=per_channel,
            # Activation quantization
            calibrate_method=calib_method_enum,
            reduce_range=reduce_range,
            # Use Int8 for both weights and activations
            weight_type=QuantType.QInt8,
            # Enable dynamic/variable batch sizes
            optimize_model=True,
            # Nodes to skip (optional, empty by default)
            nodes_to_skip=[],
        )

        elapsed = time.time() - t0
        logger.info(f"Quantization completed in {elapsed:.2f}s")

        # Report file sizes
        input_size_mb = os.path.getsize(input_model) / (1024 * 1024)
        output_size_mb = os.path.getsize(output_model) / (1024 * 1024)
        reduction_pct = (1 - output_size_mb / input_size_mb) * 100

        logger.info(f"Input model size:  {input_size_mb:.1f} MB")
        logger.info(f"Output model size: {output_size_mb:.1f} MB")
        logger.info(f"Size reduction: {reduction_pct:.1f}%")

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise


def prepare_calibration_dataset(
    image_dir: str,
    num_samples: int = 100,
    shuffle: bool = True,
) -> List[str]:
    """
    Prepare calibration dataset from a directory of images.

    Args:
        image_dir: Directory containing calibration images.
        num_samples: Number of images to use (0=all).
        shuffle: Randomly shuffle image order.

    Returns:
        List of image paths.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = []

    for root, _, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in valid_extensions:
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    if shuffle:
        np.random.shuffle(image_paths)

    if num_samples > 0:
        image_paths = image_paths[:num_samples]

    logger.info(f"Prepared calibration dataset: {len(image_paths)} images")
    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Static INT8 quantization for YOLO OBB model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Quantize with default settings (QDQ format, per-channel, MinMax calibration)
  python quantize_static.py \\
    --input yolo_obb_6class_h7.onnx \\
    --calibration-dir /path/to/calibration/images \\
    --output yolo_obb_6class_h7_int8.onnx

  # Use QOperator format (more optimized for Intel, less portable)
  python quantize_static.py \\
    --input yolo_obb_6class_h7.onnx \\
    --calibration-dir /path/to/calibration/images \\
    --output yolo_obb_6class_h7_int8.onnx \\
    --format QOperator

  # Use Entropy calibration (slower but more accurate)
  python quantize_static.py \\
    --input yolo_obb_6class_h7.onnx \\
    --calibration-dir /path/to/calibration/images \\
    --output yolo_obb_6class_h7_int8.onnx \\
    --calibration-method Entropy

  # Use per-tensor quantization (faster but less accurate)
  python quantize_static.py \\
    --input yolo_obb_6class_h7.onnx \\
    --calibration-dir /path/to/calibration/images \\
    --output yolo_obb_6class_h7_int8.onnx \\
    --no-per-channel
        """,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to float32 ONNX model",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output INT8 quantized model",
    )
    parser.add_argument(
        "--calibration-dir",
        required=True,
        help="Directory containing calibration images",
    )
    parser.add_argument(
        "--calibration-images",
        default=100,
        type=int,
        help="Number of calibration images to use (0=all, default=100)",
    )
    parser.add_argument(
        "--format",
        choices=["QDQ", "QOperator"],
        default="QDQ",
        help=(
            "Quantization format:\n"
            "  QDQ: Quantize-Dequantize (portable, good for diverse hardware)\n"
            "  QOperator: Fused operators (optimized, Intel-specific)\n"
            "  Default: QDQ"
        ),
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="Quantize weights per-channel (more accurate, default)",
    )
    parser.add_argument(
        "--no-per-channel",
        dest="per_channel",
        action="store_false",
        help="Quantize weights per-tensor (less accurate, faster)",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["MinMax", "Entropy"],
        default="MinMax",
        help=(
            "Activation calibration method:\n"
            "  MinMax: Fast, simple range-based (default)\n"
            "  Entropy: Slower, information-theoretic (more accurate)\n"
            "  Default: MinMax"
        ),
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Use 7-bit quantization instead of 8-bit (safety net for edge cases)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle calibration images (default)",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Do not shuffle calibration images",
    )

    args = parser.parse_args()

    # Validate paths
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    calib_dir = os.path.abspath(args.calibration_dir)

    if not os.path.exists(calib_dir):
        print(f"Error: Calibration directory not found: {calib_dir}")
        sys.exit(1)

    # Prepare calibration dataset
    calib_images = prepare_calibration_dataset(
        calib_dir,
        num_samples=args.calibration_images,
        shuffle=args.shuffle,
    )

    # Run quantization
    quantize_yolo_static(
        input_model=input_path,
        output_model=output_path,
        calibration_images=calib_images,
        quant_format=args.format,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        calibration_method=args.calibration_method,
    )

    logger.info(f"Output model saved: {output_path}")


if __name__ == "__main__":
    main()
