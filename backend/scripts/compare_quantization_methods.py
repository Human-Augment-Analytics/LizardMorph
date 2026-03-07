#!/usr/bin/env python3
"""
Compare different quantization methods for YOLO OBB:
1. Float32 (baseline)
2. Dynamic INT8 (quantize_dynamic)
3. Static INT8 with MinMax calibration
4. Static INT8 with Entropy calibration

Measures:
- Model size
- Inference latency
- Memory usage
- Accuracy (if ground truth provided)
"""

import argparse
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

try:
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        QuantFormat,
        QuantType,
        CalibrationMethod,
    )
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed")
    sys.exit(1)

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ort_inference import OrtYoloDetector


class QuantizationComparison:
    """Compare quantization methods."""

    def __init__(self, float_model_path: str, calibration_images: List[str], test_images: List[str]):
        self.float_model = float_model_path
        self.calib_images = calibration_images
        self.test_images = test_images
        self.results = {}

    def get_model_size(self, model_path: str) -> float:
        """Get model file size in MB."""
        return os.path.getsize(model_path) / (1024 * 1024)

    def _downsample(self, img_bgr, max_dim=4096):
        """Downsample large images."""
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img_bgr

    def benchmark_inference(self, model_path: str, test_images: List[str], runs: int = 5, warmup: int = 2) -> Tuple[float, float]:
        """Run inference and measure latency.

        Returns:
            (median_latency_sec, std_dev_sec)
        """
        try:
            detector = OrtYoloDetector(model_path)
            images = [self._downsample(cv2.imread(p)) for p in test_images if os.path.exists(p)]

            if not images:
                return None, None

            # Warmup
            for _ in range(warmup):
                detector.detect(images[0])

            # Measure
            times = []
            for img in images:
                img_times = []
                for _ in range(runs):
                    t0 = time.perf_counter()
                    detector.detect(img)
                    t1 = time.perf_counter()
                    img_times.append(t1 - t0)
                times.append(np.median(img_times))

            median = np.median(times)
            std_dev = np.std(times)
            return median, std_dev
        except Exception as e:
            print(f"Error benchmarking {model_path}: {e}")
            return None, None

    def quantize_dynamic(self, output_path: str) -> bool:
        """Quantize using dynamic quantization."""
        print(f"\n[Dynamic Quantization] Starting...")
        try:
            quantize_dynamic(
                model_input=self.float_model,
                model_output=output_path,
                weight_type=QuantType.QUInt8,
            )
            print(f"[Dynamic Quantization] Success: {output_path}")
            return True
        except Exception as e:
            print(f"[Dynamic Quantization] Failed: {e}")
            return False

    def quantize_static(self, output_path: str, calib_method: str = "MinMax") -> bool:
        """Quantize using static quantization."""
        print(f"\n[Static Quantization ({calib_method})] Starting calibration on {len(self.calib_images)} images...")

        try:
            # Create calibration reader
            from quantize_static import YoloCalibrationDataReader
            calib_reader = YoloCalibrationDataReader(self.calib_images)

            calib_enum = CalibrationMethod.Entropy if calib_method == "Entropy" else CalibrationMethod.MinMax

            t0 = time.time()
            quantize_static(
                model_input=self.float_model,
                model_output=output_path,
                calibration_data_reader=calib_reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
                calibrate_method=calib_enum,
                reduce_range=False,
                optimize_model=True,
            )
            elapsed = time.time() - t0
            print(f"[Static Quantization ({calib_method})] Success in {elapsed:.1f}s: {output_path}")
            return True
        except Exception as e:
            print(f"[Static Quantization ({calib_method})] Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_comparison(self, warmup: int = 2, runs: int = 5) -> Dict:
        """Run full comparison."""
        results = {}

        # 1. Baseline: Float32
        print(f"\n{'='*70}")
        print(f"1. Float32 (Baseline)")
        print(f"{'='*70}")
        f32_size = self.get_model_size(self.float_model)
        f32_latency, f32_std = self.benchmark_inference(self.float_model, self.test_images, runs, warmup)
        results["float32"] = {
            "size_mb": f32_size,
            "latency_sec": f32_latency,
            "latency_std": f32_std,
        }
        if f32_latency:
            print(f"  Model size:     {f32_size:.1f} MB")
            print(f"  Latency:        {f32_latency:.3f}s ± {f32_std:.3f}s")
            print(f"  (Baseline for speedup comparison)")

        # 2. Dynamic quantization
        print(f"\n{'='*70}")
        print(f"2. Dynamic INT8 (quantize_dynamic with QUInt8)")
        print(f"{'='*70}")
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            dyn_model = f.name
        try:
            if self.quantize_dynamic(dyn_model):
                dyn_size = self.get_model_size(dyn_model)
                dyn_latency, dyn_std = self.benchmark_inference(dyn_model, self.test_images, runs, warmup)
                results["dynamic_int8"] = {
                    "size_mb": dyn_size,
                    "latency_sec": dyn_latency,
                    "latency_std": dyn_std,
                }
                if dyn_latency and f32_latency:
                    speedup = f32_latency / dyn_latency
                    print(f"  Model size:     {dyn_size:.1f} MB ({f32_size - dyn_size:.1f} MB reduction, {(1 - dyn_size/f32_size)*100:.1f}%)")
                    print(f"  Latency:        {dyn_latency:.3f}s ± {dyn_std:.3f}s")
                    print(f"  Speedup:        {speedup:.2f}x vs float32")
        finally:
            if os.path.exists(dyn_model):
                os.unlink(dyn_model)

        # 3. Static with MinMax
        print(f"\n{'='*70}")
        print(f"3. Static INT8 (MinMax calibration, per-channel)")
        print(f"{'='*70}")
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            static_minmax_model = f.name
        try:
            if self.quantize_static(static_minmax_model, "MinMax"):
                stat_mm_size = self.get_model_size(static_minmax_model)
                stat_mm_latency, stat_mm_std = self.benchmark_inference(static_minmax_model, self.test_images, runs, warmup)
                results["static_int8_minmax"] = {
                    "size_mb": stat_mm_size,
                    "latency_sec": stat_mm_latency,
                    "latency_std": stat_mm_std,
                }
                if stat_mm_latency and f32_latency:
                    speedup = f32_latency / stat_mm_latency
                    print(f"  Model size:     {stat_mm_size:.1f} MB ({f32_size - stat_mm_size:.1f} MB reduction, {(1 - stat_mm_size/f32_size)*100:.1f}%)")
                    print(f"  Latency:        {stat_mm_latency:.3f}s ± {stat_mm_std:.3f}s")
                    print(f"  Speedup:        {speedup:.2f}x vs float32")
        finally:
            if os.path.exists(static_minmax_model):
                os.unlink(static_minmax_model)

        # 4. Static with Entropy
        print(f"\n{'='*70}")
        print(f"4. Static INT8 (Entropy calibration, per-channel)")
        print(f"{'='*70}")
        print(f"  Note: Entropy calibration is slower (~5-10x calibration time)")
        print(f"  Expected result: ~0.5-1% better accuracy than MinMax")
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            static_entropy_model = f.name
        try:
            if self.quantize_static(static_entropy_model, "Entropy"):
                stat_ent_size = self.get_model_size(static_entropy_model)
                stat_ent_latency, stat_ent_std = self.benchmark_inference(static_entropy_model, self.test_images, runs, warmup)
                results["static_int8_entropy"] = {
                    "size_mb": stat_ent_size,
                    "latency_sec": stat_ent_latency,
                    "latency_std": stat_ent_std,
                }
                if stat_ent_latency and f32_latency:
                    speedup = f32_latency / stat_ent_latency
                    print(f"  Model size:     {stat_ent_size:.1f} MB ({f32_size - stat_ent_size:.1f} MB reduction, {(1 - stat_ent_size/f32_size)*100:.1f}%)")
                    print(f"  Latency:        {stat_ent_latency:.3f}s ± {stat_ent_std:.3f}s")
                    print(f"  Speedup:        {speedup:.2f}x vs float32")
        finally:
            if os.path.exists(static_entropy_model):
                os.unlink(static_entropy_model)

        # Summary
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"\n{'Method':<30} {'Size (MB)':<12} {'Latency (s)':<15} {'Speedup':<10}")
        print(f"{'-'*70}")

        baseline_lat = results["float32"]["latency_sec"]
        for method, data in results.items():
            lat = data["latency_sec"]
            speedup = f"{baseline_lat / lat:.2f}x" if lat else "N/A"
            print(f"{method:<30} {data['size_mb']:<12.1f} {lat if lat else 'FAILED':<15} {speedup:<10}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare quantization methods for YOLO OBB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  python compare_quantization_methods.py \\
    --model yolo_obb_6class_h7.onnx \\
    --calibration-dir ./calibration_images \\
    --test-images test_img1.jpg test_img2.jpg test_img3.jpg

  # Run with more inference runs for statistical significance
  python compare_quantization_methods.py \\
    --model yolo_obb_6class_h7.onnx \\
    --calibration-dir ./calibration_images \\
    --test-dir ./test_images \\
    --runs 10 \\
    --warmup 5
        """,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to float32 ONNX model",
    )
    parser.add_argument(
        "--calibration-dir",
        required=True,
        help="Directory with calibration images",
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=100,
        help="Number of calibration images to use (default: 100)",
    )
    parser.add_argument(
        "--test-images",
        nargs="*",
        default=[],
        help="Test image paths",
    )
    parser.add_argument(
        "--test-dir",
        help="Directory with test images (alternative to --test-images)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Inference runs per test image (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs (default: 2)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.calibration_dir):
        print(f"Error: Calibration dir not found: {args.calibration_dir}")
        sys.exit(1)

    # Prepare calibration images
    calib_images = []
    for root, _, files in os.walk(args.calibration_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                calib_images.append(os.path.join(root, file))
        break  # Non-recursive

    if len(calib_images) > args.calibration_images:
        np.random.shuffle(calib_images)
        calib_images = calib_images[:args.calibration_images]

    print(f"Calibration images: {len(calib_images)}")

    # Prepare test images
    test_images = []
    if args.test_dir:
        if not os.path.exists(args.test_dir):
            print(f"Error: Test dir not found: {args.test_dir}")
            sys.exit(1)
        for root, _, files in os.walk(args.test_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(root, file))
            break
    elif args.test_images:
        test_images = args.test_images
    else:
        # Use calibration images as test images if none provided
        test_images = calib_images[:10]
        print("Note: Using first 10 calibration images as test set")

    if not test_images:
        print("Error: No test images found")
        sys.exit(1)

    print(f"Test images: {len(test_images)}")

    # Run comparison
    comp = QuantizationComparison(args.model, calib_images, test_images)
    comp.run_comparison(warmup=args.warmup, runs=args.runs)


if __name__ == "__main__":
    main()
