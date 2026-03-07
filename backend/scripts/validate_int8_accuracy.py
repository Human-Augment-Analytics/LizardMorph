#!/usr/bin/env python3
"""
Validate INT8 quantized YOLO model accuracy vs float32 baseline.

Compares detection results between float32 (ONNX) and INT8 (OpenVINO IR)
on test images to ensure quantization accuracy loss is acceptable.

Usage:
    python validate_int8_accuracy.py \
        --fp32-model models/yolo_obb_6class_h7.onnx \
        --int8-model models/yolo_obb_6class_h7_int8_nncf.xml \
        --test-images ./test_images \
        --threshold 2.0  # Allow 2% detection difference

Typical results after NNCF quantization:
    - Perfect match: ~70-80% of images
    - Minor differences (±1 detection): ~15-20%
    - Significant differences (>2 detections): <5%
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from openvino import Core
    import onnxruntime as ort
except ImportError as e:
    print(f"Error: Missing required packages. Install with:")
    print("  pip install openvino onnxruntime")
    sys.exit(1)

# YOLO model constants (match yolo_obb_6class_h7)
INPUT_SIZE = 1280
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASS_NAMES = {
    0: "up_finger",
    1: "up_toe",
    2: "bot_finger",
    3: "bot_toe",
    4: "ruler",
    5: "id",
}


def preprocess_image(img_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """
    Preprocess image for YOLO ONNX model.

    Returns:
        tensor: (1, 3, 1280, 1280) float32
        scale: Resize scale factor
        x_pad: X padding offset
        y_pad: Y padding offset
    """
    h, w = img_bgr.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    x_pad = (INPUT_SIZE - new_w) // 2
    y_pad = (INPUT_SIZE - new_h) // 2
    padded[y_pad:y_pad + new_h, x_pad:x_pad + new_w] = resized

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)

    return tensor[np.newaxis, ...], scale, x_pad, y_pad


def parse_detections(
    output: np.ndarray, conf_threshold: float = CONF_THRESHOLD
) -> List[Dict]:
    """
    Parse raw YOLO model output [1, 11, 33600] into detections.

    Channels: 0=cx, 1=cy, 2=w, 3=h, 4-9=class_confs, 10=angle
    """
    data = output[0]  # [11, 33600]

    class_confs = data[4:10]  # [6, 33600]
    max_class_ids = np.argmax(class_confs, axis=0)
    max_confs = np.max(class_confs, axis=0)

    mask = max_confs > conf_threshold
    indices = np.where(mask)[0]

    boxes = []
    for i in indices:
        boxes.append(
            {
                "x": float(data[0, i]),
                "y": float(data[1, i]),
                "w": float(data[2, i]),
                "h": float(data[3, i]),
                "angle": float(data[10, i]),
                "conf": float(max_confs[i]),
                "class_id": int(max_class_ids[i]),
                "class_name": CLASS_NAMES.get(int(max_class_ids[i]), "unknown"),
            }
        )

    return sorted(boxes, key=lambda b: b["conf"], reverse=True)


def iou_aabb(box_a: Dict, box_b: Dict) -> float:
    """Compute AABB IoU (axis-aligned, ignores rotation)."""
    a_min_x = box_a["x"] - box_a["w"] / 2
    a_max_x = box_a["x"] + box_a["w"] / 2
    a_min_y = box_a["y"] - box_a["h"] / 2
    a_max_y = box_a["y"] + box_a["h"] / 2

    b_min_x = box_b["x"] - box_b["w"] / 2
    b_max_x = box_b["x"] + box_b["w"] / 2
    b_min_y = box_b["y"] - box_b["h"] / 2
    b_max_y = box_b["y"] + box_b["h"] / 2

    inter_x1 = max(a_min_x, b_min_x)
    inter_y1 = max(a_min_y, b_min_y)
    inter_x2 = min(a_max_x, b_max_x)
    inter_y2 = min(a_max_y, b_max_y)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = box_a["w"] * box_a["h"]
    b_area = box_b["w"] * box_b["h"]

    return inter_area / (a_area + b_area - inter_area)


def nms_and_top_one(boxes: List[Dict]) -> List[Dict]:
    """Apply NMS per class, keep top-1 per class."""
    boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)

    after_nms = []
    for b in boxes:
        overlap = False
        for k in after_nms:
            if k["class_id"] == b["class_id"] and iou_aabb(b, k) > IOU_THRESHOLD:
                overlap = True
                break
        if not overlap:
            after_nms.append(b)

    seen = set()
    result = []
    for b in after_nms:
        if b["class_id"] not in seen:
            seen.add(b["class_id"])
            result.append(b)

    return result


def run_onnx_inference(onnx_path: str, tensor: np.ndarray) -> np.ndarray:
    """Run inference with ORT float32 ONNX."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: tensor})[0]
    return output


def run_openvino_inference(openvino_path: str, tensor: np.ndarray) -> np.ndarray:
    """Run inference with OpenVINO INT8."""
    core = Core()
    model = core.read_model(openvino_path)
    compiled = core.compile_model(model, "CPU")
    infer_request = compiled.create_infer_request()

    # Get input tensor and copy data
    input_tensor = infer_request.get_input_tensor()
    input_tensor.data[:] = tensor

    # Run inference
    infer_request.infer()

    # Get output
    output_tensor = infer_request.get_output_tensor()
    return output_tensor.data.copy()


def validate_models(
    fp32_model_path: str,
    int8_model_path: str,
    test_images_dir: str,
    threshold_pct: float = 2.0,
    max_images: int = 100,
) -> None:
    """
    Validate INT8 vs FP32 accuracy.

    Args:
        fp32_model_path: Path to float32 ONNX model
        int8_model_path: Path to INT8 OpenVINO IR model (.xml)
        test_images_dir: Directory containing test images
        threshold_pct: Acceptable detection difference percentage
        max_images: Maximum number of test images
    """

    print("=" * 70)
    print("INT8 Quantization Validation")
    print("=" * 70)

    # Step 1: Validate inputs
    print(f"\n[1/5] Validating inputs...")
    if not os.path.exists(fp32_model_path):
        print(f"Error: FP32 model not found: {fp32_model_path}")
        sys.exit(1)

    if not os.path.exists(int8_model_path):
        print(f"Error: INT8 model not found: {int8_model_path}")
        sys.exit(1)

    if not os.path.isdir(test_images_dir):
        print(f"Error: Test images directory not found: {test_images_dir}")
        sys.exit(1)

    print(f"  FP32 model: {fp32_model_path}")
    print(f"  INT8 model: {int8_model_path}")
    print(f"  Test dir:   {test_images_dir}")

    # Step 2: Load test images
    print(f"\n[2/5] Loading test images...")
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []

    for fname in sorted(os.listdir(test_images_dir)):
        if Path(fname).suffix.lower() not in valid_exts:
            continue
        img_path = os.path.join(test_images_dir, fname)
        img = cv2.imread(img_path)
        if img is not None:
            image_paths.append(img_path)
            if len(image_paths) >= max_images:
                break

    if not image_paths:
        print(f"Error: No valid images found in {test_images_dir}")
        sys.exit(1)

    print(f"  Loaded {len(image_paths)} images")

    # Step 3: Run inference on all images
    print(f"\n[3/5] Running inference on FP32...")
    fp32_results = []
    fp32_times = []

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        tensor, _, _, _ = preprocess_image(img)

        start = time.perf_counter()
        fp32_output = run_onnx_inference(fp32_model_path, tensor)
        fp32_time = time.perf_counter() - start

        fp32_dets = nms_and_top_one(parse_detections(fp32_output, CONF_THRESHOLD))
        fp32_results.append(fp32_dets)
        fp32_times.append(fp32_time)

        if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
            print(f"  [{i+1}/{len(image_paths)}] {len(fp32_dets)} detections")

    print(f"\n[4/5] Running inference on INT8...")
    int8_results = []
    int8_times = []

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        tensor, _, _, _ = preprocess_image(img)

        start = time.perf_counter()
        int8_output = run_openvino_inference(int8_model_path, tensor)
        int8_time = time.perf_counter() - start

        int8_dets = nms_and_top_one(parse_detections(int8_output, CONF_THRESHOLD))
        int8_results.append(int8_dets)
        int8_times.append(int8_time)

        if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
            print(f"  [{i+1}/{len(image_paths)}] {len(int8_dets)} detections")

    # Step 4: Analyze differences
    print(f"\n[5/5] Analyzing results...")

    exact_matches = 0
    minor_diffs = 0
    major_diffs = 0
    total_fp32_dets = 0
    total_int8_dets = 0
    total_diff = 0

    det_diff_by_class = {}

    for i, (fp32_dets, int8_dets) in enumerate(zip(fp32_results, int8_results)):
        img_name = os.path.basename(image_paths[i])

        total_fp32_dets += len(fp32_dets)
        total_int8_dets += len(int8_dets)

        diff = abs(len(fp32_dets) - len(int8_dets))
        total_diff += diff

        if diff == 0:
            exact_matches += 1
        elif diff <= 1:
            minor_diffs += 1
        else:
            major_diffs += 1

        # Track per-class differences
        for det in fp32_dets:
            cls = det["class_name"]
            if cls not in det_diff_by_class:
                det_diff_by_class[cls] = {"fp32": 0, "int8": 0}
            det_diff_by_class[cls]["fp32"] += 1

        for det in int8_dets:
            cls = det["class_name"]
            if cls not in det_diff_by_class:
                det_diff_by_class[cls] = {"fp32": 0, "int8": 0}
            det_diff_by_class[cls]["int8"] += 1

    # Step 5: Print results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    num_images = len(image_paths)
    avg_fp32_time = np.mean(fp32_times) * 1000
    avg_int8_time = np.mean(int8_times) * 1000
    speedup = avg_fp32_time / avg_int8_time if avg_int8_time > 0 else 0

    print(f"\nImages tested: {num_images}")
    print(f"\nDetection counts:")
    print(f"  Total FP32 detections: {total_fp32_dets}")
    print(f"  Total INT8 detections: {total_int8_dets}")
    print(f"  Difference: {abs(total_fp32_dets - total_int8_dets)} ({abs(total_fp32_dets - total_int8_dets) / total_fp32_dets * 100:.1f}%)")

    print(f"\nDetection consistency:")
    print(f"  Exact matches: {exact_matches} ({exact_matches / num_images * 100:.1f}%)")
    print(f"  Minor diffs (±1): {minor_diffs} ({minor_diffs / num_images * 100:.1f}%)")
    print(f"  Major diffs (>1): {major_diffs} ({major_diffs / num_images * 100:.1f}%)")

    print(f"\nPerformance:")
    print(f"  FP32 avg: {avg_fp32_time:.1f} ms/image")
    print(f"  INT8 avg: {avg_int8_time:.1f} ms/image")
    print(f"  Speedup:  {speedup:.2f}x")

    print(f"\nPer-class results:")
    for cls in sorted(det_diff_by_class.keys()):
        fp32_cnt = det_diff_by_class[cls]["fp32"]
        int8_cnt = det_diff_by_class[cls]["int8"]
        pct_diff = (
            abs(fp32_cnt - int8_cnt) / max(fp32_cnt, 1) * 100
            if fp32_cnt > 0
            else 0
        )
        print(f"  {cls:15s}: FP32={fp32_cnt:2d}, INT8={int8_cnt:2d}, diff={pct_diff:.1f}%")

    # Step 6: Verdict
    print("\n" + "=" * 70)
    accuracy_loss_pct = abs(total_fp32_dets - total_int8_dets) / total_fp32_dets * 100

    if accuracy_loss_pct <= threshold_pct:
        print(f"✓ PASS: Accuracy loss {accuracy_loss_pct:.1f}% is within threshold {threshold_pct}%")
        return 0
    else:
        print(f"✗ FAIL: Accuracy loss {accuracy_loss_pct:.1f}% exceeds threshold {threshold_pct}%")
        print(f"\n  Recommendation:")
        print(f"    1. Increase calibration samples to 500+")
        print(f"    2. Use --preset MIXED instead of PERFORMANCE")
        print(f"    3. Try --no-fast-bias for more conservative quantization")
        print(f"    4. Manually review mismatches (may be noise in float32 too)")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate INT8 quantized model accuracy vs FP32 baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_int8_accuracy.py \\
    --fp32-model models/yolo_obb_6class_h7.onnx \\
    --int8-model models/yolo_obb_6class_h7_int8_nncf.xml \\
    --test-images ./test_images \\
    --threshold 2.0

Expected output (good quantization):
  - Exact matches: >70%
  - Minor diffs: 15-25%
  - Major diffs: <5%
  - Accuracy loss: <2%
        """,
    )

    parser.add_argument(
        "--fp32-model",
        required=True,
        help="Path to float32 ONNX model",
    )
    parser.add_argument(
        "--int8-model",
        required=True,
        help="Path to INT8 OpenVINO IR model (.xml)",
    )
    parser.add_argument(
        "--test-images",
        required=True,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Acceptable accuracy loss percentage (default: 2.0%%)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Maximum test images to process (default: 100)",
    )

    args = parser.parse_args()

    sys.exit(
        validate_models(
            fp32_model_path=args.fp32_model,
            int8_model_path=args.int8_model,
            test_images_dir=args.test_images,
            threshold_pct=args.threshold,
            max_images=args.max_images,
        )
    )


if __name__ == "__main__":
    main()
