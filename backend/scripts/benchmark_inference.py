#!/usr/bin/env python3
"""Benchmark YOLO OBB inference across 4 backends:
1. Ultralytics PyTorch (.pt)
2. Ultralytics ONNX float32 (.onnx)
3. ORT INT8 quantized (.onnx)
4. OpenVINO (.onnx)

All benchmarks measure dual-pass inference (normal + flipped).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models/lizard-toe-pad")


def _downsample(img_bgr, max_dim=4096):
    """Downsample large images (matching production behavior)."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr


def benchmark_ultralytics(model_path, image_paths, warmup=2, runs=5):
    """Benchmark Ultralytics YOLO inference (dual-pass). Works with .pt or .onnx."""
    from ultralytics import YOLO

    model = YOLO(model_path, task="obb")
    images_bgr = [_downsample(cv2.imread(p)) for p in image_paths]
    images_pil = [PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_bgr]

    # Warmup
    for _ in range(warmup):
        model(images_pil[0], conf=0.25, imgsz=1280, device="cpu", verbose=False)

    times = []
    all_detections = []
    for img_bgr, img_pil in zip(images_bgr, images_pil):
        img_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            results = model(img_pil, conf=0.25, imgsz=1280, device="cpu", verbose=False)
            flipped_pil = PILImage.fromarray(cv2.cvtColor(cv2.flip(img_bgr, 0), cv2.COLOR_BGR2RGB))
            model(flipped_pil, conf=0.25, imgsz=1280, device="cpu", verbose=False)
            t1 = time.perf_counter()
            img_times.append(t1 - t0)

        times.append(np.median(img_times))
        res = results[0]
        det_summary = []
        if res.obb is not None:
            for idx in range(len(res.obb)):
                det_summary.append((res.names[int(res.obb.cls[idx].item())], round(float(res.obb.conf[idx].item()), 3)))
        all_detections.append(det_summary)

    return times, all_detections


def benchmark_ort(model_path, image_paths, warmup=2, runs=5):
    """Benchmark ORT inference (dual-pass). Works with float32 or INT8 ONNX."""
    from ort_inference import OrtYoloDetector

    detector = OrtYoloDetector(model_path)
    images_bgr = [cv2.imread(p) for p in image_paths]

    # Warmup
    for _ in range(warmup):
        detector.detect(images_bgr[0])

    times = []
    all_detections = []
    for img_bgr in images_bgr:
        img_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            detections = detector.detect(img_bgr)
            t1 = time.perf_counter()
            img_times.append(t1 - t0)

        times.append(np.median(img_times))
        det_summary = [(cat, round(d["conf"], 3)) for cat, dets in detections.items() for d in dets]
        all_detections.append(det_summary)

    return times, all_detections


def benchmark_openvino(model_path, image_paths, warmup=2, runs=5):
    """Benchmark OpenVINO inference (dual-pass) using same pre/post-processing as ORT."""
    from openvino import Core
    from ort_inference import OrtYoloDetector, INPUT_SIZE, CONF_THRESHOLD, IOU_THRESHOLD

    # Create a thin detector that uses OpenVINO for inference but ORT's pre/post-processing
    core = Core()
    ov_model = core.read_model(model_path)
    compiled = core.compile_model(ov_model, "CPU")
    infer_request = compiled.create_infer_request()

    # Use OrtYoloDetector for pre/post-processing helpers only
    # We'll create an instance but override _run_inference
    helper = OrtYoloDetector.__new__(OrtYoloDetector)
    # Copy class methods we need (they're all static or don't need session)

    images_bgr = [cv2.imread(p) for p in image_paths]

    def detect_openvino(img_bgr):
        """Dual-pass detect using OpenVINO inference."""
        h_img, w_img = img_bgr.shape[:2]
        max_dim = max(h_img, w_img)
        if max_dim > 4096:
            ds_scale = 4096 / max_dim
            small_bgr = cv2.resize(img_bgr, (int(w_img * ds_scale), int(h_img * ds_scale)), interpolation=cv2.INTER_AREA)
        else:
            ds_scale = 1.0
            small_bgr = img_bgr
        inv_scale = 1.0 / ds_scale

        def preprocess(bgr):
            h, w = bgr.shape[:2]
            scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
            new_w, new_h = round(w * scale), round(h * scale)
            resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
            x_pad = (INPUT_SIZE - new_w) // 2
            y_pad = (INPUT_SIZE - new_h) // 2
            padded[y_pad:y_pad + new_h, x_pad:x_pad + new_w] = resized
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            return tensor[np.newaxis, ...], scale, x_pad, y_pad

        def run_ov(tensor):
            infer_request.infer({0: tensor})
            return infer_request.get_output_tensor(0).data.copy()

        # Normal pass
        tensor, scale, x_pad, y_pad = preprocess(small_bgr)
        raw_normal = run_ov(tensor)
        normal_boxes = OrtYoloDetector._nms_and_top_one(OrtYoloDetector._parse_detections(raw_normal, CONF_THRESHOLD))

        # Flipped pass
        tensor_flip, _, _, _ = preprocess(cv2.flip(small_bgr, 0))
        raw_flip = run_ov(tensor_flip)
        flip_boxes = OrtYoloDetector._nms_and_top_one(OrtYoloDetector._parse_detections(raw_flip, CONF_THRESHOLD))

        # Map to detections dict (same as OrtYoloDetector.detect)
        from ort_inference import CLASS_NAMES
        detections = {'bot_finger': [], 'bot_toe': [], 'up_finger': [], 'up_toe': [], 'scale': [], 'id': []}

        for box in normal_boxes:
            cls_name = CLASS_NAMES.get(box["class_id"], "")
            corners = OrtYoloDetector._get_obb_corners(box["x"], box["y"], box["w"], box["h"], box["angle"])
            corners_orig = (corners - np.array([x_pad, y_pad])) / scale * inv_scale
            obb_w = box["w"] / scale * inv_scale
            obb_h = box["h"] / scale * inv_scale
            if "ruler" in cls_name or "scale" in cls_name:
                detections["scale"].append({"conf": box["conf"], "corners": corners_orig, "obb_wh": (obb_w, obb_h)})
            elif cls_name == "bot_finger":
                detections["bot_finger"].append({"conf": box["conf"], "corners": corners_orig})
            elif cls_name == "bot_toe":
                detections["bot_toe"].append({"conf": box["conf"], "corners": corners_orig})
            elif cls_name == "id":
                detections["id"].append({"conf": box["conf"], "corners": corners_orig})

        for box in flip_boxes:
            cls_name = CLASS_NAMES.get(box["class_id"], "")
            corners = OrtYoloDetector._get_obb_corners(box["x"], box["y"], box["w"], box["h"], box["angle"])
            corners_flip = (corners - np.array([x_pad, y_pad])) / scale * inv_scale
            if cls_name == "bot_finger":
                detections["up_finger"].append({"conf": box["conf"], "corners": corners_flip})
            elif cls_name == "bot_toe":
                detections["up_toe"].append({"conf": box["conf"], "corners": corners_flip})

        return detections

    # Warmup
    for _ in range(warmup):
        detect_openvino(images_bgr[0])

    times = []
    all_detections = []
    for img_bgr in images_bgr:
        img_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            detections = detect_openvino(img_bgr)
            t1 = time.perf_counter()
            img_times.append(t1 - t0)

        times.append(np.median(img_times))
        det_summary = [(cat, round(d["conf"], 3)) for cat, dets in detections.items() for d in dets]
        all_detections.append(det_summary)

    return times, all_detections


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO OBB: PyTorch vs ONNX vs INT8 vs OpenVINO")
    parser.add_argument("images", nargs="+", help="Image file paths to benchmark")
    parser.add_argument("--pt-model", default=os.path.join(MODELS_DIR, "yolo_obb_6class_h7.pt"))
    parser.add_argument("--onnx-model", default=os.path.join(MODELS_DIR, "yolo_obb_6class_h7.onnx"))
    parser.add_argument("--int8-model", default=os.path.join(MODELS_DIR, "yolo_obb_6class_h7_int8.onnx"))
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs per image")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--threads", type=int, default=0, help="Limit CPU threads (0=auto)")
    args = parser.parse_args()

    if args.threads > 0:
        import torch
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        print(f"Thread limit: {args.threads}")

    for path in args.images:
        if not os.path.exists(path):
            print(f"Error: Image not found: {path}")
            sys.exit(1)

    print(f"Benchmarking {len(args.images)} image(s), {args.runs} runs each, {args.warmup} warmup")
    print(f"{'=' * 70}")

    results = {}

    # 1. Ultralytics PyTorch
    if os.path.exists(args.pt_model):
        print(f"\n--- 1. Ultralytics PyTorch (.pt) ---")
        results["PyTorch"] = benchmark_ultralytics(args.pt_model, args.images, args.warmup, args.runs)
    else:
        print(f"\nSkipping PyTorch: {args.pt_model} not found")

    # 2. Ultralytics ONNX float32
    if os.path.exists(args.onnx_model):
        print(f"\n--- 2. Ultralytics ONNX float32 ---")
        results["ONNX-f32"] = benchmark_ultralytics(args.onnx_model, args.images, args.warmup, args.runs)
    else:
        print(f"\nSkipping ONNX float32: {args.onnx_model} not found")

    # 3. ORT INT8 quantized
    if os.path.exists(args.int8_model):
        print(f"\n--- 3. ORT ONNX INT8 ---")
        results["ORT-INT8"] = benchmark_ort(args.int8_model, args.images, args.warmup, args.runs)
    else:
        print(f"\nSkipping ORT INT8: {args.int8_model} not found")

    # 4. OpenVINO
    if os.path.exists(args.onnx_model):
        print(f"\n--- 4. OpenVINO ---")
        results["OpenVINO"] = benchmark_openvino(args.onnx_model, args.images, args.warmup, args.runs)
    else:
        print(f"\nSkipping OpenVINO: {args.onnx_model} not found")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS (median dual-pass time per image)")
    print(f"{'=' * 70}")

    for i, path in enumerate(args.images):
        name = os.path.basename(path)
        print(f"\n{name}:")
        baseline = None
        for label, (times, dets) in results.items():
            t = times[i]
            if baseline is None:
                baseline = t
            speedup = baseline / t if t > 0 else float("inf")
            print(f"  {label:12s}: {t:.3f}s  ({speedup:.2f}x vs first)")
            print(f"    detections: {dets[i]}")

    if len(args.images) > 1:
        print(f"\nOverall average:")
        baseline_avg = None
        for label, (times, _) in results.items():
            avg = np.mean(times)
            if baseline_avg is None:
                baseline_avg = avg
            print(f"  {label:12s}: {avg:.3f}s  ({baseline_avg / avg:.2f}x)")


if __name__ == "__main__":
    main()
