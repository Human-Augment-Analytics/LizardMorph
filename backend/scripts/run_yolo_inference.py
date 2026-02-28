"""
run_yolo_inference.py
---------------------
Run YOLO OBB inference on a single image and print all detections.

Usage:
    conda run -n lizard python scripts/run_yolo_inference.py <image_path> [--model <model_path>]

Example:
    conda run -n lizard python scripts/run_yolo_inference.py /path/to/1001.jpg
"""
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

DEFAULT_MODEL = "../models/lizard-toe-pad/yolo_obb_6class_h7.pt"

def main():
    parser = argparse.ArgumentParser(description="Run YOLO OBB inference on an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"Model classes: {model.names}")

    print(f"\nRunning inference on: {args.image}")
    results = model(args.image, conf=args.conf, device='cpu', verbose=False)

    for idx, result in enumerate(results):
        if result.obb is not None and len(result.obb) > 0:
            print(f"\nFound {len(result.obb)} OBB detections:")
            for i in range(len(result.obb)):
                cls_id = int(result.obb.cls[i].item())
                conf = float(result.obb.conf[i].item())
                cls_name = model.names.get(cls_id, "unknown")
                print(f"  [{i}] {cls_name} (id={cls_id})  conf={conf:.3f}")
        else:
            print("No OBB detections found.")

if __name__ == "__main__":
    main()
