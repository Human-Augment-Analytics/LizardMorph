"""
run_yolo_legacy.py
------------------
Run the legacy (non-OBB) YOLO bounding-box model on an image.
Useful for comparing the old model's detections against the new OBB model.

Usage:
    conda run -n lizard python scripts/run_yolo_legacy.py <image_path> [--model <model_path>]
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

DEFAULT_MODEL = "../models/lizard-toe-pad/yolo_bounding_box.pt"

def main():
    parser = argparse.ArgumentParser(description="Run legacy (non-OBB) YOLO inference on an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    print(f"Loading legacy model: {args.model}")
    model = YOLO(args.model)
    print(f"Model classes: {model.names}\n")

    results = model(args.image, conf=args.conf, device='cpu', verbose=False)

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"Found {len(result.boxes)} detections:")
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                conf = float(result.boxes.conf[i].item())
                cls_name = model.names.get(cls_id, "unknown")
                print(f"  [{i}] {cls_name} (id={cls_id})  conf={conf:.3f}")
        else:
            print("No detections found.")

if __name__ == "__main__":
    main()
