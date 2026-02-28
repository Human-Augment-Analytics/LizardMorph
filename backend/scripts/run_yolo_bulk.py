"""
run_yolo_bulk.py
----------------
Run YOLO OBB inference on all .jpg files in a directory and report
whether id (class 5) and ruler (class 4) were detected per image.

Usage:
    conda run -n lizard python scripts/run_yolo_bulk.py <data_dir> [--model <model_path>] [--limit N]
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO

DEFAULT_MODEL = "../models/lizard-toe-pad/yolo_obb_6class_h7.pt"

def main():
    parser = argparse.ArgumentParser(description="Bulk YOLO OBB inference on a directory")
    parser.add_argument("data_dir", help="Directory containing .jpg images")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N files (0 = all)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"Model classes: {model.names}\n")

    files = sorted(f for f in os.listdir(args.data_dir) if f.lower().endswith(".jpg"))
    if args.limit:
        files = files[:args.limit]

    for filename in files:
        image_path = os.path.join(args.data_dir, filename)
        results = model(image_path, conf=args.conf, device='cpu', verbose=False)

        found_id = found_ruler = False
        for result in results:
            if result.obb is not None:
                for i in range(len(result.obb)):
                    cls_id = int(result.obb.cls[i].item())
                    if cls_id == 5:
                        found_id = True
                    if cls_id == 4:
                        found_ruler = True

        status = []
        if found_id: status.append("✓ id")
        else:         status.append("✗ id")
        if found_ruler: status.append("✓ ruler")
        else:           status.append("✗ ruler")
        print(f"{filename}: {', '.join(status)}")

if __name__ == "__main__":
    main()
