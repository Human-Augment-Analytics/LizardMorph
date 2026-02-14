
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import dlib
import time

# Class Definitions
CLASSES = {0: 'bot_finger', 1: 'bot_toe', 2: 'up_finger', 3: 'up_toe'}
COLORS = {0: (255,0,255), 1: (0,255,0), 2: (255,165,0), 3: (0,255,255)}

# Max dimension for YOLO detection (downsample large images)
YOLO_MAX_DIM = 4096


def get_padded_crop(img, corners, padding_ratio=0.3):
    """Get a padded axis-aligned crop from the image given OBB corners.
    Returns (crop, x_offset, y_offset) where offsets map crop coords back to original."""
    x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
    img_h, img_w = img.shape[:2]
    px = int(w * padding_ratio)
    py = int(h * padding_ratio)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(img_w, x + w + px)
    y2 = min(img_h, y + h + py)
    crop = img[y1:y2, x1:x2]
    return crop, x1, y1


def run_flip_inference_with_landmarks(model, finger_predictor, toe_predictor, img, conf=0.25):
    h, w = img.shape[:2]
    landmarks_list = []
    detections = []

    # Downsample for YOLO detection if image is large
    max_dim = max(h, w)
    if max_dim > YOLO_MAX_DIM:
        scale = YOLO_MAX_DIM / max_dim
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = img
    inv_scale = 1.0 / scale

    def _predict_landmarks(corners_orig, cls_id, source_img, is_flipped=False):
        """Run dlib on a cropped region. Returns (points_in_original, corners_orig)."""
        crop, x_off, y_off = get_padded_crop(source_img, corners_orig)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.uint8)
        crop_h, crop_w = crop.shape[:2]

        rect = dlib.rectangle(0, 0, crop_w, crop_h)
        predictor = finger_predictor if cls_id == 0 else toe_predictor
        shape = predictor(crop_rgb, rect)

        points = []
        for k in range(shape.num_parts):
            p = shape.part(k)
            px_orig = p.x + x_off
            py_orig = p.y + y_off
            if is_flipped:
                py_orig = h - 1 - py_orig
            points.append((px_orig, py_orig))
        return points

    # 1. Standard Inference (on downsampled image)
    results_orig = model.predict(small, imgsz=1280, conf=conf, verbose=False)[0]

    if results_orig.obb is not None:
        for i in range(len(results_orig.obb)):
            cls_id = int(results_orig.obb.cls[i])
            corners_small = results_orig.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)

            if cls_id in [0, 1]:
                # Scale corners back to original image coords
                corners_orig = corners_small * inv_scale
                points = _predict_landmarks(corners_orig, cls_id, img)

                landmarks_list.append({'cls': cls_id, 'points': points})
                detections.append({'cls': cls_id, 'corners': corners_orig})

    # 2. Flipped Inference (flip the downsampled image only)
    small_flipped = cv2.flip(small, 0)
    results_flipped = model.predict(small_flipped, imgsz=1280, conf=conf, verbose=False)[0]

    if results_flipped.obb is not None and len(results_flipped.obb) > 0:
        # Flip the full-res image once for all flipped detections
        img_flipped = cv2.flip(img, 0)

        for i in range(len(results_flipped.obb)):
            cls_id = int(results_flipped.obb.cls[i])
            corners_small = results_flipped.obb.xyxyxyxy[i].cpu().numpy().astype(np.float32)

            target_cls = None
            if cls_id == 0:
                target_cls = 2
            elif cls_id == 1:
                target_cls = 3

            if target_cls is not None:
                # Scale corners to full-res flipped image coords
                corners_flipped_orig = corners_small * inv_scale
                points = _predict_landmarks(corners_flipped_orig, cls_id, img_flipped, is_flipped=True)

                # Flip corners back to original image coords for visualization
                corners_orig = corners_flipped_orig.copy()
                corners_orig[:, 1] = h - 1 - corners_orig[:, 1]

                landmarks_list.append({'cls': target_cls, 'points': points})
                detections.append({'cls': target_cls, 'corners': corners_orig})

    return detections, landmarks_list


def draw_results(img, detections, landmarks_list):
    vis_img = img.copy()
    h, w = img.shape[:2]
    scale = max(w, h) / 2000
    line_thickness = max(2, int(4 * scale))
    circle_radius = 10

    for d in detections:
        cls_id = d['cls']
        corners = d['corners'].astype(np.int32)
        color = COLORS.get(cls_id, (255,255,255))
        cv2.polylines(vis_img, [corners], True, color, line_thickness)

    for l in landmarks_list:
        cls_id = l['cls']
        points = l['points']
        color = COLORS.get(cls_id, (255,255,255))
        for (x, y) in points:
            cv2.circle(vis_img, (int(x), int(y)), circle_radius, (0, 0, 255), -1)

    return vis_img


def main():
    parser = argparse.ArgumentParser(description="Run inference with flip strategy + landmarks")
    parser.add_argument('--model', default='../models/lizard-toe-pad/yolo_obb_2class.pt', help="Path to YOLO model")
    parser.add_argument('--finger-predictor', default='../models/lizard-toe-pad/finger_predictor_yolo_bbox.dat', help="Path to finger shape predictor")
    parser.add_argument('--toe-predictor', default='../models/lizard-toe-pad/toe_predictor_yolo_bbox.dat', help="Path to toe shape predictor")
    parser.add_argument('--source', required=True, help="Image file or directory")
    parser.add_argument('--output-dir', default='inference_results', help="Output directory")
    args = parser.parse_args()

    print(f"Loading YOLO model from {args.model}")
    model = YOLO(args.model)

    print(f"Loading Finger Predictor from {args.finger_predictor}")
    finger_predictor = dlib.shape_predictor(args.finger_predictor)

    print(f"Loading Toe Predictor from {args.toe_predictor}")
    toe_predictor = dlib.shape_predictor(args.toe_predictor)

    source_path = Path(args.source)
    if source_path.is_dir():
        all_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        files_to_process = [f for f in all_files if "_landmarks" not in f.name]
    else:
        files_to_process = [source_path]

    print(f"Found {len(files_to_process)} files to process.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for current_file in files_to_process:
        print(f"Processing: {current_file}")
        try:
            t0 = time.time()
            img = cv2.imread(str(current_file))
            if img is None:
                print(f"Error: Could not read image {current_file}")
                continue

            detections, landmarks = run_flip_inference_with_landmarks(model, finger_predictor, toe_predictor, img)
            vis_img = draw_results(img, detections, landmarks)

            out_path = output_dir / current_file.name
            cv2.imwrite(str(out_path), vis_img)
            elapsed = time.time() - t0
            print(f"Saved result to {out_path} ({elapsed:.1f}s, {len(detections)} detections)")
        except Exception as e:
            print(f"Error processing {current_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
