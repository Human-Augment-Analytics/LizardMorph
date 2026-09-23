import os
import math
import shutil
import csv
import json
import random
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional

try:
    from backend.datasets.canonical import CanonicalDataset
except ImportError:
    from datasets.canonical import CanonicalDataset


def export_model_to_onnx(model) -> str:
    """Export through Torch's source-independent legacy ONNX path.

    Torch 2.9 defaults ``torch.onnx.export`` to the dynamo/onnxscript exporter.
    PyInstaller cannot provide Python source for onnxscript's decorated torch-lib
    functions, so that exporter fails inside the bundled Tauri sidecar.  The
    legacy exporter is fully supported by Ultralytics and does not inspect those
    source files.
    """
    import torch

    original_export = torch.onnx.export

    def legacy_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return original_export(*args, **kwargs)

    torch.onnx.export = legacy_export
    try:
        return str(model.export(format="onnx", simplify=False))
    finally:
        torch.onnx.export = original_export


def resolve_image_path(file_path: Optional[str]) -> Optional[str]:
    if not file_path:
        return None
    if os.path.exists(file_path):
        return os.path.abspath(file_path)
    base = os.path.basename(file_path)
    candidates = [
        os.path.join("uploads", base),
        os.path.join("uploads", file_path),
        os.path.join("ml_morph_examples", "image-examples", base),
        os.path.join("data", base),
        os.path.join("backend", base),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


class YoloOBBTrainer:
    """Trainer adapter for formatting CanonicalDataset into YOLO OBB format and running detector training."""

    def __init__(self, job_dir: str):
        self.job_dir = job_dir

    def prepare_dataset(
        self,
        canonical_dataset: CanonicalDataset,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> str:
        """
        Converts CanonicalDataset into YOLO OBB format dataset directory structure.

        Args:
            canonical_dataset: Source dataset.
            train_ratio: Ratio of images assigned to training set.

        Returns:
            Absolute path to generated data.yaml file.
        """
        yolo_dir = os.path.join(self.job_dir, "yolo_dataset")
        train_img_dir = os.path.join(yolo_dir, "images", "train")
        val_img_dir = os.path.join(yolo_dir, "images", "val")
        train_lbl_dir = os.path.join(yolo_dir, "labels", "train")
        val_lbl_dir = os.path.join(yolo_dir, "labels", "val")

        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        os.makedirs(val_lbl_dir, exist_ok=True)

        # Collect unique class names
        class_names = set()
        for img in canonical_dataset.images:
            for obj in img.objects:
                if obj.class_name:
                    class_names.add(obj.class_name)
        
        sorted_classes = sorted(list(class_names)) if class_names else ["object"]
        class_map = {name: idx for idx, name in enumerate(sorted_classes)}

        num_images = len(canonical_dataset.images)
        if num_images < 2:
            raise ValueError("YOLO training requires at least two annotated images.")

        num_train = min(num_images - 1, max(1, int(round(num_images * train_ratio))))
        indices = list(range(num_images))
        random.Random(seed).shuffle(indices)
        train_indices = set(indices[:num_train])

        for idx, cimg in enumerate(canonical_dataset.images):
            split = "train" if idx in train_indices else "val"
            img_dir = train_img_dir if split == "train" else val_img_dir
            lbl_dir = train_lbl_dir if split == "train" else val_lbl_dir

            base_name = f"image_{idx + 1}"
            actual_img_path = resolve_image_path(cimg.file_path)
            if not actual_img_path or not os.path.exists(actual_img_path):
                raise FileNotFoundError(
                    f"Image '{cimg.file_path}' referenced by the annotations was not provided."
                )

            ext = os.path.splitext(actual_img_path)[1] or ".jpg"
            dest_img_name = f"{base_name}{ext}"
            dest_img_path = os.path.join(img_dir, dest_img_name)
            shutil.copy(actual_img_path, dest_img_path)

            # Determine actual dimensions
            loaded_img = cv2.imread(dest_img_path)
            if loaded_img is None:
                raise ValueError(f"Unable to decode training image '{cimg.file_path}'.")
            img_h, img_w = loaded_img.shape[:2]

            # Write OBB labels
            lbl_filename = f"{base_name}.txt"
            lbl_path = os.path.join(lbl_dir, lbl_filename)

            label_lines = []
            for obj in cimg.objects:
                class_id = class_map.get(obj.class_name, 0)
                obb = obj.obb
                if len(obb) < 5:
                    continue

                cx, cy, w, h, angle_deg = obb[:5]
                angle_rad = math.radians(angle_deg)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                hw = w / 2.0
                hh = h / 2.0
                local_corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

                norm_pts = []
                for lx, ly in local_corners:
                    rx = cx + lx * cos_a - ly * sin_a
                    ry = cy + lx * sin_a + ly * cos_a
                    nx = min(1.0, max(0.0, rx / float(img_w)))
                    ny = min(1.0, max(0.0, ry / float(img_h)))
                    norm_pts.extend([nx, ny])

                line = f"{class_id} " + " ".join([f"{val:.6f}" for val in norm_pts]) + "\n"
                label_lines.append(line)

            with open(lbl_path, "w", encoding="utf-8") as f:
                f.writelines(label_lines)

            if not label_lines:
                raise ValueError(
                    f"Image '{cimg.file_path}' has no valid oriented bounding-box annotations."
                )

        # Generate data.yaml
        yaml_path = os.path.join(yolo_dir, "data.yaml")
        names_dict_str = "\n".join(
            [f"  {idx}: {json.dumps(name)}" for name, idx in class_map.items()]
        )
        
        yaml_content = f"""path: {json.dumps(os.path.abspath(yolo_dir))}
train: images/train
val: images/val
names:
{names_dict_str}
"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        with open(os.path.join(yolo_dir, "classes.json"), "w", encoding="utf-8") as f:
            json.dump(sorted_classes, f, indent=2)

        return yaml_path

    def train(
        self, data_yaml_path: str, epochs: int = 10, mock: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Executes YOLO OBB training, or an explicit test-only mock when requested.

        Args:
            data_yaml_path: Path to data.yaml file.
            epochs: Number of training epochs.
            mock: If True, write explicit test fixtures instead of training.

        Returns:
            Dictionary containing metrics and detector weights path.
        """
        weights_dir = os.path.join(self.job_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        detector_weights = os.path.join(weights_dir, "detector_yolo_obb.onnx")

        if mock:
            # Explicit test-only mock. Production failures are never converted to models.
            with open(detector_weights, "wb") as f:
                f.write(b"EXPLICIT_TEST_MOCK_ONNX")

            return {
                "detector_weights": detector_weights,
                "detector_weights_pt": detector_weights,
                "mAP50": 0.95,
                "mAP50-95": 0.85,
                "epochs": epochs,
                "mock": True,
            }

        from ultralytics import YOLO

        training_dir = os.path.join(self.job_dir, "yolo_obb_train")
        best_weights = os.path.join(training_dir, "weights", "best.pt")
        results_csv = os.path.join(training_dir, "results.csv")

        # A backend/package restart can happen after Ultralytics has completed
        # every epoch but before ONNX export. Reuse that validated checkpoint
        # instead of throwing away a potentially long detector run.
        completed_rows = []
        if os.path.exists(best_weights) and os.path.exists(results_csv):
            try:
                with open(results_csv, "r", encoding="utf-8", newline="") as f:
                    completed_rows = list(csv.DictReader(f))
            except (OSError, csv.Error):
                completed_rows = []
        if len(completed_rows) >= epochs:
            checkpoint_model = YOLO(best_weights)
            exported_path = export_model_to_onnx(checkpoint_model)
            if not os.path.exists(exported_path):
                raise FileNotFoundError("YOLO ONNX export did not produce an artifact.")
            shutil.copy2(exported_path, detector_weights)
            final_metrics = completed_rows[-1]

            def metric(name):
                value = final_metrics.get(name)
                return float(value) if value not in (None, "") else None

            return {
                "detector_weights": detector_weights,
                "mAP50": metric("metrics/mAP50(B)"),
                "mAP50-95": metric("metrics/mAP50-95(B)"),
                "detector_metrics": {
                    key: metric(key)
                    for key in final_metrics
                    if key.startswith("metrics/")
                },
                "epochs": epochs,
                "recovered_from_checkpoint": True,
            }

        base_model_arg = kwargs.pop("base_model", None)
        if not base_model_arg:
            candidates = [
                os.path.join(getattr(sys, "_MEIPASS", ""), "models", "lizard-toe-pad", "yolov8n-obb.pt"),
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "lizard-toe-pad", "yolov8n-obb.pt"),
                os.path.abspath("yolov8n-obb.pt"),
            ]
            for cand in candidates:
                if cand and os.path.isfile(cand):
                    base_model_arg = cand
                    break
            if not base_model_arg:
                base_model_arg = "yolov8n-obb.pt"
        model = YOLO(base_model_arg)

        def on_epoch_end(trainer):
            try:
                curr_epoch = getattr(trainer, "epoch", 0) + 1
                tot_epochs = getattr(trainer, "epochs", epochs) or epochs
                sub_prog = 0.15 + (curr_epoch / float(tot_epochs)) * 0.35
                status_path = os.path.join(self.job_dir, "status.json")
                status_data = {}
                if os.path.exists(status_path):
                    with open(status_path, "r", encoding="utf-8") as f:
                        status_data = json.load(f)
                status_data.update(
                    {
                        "status": "running",
                        "stage": f"Training detector (Epoch {curr_epoch}/{tot_epochs})",
                        "progress": round(sub_prog, 3),
                        "updated_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    }
                )
                status_data.pop("error", None)
                tmp_path = f"{status_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(status_data, f, indent=2)
                os.replace(tmp_path, status_path)
            except Exception:
                pass

        model.add_callback("on_train_epoch_end", on_epoch_end)

        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            project=self.job_dir,
            name="yolo_obb_train",
            workers=int(kwargs.pop("workers", 0)),
            **kwargs,
        )

        save_dir = str(getattr(results, "save_dir", os.path.join(self.job_dir, "yolo_obb_train")))
        best_weights = os.path.join(save_dir, "weights", "best.pt")
        if not os.path.exists(best_weights):
            raise FileNotFoundError("YOLO training completed without producing best.pt weights.")

        exported_path = export_model_to_onnx(YOLO(best_weights))
        if not os.path.exists(exported_path):
            raise FileNotFoundError("YOLO ONNX export did not produce an artifact.")
        shutil.copy2(exported_path, detector_weights)

        raw_metrics = dict(getattr(results, "results_dict", {}) or {})
        map50 = raw_metrics.get("metrics/mAP50(B)", raw_metrics.get("mAP50"))
        map5095 = raw_metrics.get("metrics/mAP50-95(B)", raw_metrics.get("mAP50-95"))
        return {
            "detector_weights": detector_weights,
            "detector_weights_pt": best_weights,
            "mAP50": float(map50) if map50 is not None else None,
            "mAP50-95": float(map5095) if map5095 is not None else None,
            "detector_metrics": raw_metrics,
            "epochs": epochs,
        }
