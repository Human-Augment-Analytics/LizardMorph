import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional

from backend.datasets.canonical import CanonicalDataset

try:
    import dlib
except ImportError:
    dlib = None

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


class MLMorphTrainer:
    """Trainer adapter for cropping landmark regions with box jitter and training dlib shape predictors."""

    def __init__(self, job_dir: str):
        self.job_dir = job_dir

    def prepare_landmark_dataset(
        self,
        canonical_dataset: CanonicalDataset,
        jitter_padding: float = 0.2,
        box_jitter: float = 0.05,
    ) -> str:
        """
        Crops images with box jitter padding around OBB and generates dlib training XML.

        Args:
            canonical_dataset: Source dataset.
            jitter_padding: Padding factor added around OBB.
            box_jitter: Fractional jitter applied to box position and size to tolerate detector error.

        Returns:
            Absolute path to generated dlib XML annotation file.
        """
        lm_dir = os.path.join(self.job_dir, "landmark_dataset")
        crops_dir = os.path.join(lm_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        dataset_elem = ET.Element("dataset")
        name_elem = ET.SubElement(dataset_elem, "name")
        name_elem.text = "ML-Morph Training Dataset"
        comment_elem = ET.SubElement(dataset_elem, "comment")
        comment_elem.text = "Auto-generated landmark crops with box jitter"

        images_elem = ET.SubElement(dataset_elem, "images")

        for cimg in canonical_dataset.images:
            # Load or create base image
            img_w = cimg.width if cimg.width > 0 else 100
            img_h = cimg.height if cimg.height > 0 else 100

            actual_img_path = resolve_image_path(cimg.file_path)
            if actual_img_path and os.path.exists(actual_img_path):
                img = cv2.imread(actual_img_path)
                if img is None:
                    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 128
                else:
                    img_h, img_w = img.shape[:2]
            else:
                img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 128

            for obj in cimg.objects:
                if not obj.landmarks or len(obj.obb) < 5:
                    continue

                cx, cy, w, h, angle_deg = obj.obb[:5]

                # Apply box jitter if requested
                if box_jitter > 0:
                    dx = np.random.uniform(-box_jitter, box_jitter) * w
                    dy = np.random.uniform(-box_jitter, box_jitter) * h
                    dw = np.random.uniform(-box_jitter, box_jitter) * w
                    dh = np.random.uniform(-box_jitter, box_jitter) * h
                    j_cx = cx + dx
                    j_cy = cy + dy
                    j_w = max(1.0, w + dw)
                    j_h = max(1.0, h + dh)
                else:
                    j_cx, j_cy, j_w, j_h = cx, cy, w, h

                padded_w = float(j_w) * (1.0 + jitter_padding)
                padded_h = float(j_h) * (1.0 + jitter_padding)

                # Compute affine transformation matrix for crop
                M = cv2.getRotationMatrix2D((j_cx, j_cy), angle_deg, 1.0)
                M[0, 2] += padded_w / 2.0 - j_cx
                M[1, 2] += padded_h / 2.0 - j_cy

                out_w = max(1, int(round(padded_w)))
                out_h = max(1, int(round(padded_h)))

                crop = cv2.warpAffine(
                    img,
                    M,
                    (out_w, out_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

                crop_filename = f"crop_{cimg.image_id}_{obj.object_id}.jpg"
                crop_path = os.path.join(crops_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                # Relative path for dlib XML
                rel_crop_path = os.path.join("crops", crop_filename)

                img_node = ET.SubElement(
                    images_elem,
                    "image",
                    attrib={
                        "file": rel_crop_path,
                        "width": str(out_w),
                        "height": str(out_h),
                    },
                )
                box_node = ET.SubElement(
                    img_node,
                    "box",
                    attrib={
                        "top": "0",
                        "left": "0",
                        "width": str(out_w),
                        "height": str(out_h),
                        "label": obj.class_name or "object",
                    },
                )

                for lm in obj.landmarks:
                    # Transform landmark point into crop space
                    px = M[0, 0] * lm.x + M[0, 1] * lm.y + M[0, 2]
                    py = M[1, 0] * lm.x + M[1, 1] * lm.y + M[1, 2]
                    ET.SubElement(
                        box_node,
                        "part",
                        attrib={
                            "name": str(lm.name),
                            "x": str(int(round(px))),
                            "y": str(int(round(py))),
                        },
                    )

        xml_path = os.path.join(lm_dir, "train_landmarks.xml")
        tree = ET.ElementTree(dataset_elem)
        
        # Write formatted XML
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        return xml_path

    def train(
        self,
        xml_path: str,
        output_model_name: str = "shape_predictor.dat",
        custom_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs dlib.train_shape_predictor on generated dataset.

        Args:
            xml_path: Path to dlib XML dataset file.
            output_model_name: Filename of output shape predictor .dat file.
            custom_options: Dict of shape predictor training hyperparameters.

        Returns:
            Dict containing model path and training metrics.
        """
        opts = custom_options or {}
        models_dir = os.path.join(self.job_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        output_model_path = os.path.join(models_dir, output_model_name)

        if dlib is not None:
            try:
                try:
                    train_tree = ET.parse(xml_path)
                    num_images = len(train_tree.getroot().findall(".//image"))
                except Exception:
                    num_images = 1

                options = dlib.shape_predictor_training_options()
                options.num_threads = min(4, os.cpu_count() or 1)
                if num_images > 0:
                    options.num_threads = min(options.num_threads, num_images)

                options.nu = float(opts.get("nu", 0.1))
                options.tree_depth = int(opts.get("tree_depth", 4))
                options.cascade_depth = int(opts.get("cascade_depth", 15))
                options.oversampling_amount = int(opts.get("oversampling_amount", 5))
                options.feature_pool_size = int(opts.get("feature_pool_size", 400))
                options.num_test_splits = int(opts.get("num_test_splits", 20))
                options.be_verbose = False

                dlib.train_shape_predictor(xml_path, output_model_path, options)

                return {
                    "landmark_model": output_model_path,
                    "model_path": output_model_path,
                    "status": "success",
                }
            except Exception as e:
                # Fallback mock file on dlib training error
                with open(output_model_path, "wb") as f:
                    f.write(b"MOCK_DLIB_SHAPE_PREDICTOR_DATA")

                return {
                    "landmark_model": output_model_path,
                    "model_path": output_model_path,
                    "status": "mock_fallback",
                    "warning": str(e),
                }
        else:
            # Fallback mock when dlib is not installed
            with open(output_model_path, "wb") as f:
                f.write(b"MOCK_DLIB_SHAPE_PREDICTOR_DATA")

            return {
                "landmark_model": output_model_path,
                "model_path": output_model_path,
                "status": "mock",
            }
