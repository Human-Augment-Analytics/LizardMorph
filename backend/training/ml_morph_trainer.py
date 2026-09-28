import os
import json
import random
import re
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional

try:
    from backend.datasets.canonical import CanonicalDataset
    from backend.geometry import canonicalize_obb_for_crop
except ImportError:
    from datasets.canonical import CanonicalDataset
    from geometry import canonicalize_obb_for_crop

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
        class_name: Optional[str] = None,
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
        class_slug = (
            re.sub(r"[^A-Za-z0-9_.-]+", "_", class_name or "default").strip("._")
            or "default"
        )
        lm_dir = os.path.join(self.job_dir, "landmark_dataset", class_slug)
        crops_dir = os.path.join(lm_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        dataset_elem = ET.Element("dataset")
        name_elem = ET.SubElement(dataset_elem, "name")
        name_elem.text = "ML-Morph Training Dataset"
        comment_elem = ET.SubElement(dataset_elem, "comment")
        comment_elem.text = "Auto-generated landmark crops with box jitter"

        images_elem = ET.SubElement(dataset_elem, "images")
        source_groups = {}

        random_generator = np.random.default_rng(42)
        for image_index, cimg in enumerate(canonical_dataset.images):
            # Every referenced source image must be present. Silent placeholder images
            # produce apparently successful but unusable predictors.
            img_w = cimg.width if cimg.width > 0 else 100
            img_h = cimg.height if cimg.height > 0 else 100

            actual_img_path = resolve_image_path(cimg.file_path)
            if not actual_img_path or not os.path.exists(actual_img_path):
                raise FileNotFoundError(
                    f"Image '{cimg.file_path}' referenced by the annotations was not provided."
                )
            img = cv2.imread(actual_img_path)
            if img is None:
                raise ValueError(f"Unable to decode training image '{cimg.file_path}'.")
            img_h, img_w = img.shape[:2]

            for object_index, obj in enumerate(cimg.objects):
                if class_name is not None and obj.class_name != class_name:
                    continue
                if not obj.landmarks or len(obj.obb) < 5:
                    continue

                cx, cy, w, h, angle_deg = canonicalize_obb_for_crop(obj.obb)

                # Apply box jitter if requested
                if box_jitter > 0:
                    dx = random_generator.uniform(-box_jitter, box_jitter) * w
                    dy = random_generator.uniform(-box_jitter, box_jitter) * h
                    dw = random_generator.uniform(-box_jitter, box_jitter) * w
                    dh = random_generator.uniform(-box_jitter, box_jitter) * h
                    j_cx = cx + dx
                    j_cy = cy + dy
                    j_w = max(1.0, w + dw)
                    j_h = max(1.0, h + dh)
                else:
                    j_cx, j_cy, j_w, j_h = cx, cy, w, h

                def make_transform(center_x, center_y, box_width, box_height):
                    padded_width = float(box_width) * (1.0 + jitter_padding)
                    padded_height = float(box_height) * (1.0 + jitter_padding)
                    transform = cv2.getRotationMatrix2D(
                        (center_x, center_y), angle_deg, 1.0
                    )
                    transform[0, 2] += padded_width / 2.0 - center_x
                    transform[1, 2] += padded_height / 2.0 - center_y
                    return (
                        transform,
                        max(1, int(round(padded_width))),
                        max(1, int(round(padded_height))),
                    )

                M, out_w, out_h = make_transform(j_cx, j_cy, j_w, j_h)

                def transform_landmarks(transform):
                    return [
                        (
                            landmark,
                            transform[0, 0] * landmark.x
                            + transform[0, 1] * landmark.y
                            + transform[0, 2],
                            transform[1, 0] * landmark.x
                            + transform[1, 1] * landmark.y
                            + transform[1, 2],
                        )
                        for landmark in obj.landmarks
                    ]

                transformed_landmarks = transform_landmarks(M)
                if any(
                    px < 0 or py < 0 or px > out_w - 1 or py > out_h - 1
                    for _, px, py in transformed_landmarks
                ):
                    # A jittered crop must never discard its ground-truth points.
                    M, out_w, out_h = make_transform(cx, cy, w, h)
                    transformed_landmarks = transform_landmarks(M)

                crop = cv2.warpAffine(
                    img,
                    M,
                    (out_w, out_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

                crop_filename = f"crop_{image_index + 1}_{object_index + 1}.jpg"
                crop_path = os.path.join(crops_dir, crop_filename)
                if not cv2.imwrite(crop_path, crop):
                    raise OSError(f"Unable to write landmark crop '{crop_path}'.")

                # Relative path for dlib XML
                rel_crop_path = os.path.join("crops", crop_filename)
                source_groups[rel_crop_path] = os.path.realpath(actual_img_path)

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

                for lm, px, py in transformed_landmarks:
                    ET.SubElement(
                        box_node,
                        "part",
                        attrib={
                            "name": str(lm.name),
                            "x": str(min(out_w - 1, max(0, int(round(px))))),
                            "y": str(min(out_h - 1, max(0, int(round(py))))),
                        },
                    )

        xml_path = os.path.join(lm_dir, "train_landmarks.xml")
        tree = ET.ElementTree(dataset_elem)
        
        # Write formatted XML
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        if not images_elem.findall("image"):
            raise ValueError(f"No landmark annotations found for class '{class_name or 'default'}'.")

        with open(xml_path + ".groups.json", "w", encoding="utf-8") as handle:
            json.dump(source_groups, handle, indent=2)

        return xml_path

    @staticmethod
    def _split_dataset(xml_path: str, test_split: float, seed: int = 42):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        images_node = root.find("images")
        if images_node is None:
            raise ValueError("Generated dlib dataset is missing its images element.")
        images = list(images_node.findall("image"))
        # Generated crops from one source image are correlated observations.
        # Split source groups, never individual crops. Legacy XML without a
        # sidecar is grouped by image file (including repeated references).
        groups_path = xml_path + ".groups.json"
        if os.path.exists(groups_path):
            with open(groups_path, encoding="utf-8") as handle:
                source_groups = json.load(handle)
            group_keys = [source_groups[image.get("file")] for image in images]
        else:
            group_keys = [image.get("file") for image in images]
        groups = list(dict.fromkeys(group_keys))
        if len(groups) < 2 or test_split <= 0:
            return xml_path, None

        random.Random(seed).shuffle(groups)
        test_count = min(len(groups) - 1, max(1, int(round(len(groups) * test_split))))
        test_groups = set(groups[:test_count])

        def write_subset(path: str, include_test: bool):
            subset_root = ET.Element("dataset")
            ET.SubElement(subset_root, "name").text = "ML-Morph Dataset"
            subset_images = ET.SubElement(subset_root, "images")
            for index, image in enumerate(images):
                if (group_keys[index] in test_groups) == include_test:
                    subset_images.append(ET.fromstring(ET.tostring(image)))
            ET.ElementTree(subset_root).write(path, encoding="utf-8", xml_declaration=True)

        base_dir = os.path.dirname(xml_path)
        train_path = os.path.join(base_dir, "train_split.xml")
        test_path = os.path.join(base_dir, "test_split.xml")
        write_subset(train_path, False)
        write_subset(test_path, True)
        return train_path, test_path

    def train(
        self,
        xml_path: str,
        output_model_name: str = "shape_predictor.dat",
        custom_options: Optional[Dict[str, Any]] = None,
        test_split: float = 0.2,
        mock: bool = False,
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
        if os.path.basename(output_model_name) != output_model_name:
            raise ValueError("Landmark model filename must not contain a directory path.")
        output_model_path = os.path.join(models_dir, output_model_name)

        if mock:
            with open(output_model_path, "wb") as f:
                f.write(b"EXPLICIT_TEST_MOCK_DLIB")
            return {
                "landmark_model": output_model_path,
                "model_path": output_model_path,
                "status": "mock",
                "test_error": 0.0,
            }

        if dlib is None:
            raise RuntimeError("dlib is required for landmark training but is not installed.")

        try:
            train_xml_path, test_xml_path = self._split_dataset(xml_path, test_split)
            try:
                train_tree = ET.parse(train_xml_path)
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

            dlib.train_shape_predictor(train_xml_path, output_model_path, options)

            # Loading the artifact catches truncated or invalid predictors immediately.
            dlib.shape_predictor(output_model_path)
            test_error = None
            if test_xml_path:
                test_error = float(dlib.test_shape_predictor(test_xml_path, output_model_path))

            return {
                "landmark_model": output_model_path,
                "model_path": output_model_path,
                "status": "success",
                "test_error": test_error,
            }
        except Exception:
            if os.path.exists(output_model_path):
                os.remove(output_model_path)
            raise
