import os
import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import cv2

try:
    import dlib
except ImportError:
    dlib = None

try:
    from backend.ort_inference import OrtYoloDetector
except ImportError:
    try:
        from ort_inference import OrtYoloDetector
    except ImportError:
        OrtYoloDetector = None

try:
    from backend.domain.models import Manifest
    from backend.geometry import canonicalize_obb_for_crop
    from backend.inference.detectors import (
        GenericOrtYoloOBBDetector,
        UltralyticsYoloOBBDetector,
    )
except ImportError:
    from domain.models import Manifest
    from geometry import canonicalize_obb_for_crop
    from inference.detectors import GenericOrtYoloOBBDetector, UltralyticsYoloOBBDetector

logger = logging.getLogger(__name__)


def _shape_points_in_schema_order(shape, expected_names: List[str]) -> np.ndarray:
    """Convert dlib's lexically indexed parts back to manifest schema order."""
    raw_points = np.array(
        [[shape.part(index).x, shape.part(index).y] for index in range(shape.num_parts)],
        dtype=np.float32,
    )
    if not expected_names:
        return raw_points

    dlib_names = sorted(expected_names)
    points_by_name = {
        name: raw_points[index] for index, name in enumerate(dlib_names)
    }
    return np.array([points_by_name[name] for name in expected_names], dtype=np.float32)


class GenericPipelineEngine:
    """Generic inference engine for YOLO-OBB detection and dlib ML-Morph landmark prediction."""

    def __init__(self, model_repo=None, models_dir: str = "models"):
        self.model_repo = model_repo
        self.models_dir = models_dir
        self._predictor_cache: Dict[str, Any] = {}
        self._detector_cache: Dict[str, Any] = {}

    def crop_oriented_bbox(
        self, image: np.ndarray, obb: List[float], padding: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop and rectify a rotated oriented bounding box (OBB) into an axis-aligned crop.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W).
            obb: [cx, cy, w, h, angle_deg]
            padding: Padding factor around width and height.

        Returns:
            Tuple of (crop_image, affine_transform_matrix_M)
            M is a (2, 3) matrix mapping original image points to crop points.
        """
        cx, cy, w, h, angle_deg = obb
        padded_w = float(w) * (1.0 + padding)
        padded_h = float(h) * (1.0 + padding)

        # OpenCV rotation matrix around center (cx, cy)
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

        # Shift translation component so (cx, cy) maps to center of crop canvas (padded_w/2, padded_h/2)
        M[0, 2] += padded_w / 2.0 - cx
        M[1, 2] += padded_h / 2.0 - cy

        out_w = max(1, int(round(padded_w)))
        out_h = max(1, int(round(padded_h)))

        crop = cv2.warpAffine(image, M, (out_w, out_h))
        return crop, M

    def _get_detector(
        self,
        artifact_path: str,
        class_names: List[str],
        legacy_toepad: bool = False,
        iou_threshold: float = 0.45,
    ):
        cache_key = (artifact_path, tuple(class_names), legacy_toepad, iou_threshold)
        if cache_key in self._detector_cache:
            return self._detector_cache[cache_key]

        if artifact_path.endswith(".onnx"):
            if legacy_toepad and OrtYoloDetector is not None:
                detector = OrtYoloDetector(artifact_path)
            else:
                detector = GenericOrtYoloOBBDetector(
                    artifact_path, class_names, iou_threshold=iou_threshold
                )
        elif artifact_path.endswith(".pt"):
            detector = UltralyticsYoloOBBDetector(artifact_path, class_names)
        else:
            raise ValueError(f"Unsupported detector artifact: {artifact_path}")

        self._detector_cache[cache_key] = detector
        return detector

    def _get_shape_predictor(self, predictor_path: str):
        if predictor_path in self._predictor_cache:
            return self._predictor_cache[predictor_path]

        if dlib is None:
            raise RuntimeError("dlib is required for landmark inference but is not installed.")
        try:
            predictor = dlib.shape_predictor(predictor_path)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load dlib shape predictor from '{predictor_path}': {error}"
            ) from error
        self._predictor_cache[predictor_path] = predictor
        return predictor

    def _resolve_artifact_path(
        self, artifact: str, bundle_dir: Optional[str] = None
    ) -> Optional[str]:
        """Resolve an artifact from a model bundle, models directory, or app root."""
        if not artifact:
            return None

        candidates = []
        if os.path.isabs(artifact):
            candidates.append(artifact)
        else:
            if bundle_dir:
                candidates.append(os.path.join(bundle_dir, artifact))
            candidates.extend(
                [
                    os.path.join(self.models_dir, artifact),
                    os.path.join(os.path.dirname(self.models_dir), artifact),
                ]
            )

        for candidate in candidates:
            normalized = os.path.abspath(candidate)
            if os.path.exists(normalized):
                return normalized
        return None

    def predict(
        self, image: np.ndarray, manifest: Manifest, bundle_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute prediction pipeline for a given image and Manifest.

        Args:
            image: Image numpy array.
            manifest: Manifest instance.
            bundle_dir: Directory containing model files (defaults to self.models_dir).

        Returns:
            List of object prediction dicts:
            [
                {
                    "class_name": str,
                    "obb": [cx, cy, w, h, angle_deg],
                    "landmarks": [{"x": float, "y": float}, ...]
                },
                ...
            ]
        """
        if bundle_dir is None:
            bundle_dir = self.models_dir

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Inference input image is empty or invalid.")
        if image.ndim not in (2, 3):
            raise ValueError(f"Inference input must be a 2D or 3D image, got shape {image.shape}.")
        if not manifest.classes:
            raise ValueError(f"Model '{manifest.name}' does not define any classes.")

        h_img, w_img = image.shape[:2]
        targets = []

        # Run detector if available
        detector_artifact = manifest.detector.artifact if manifest.detector else ""
        class_names = [class_config.name for class_config in manifest.classes]

        if detector_artifact:
            artifact_path = self._resolve_artifact_path(detector_artifact, bundle_dir)
            if not artifact_path:
                raise FileNotFoundError(
                    f"Detector artifact '{detector_artifact}' for model '{manifest.name}' was not found."
                )
            detector = self._get_detector(
                artifact_path,
                class_names,
                legacy_toepad=manifest.id == "lizard-toepad-v1",
                iou_threshold=manifest.detector.iou,
            )
            conf = manifest.detector.confidence if manifest.detector else 0.25
            detections = detector.detect(image, conf_threshold=conf)
            for cls_cfg in manifest.classes:
                for detection in detections.get(cls_cfg.name, []):
                    corners = detection.get("corners")
                    if (
                        manifest.id == "lizard-toepad-v1"
                        and cls_cfg.name.startswith("up_")
                        and corners is not None
                    ):
                        corners = np.asarray(corners, dtype=np.float32).copy()
                        corners[:, 1] = h_img - 1 - corners[:, 1]
                    obb = detection.get("obb")
                    if obb is None and corners is not None and len(corners) == 4:
                        (cx, cy), (w, h), angle = cv2.minAreaRect(
                            np.asarray(corners, dtype=np.float32)
                        )
                        obb = [float(cx), float(cy), float(w), float(h), float(angle)]
                    if obb is None:
                        continue
                    targets.append(
                        {
                            "class_name": cls_cfg.name,
                            "obb": obb,
                            "corners": corners,
                            "confidence": detection.get("conf"),
                            "class_config": cls_cfg,
                        }
                    )

        if not detector_artifact:
            for cls_cfg in manifest.classes:
                obb = [w_img / 2.0, h_img / 2.0, float(w_img), float(h_img), 0.0]
                targets.append(
                    {
                        "class_name": cls_cfg.name,
                        "obb": obb,
                        "class_config": cls_cfg,
                    }
                )

        results = []
        for target in targets:
            class_name = target["class_name"]
            obb = canonicalize_obb_for_crop(target["obb"])
            cls_cfg = target["class_config"]
            padding = cls_cfg.crop_padding if cls_cfg.crop_padding is not None else 0.2

            crop, M = self.crop_oriented_bbox(image, obb, padding=padding)
            landmarks = []

            if cls_cfg.predictor:
                pred_path = self._resolve_artifact_path(cls_cfg.predictor, bundle_dir)
                if not pred_path:
                    raise FileNotFoundError(
                        f"Landmark predictor '{cls_cfg.predictor}' for class '{class_name}' was not found."
                    )
                sp = self._get_shape_predictor(pred_path)
                try:
                    if len(crop.shape) == 3 and crop.shape[2] == 3:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    else:
                        crop_rgb = crop
                    rect = dlib.rectangle(
                        0, 0, max(0, crop.shape[1] - 1), max(0, crop.shape[0] - 1)
                    )
                    shape = sp(crop_rgb, rect)
                    schema = manifest.landmark_schemas.get(cls_cfg.landmark_schema or "")
                    expected_names = schema.points if schema else []
                    if expected_names and shape.num_parts != len(expected_names):
                        raise ValueError(
                            f"Predictor for class '{class_name}' returned {shape.num_parts} landmarks; "
                            f"the manifest declares {len(expected_names)}."
                        )
                    pts_crop = _shape_points_in_schema_order(shape, expected_names)

                    if len(pts_crop) > 0:
                        M_rev = cv2.invertAffineTransform(M)
                        pts_h = np.hstack(
                            [pts_crop, np.ones((len(pts_crop), 1), dtype=np.float32)]
                        )
                        pts_orig = (M_rev @ pts_h.T).T
                        landmarks = [
                            {
                                "name": expected_names[index] if index < len(expected_names) else str(index),
                                "x": float(point[0]),
                                "y": float(point[1]),
                            }
                            for index, point in enumerate(pts_orig)
                        ]
                except Exception as e:
                    raise RuntimeError(
                        f"Landmark inference failed for class '{class_name}': {e}"
                    ) from e

            results.append(
                {
                    "class_name": class_name,
                    "obb": obb,
                    "corners": target.get("corners"),
                    "confidence": target.get("confidence"),
                    "landmarks": landmarks,
                }
            )

        return results
