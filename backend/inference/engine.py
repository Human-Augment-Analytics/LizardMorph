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

from backend.domain.models import Manifest

logger = logging.getLogger(__name__)


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

    def _get_detector(self, artifact_path: str):
        if artifact_path in self._detector_cache:
            return self._detector_cache[artifact_path]

        if OrtYoloDetector is not None and artifact_path.endswith(".onnx"):
            try:
                detector = OrtYoloDetector(artifact_path)
                self._detector_cache[artifact_path] = detector
                return detector
            except Exception as e:
                logger.warning(f"Failed to load OrtYoloDetector from {artifact_path}: {e}")
                return None
        return None

    def _get_shape_predictor(self, predictor_path: str):
        if predictor_path in self._predictor_cache:
            return self._predictor_cache[predictor_path]

        if dlib is not None:
            try:
                predictor = dlib.shape_predictor(predictor_path)
                self._predictor_cache[predictor_path] = predictor
                return predictor
            except Exception as e:
                logger.warning(f"Failed to load dlib shape predictor from {predictor_path}: {e}")
                return None
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

        h_img, w_img = image.shape[:2]
        targets = []

        # Run detector if available
        detector_artifact = manifest.detector.artifact if manifest.detector else ""
        detector_run_successful = False

        if detector_artifact:
            artifact_path = os.path.join(bundle_dir, detector_artifact)
            if not os.path.exists(artifact_path):
                artifact_path = os.path.join(self.models_dir, detector_artifact)

            if os.path.exists(artifact_path):
                detector = self._get_detector(artifact_path)
                if detector is not None:
                    try:
                        conf = manifest.detector.confidence if manifest.detector else 0.25
                        detections = detector.detect(image, conf_threshold=conf)
                        detector_run_successful = True

                        for cls_cfg in manifest.classes:
                            cls_dets = detections.get(cls_cfg.name, [])
                            if cls_dets:
                                for det in cls_dets:
                                    corners = det.get("corners")
                                    if corners is not None and len(corners) == 4:
                                        (cx, cy), (w, h), angle = cv2.minAreaRect(corners)
                                        obb = [float(cx), float(cy), float(w), float(h), float(angle)]
                                    else:
                                        obb = [w_img / 2.0, h_img / 2.0, float(w_img), float(h_img), 0.0]
                                    targets.append(
                                        {
                                            "class_name": cls_cfg.name,
                                            "obb": obb,
                                            "class_config": cls_cfg,
                                        }
                                    )
                            else:
                                obb = [w_img / 2.0, h_img / 2.0, float(w_img), float(h_img), 0.0]
                                targets.append(
                                    {
                                        "class_name": cls_cfg.name,
                                        "obb": obb,
                                        "class_config": cls_cfg,
                                    }
                                )
                    except Exception as e:
                        logger.warning(f"Error running detector: {e}")
                        detector_run_successful = False

        if not detector_run_successful:
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
            obb = target["obb"]
            cls_cfg = target["class_config"]
            padding = cls_cfg.crop_padding if cls_cfg.crop_padding is not None else 0.2

            crop, M = self.crop_oriented_bbox(image, obb, padding=padding)
            landmarks = []

            if cls_cfg.predictor:
                pred_path = os.path.join(bundle_dir, cls_cfg.predictor)
                if not os.path.exists(pred_path):
                    pred_path = os.path.join(self.models_dir, cls_cfg.predictor)

                if os.path.exists(pred_path):
                    sp = self._get_shape_predictor(pred_path)
                    if sp is not None:
                        try:
                            if len(crop.shape) == 3 and crop.shape[2] == 3:
                                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            else:
                                crop_rgb = crop
                            rect = dlib.rectangle(0, 0, crop.shape[1], crop.shape[0])
                            shape = sp(crop_rgb, rect)
                            pts_crop = np.array(
                                [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)],
                                dtype=np.float32,
                            )

                            if len(pts_crop) > 0:
                                M_rev = cv2.invertAffineTransform(M)
                                pts_h = np.hstack(
                                    [pts_crop, np.ones((len(pts_crop), 1), dtype=np.float32)]
                                )
                                pts_orig = (M_rev @ pts_h.T).T
                                landmarks = [
                                    {"x": float(pt[0]), "y": float(pt[1])} for pt in pts_orig
                                ]
                        except Exception as e:
                            logger.warning(f"Error predicting shape landmarks: {e}")

            results.append(
                {
                    "class_name": class_name,
                    "obb": obb,
                    "landmarks": landmarks,
                }
            )

        return results
