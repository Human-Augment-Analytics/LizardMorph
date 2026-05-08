"""
ORT-based YOLO OBB inference module.

Direct ONNX Runtime inference for YOLO OBB toepad models. Auto-detects
the model schema from class names embedded in the ONNX metadata:

- H7 6-class (up_finger, up_toe, bot_finger, bot_toe, ruler, id):
  normal pass produces all 6 classes; flipped pass acts as fallback for
  missed up_* detections.

- H11 4-class (finger, toe, ruler, id, trained bottom-only):
  normal pass → bot_*; flipped pass → up_* (Y-flipped).
"""

import ast
import math
import numpy as np
import cv2
import onnxruntime as ort


# Model constants
INPUT_SIZE = 1280
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
YOLO_MAX_DIM = 4096

SCHEMA_H7_6CLASS = "h7_6class"
SCHEMA_H11_4CLASS = "h11_4class"

def _get_execution_providers():
    """Return available execution providers, preferring GPU."""
    available = ort.get_available_providers()
    preferred = [
        "CoreMLExecutionProvider",
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    providers = [p for p in preferred if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


# Default fallback for legacy 6-class H7 models without proper metadata
DEFAULT_H7_CLASS_NAMES = {
    0: "up_finger",
    1: "up_toe",
    2: "bot_finger",
    3: "bot_toe",
    4: "ruler",
    5: "id",
}


def _read_class_names_from_onnx(session: ort.InferenceSession) -> dict[int, str] | None:
    """Read class names dict from ONNX model metadata.

    Ultralytics writes a `names` key like "{0: 'finger', 1: 'toe', ...}".
    """
    try:
        meta = session.get_modelmeta().custom_metadata_map
        names_str = meta.get("names")
        if not names_str:
            return None
        parsed = ast.literal_eval(names_str)
        if isinstance(parsed, dict):
            return {int(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    return None


def _infer_schema(class_names: dict[int, str]) -> str:
    """Pick the merge strategy based on the number of classes.

    - 4 classes → H11 schema (dual-pass routes by pass: normal=bot, flipped=up)
    - 6 classes (or anything else) → H7 schema (class name tells up vs bot)
    """
    return SCHEMA_H11_4CLASS if len(class_names) == 4 else SCHEMA_H7_6CLASS


class OrtYoloDetector:
    """YOLO OBB detector using ONNX Runtime directly."""

    def __init__(self, model_path: str):
        """Load ONNX model with best available execution provider.

        Args:
            model_path: Path to .onnx model file.
        """
        providers = _get_execution_providers()
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.class_names = (
            _read_class_names_from_onnx(self.session) or DEFAULT_H7_CLASS_NAMES
        )
        self.num_classes = len(self.class_names)
        self.schema = _infer_schema(self.class_names)
        print(
            f"[OrtYoloDetector] Loaded {model_path} "
            f"(schema={self.schema}, classes={self.num_classes}: {list(self.class_names.values())})"
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(img_bgr: np.ndarray):
        """Resize with aspect ratio preservation, pad to INPUT_SIZE x INPUT_SIZE.

        Returns:
            tensor: float32 NCHW array [1, 3, INPUT_SIZE, INPUT_SIZE]
            scale: resize scale factor
            x_pad: horizontal padding (left)
            y_pad: vertical padding (top)
        """
        h, w = img_bgr.shape[:2]
        scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
        new_w = round(w * scale)
        new_h = round(h * scale)

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas with value 114
        canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        x_pad = (INPUT_SIZE - new_w) // 2
        y_pad = (INPUT_SIZE - new_h) // 2
        canvas[y_pad : y_pad + new_h, x_pad : x_pad + new_w] = resized

        # BGR -> RGB, HWC -> CHW, normalize to [0,1], add batch dim
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = chw[np.newaxis, ...]  # [1, 3, H, W]

        return tensor, scale, x_pad, y_pad

    @staticmethod
    def _flip_tensor_vertically(tensor: np.ndarray) -> np.ndarray:
        """Flip a [1, 3, H, W] NCHW tensor vertically (reverse rows)."""
        return tensor[:, :, ::-1, :].copy()

    # ------------------------------------------------------------------
    # Detection parsing
    # ------------------------------------------------------------------

    def _parse_detections(self, output: np.ndarray, conf_threshold: float):
        """Parse raw model output [1, 4+nc+1, num_anchors] into detection list.

        Channels: 0=cx, 1=cy, 2=w, 3=h, 4..(4+nc-1)=class_confs, last=angle

        Returns list of dicts with keys: x, y, w, h, angle, conf, class_id
        """
        data = output[0]  # [C, num_anchors]
        nc = self.num_classes
        class_confs = data[4 : 4 + nc]
        max_class_ids = np.argmax(class_confs, axis=0)
        max_confs = np.max(class_confs, axis=0)
        angle_row = data[4 + nc]

        mask = max_confs > conf_threshold
        indices = np.where(mask)[0]

        boxes = []
        for i in indices:
            boxes.append({
                "x": float(data[0, i]),
                "y": float(data[1, i]),
                "w": float(data[2, i]),
                "h": float(data[3, i]),
                "angle": float(angle_row[i]),
                "conf": float(max_confs[i]),
                "class_id": int(max_class_ids[i]),
            })
        return boxes

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    @staticmethod
    def _iou_aabb(a, b):
        """AABB IoU approximation for NMS (ignoring rotation)."""
        a_min_x = a["x"] - a["w"] / 2
        a_max_x = a["x"] + a["w"] / 2
        a_min_y = a["y"] - a["h"] / 2
        a_max_y = a["y"] + a["h"] / 2

        b_min_x = b["x"] - b["w"] / 2
        b_max_x = b["x"] + b["w"] / 2
        b_min_y = b["y"] - b["h"] / 2
        b_max_y = b["y"] + b["h"] / 2

        inter_x1 = max(a_min_x, b_min_x)
        inter_y1 = max(a_min_y, b_min_y)
        inter_x2 = min(a_max_x, b_max_x)
        inter_y2 = min(a_max_y, b_max_y)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        a_area = a["w"] * a["h"]
        b_area = b["w"] * b["h"]

        return inter_area / (a_area + b_area - inter_area)

    @classmethod
    def _nms_and_top_one(cls, boxes):
        """Apply NMS per class then keep only top-1 per class."""
        boxes.sort(key=lambda b: b["conf"], reverse=True)

        after_nms = []
        for b in boxes:
            overlap = False
            for k in after_nms:
                if k["class_id"] == b["class_id"] and cls._iou_aabb(b, k) > IOU_THRESHOLD:
                    overlap = True
                    break
            if not overlap:
                after_nms.append(b)

        # Keep top-1 per class
        seen = set()
        result = []
        for b in after_nms:
            if b["class_id"] not in seen:
                seen.add(b["class_id"])
                result.append(b)
        return result

    # ------------------------------------------------------------------
    # OBB corners
    # ------------------------------------------------------------------

    @staticmethod
    def _get_obb_corners(cx, cy, w, h, angle):
        """Compute 4 corners of a rotated rectangle.

        Returns np.ndarray of shape (4, 2).
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        half_w = w / 2
        half_h = h / 2

        # Corner offsets relative to center
        offsets = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]

        corners = np.array(
            [
                [cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a]
                for dx, dy in offsets
            ],
            dtype=np.float32,
        )
        return corners

    # ------------------------------------------------------------------
    # Main detect method
    # ------------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray, conf_threshold: float = CONF_THRESHOLD):
        """Run dual-pass OBB detection on a BGR image.

        Args:
            img_bgr: Input image in BGR format (OpenCV convention).
            conf_threshold: Minimum confidence threshold.

        Returns:
            dict with keys: bot_finger, bot_toe, up_finger, up_toe, scale, id.
            Each value is a list of detection dicts with 'conf' and 'corners' (np.ndarray(4,2)).
            Scale detections also have 'obb_wh'.
            Coordinates are in the original image space.
            up_finger/up_toe corners from the flipped pass are in FLIPPED image coordinates
            (consistent with existing utils.py expectations).
        """
        h_img, w_img = img_bgr.shape[:2]

        # Downsample if image is too large
        max_dim = max(h_img, w_img)
        if max_dim > YOLO_MAX_DIM:
            ds_scale = YOLO_MAX_DIM / max_dim
            img_bgr = cv2.resize(
                img_bgr,
                (int(w_img * ds_scale), int(h_img * ds_scale)),
                interpolation=cv2.INTER_AREA,
            )
            print(
                f"[OrtYoloDetector] Downsampled: {w_img}x{h_img} -> "
                f"{img_bgr.shape[1]}x{img_bgr.shape[0]} (scale={ds_scale:.3f})"
            )
        else:
            ds_scale = 1.0
        inv_scale = 1.0 / ds_scale

        # Preprocess
        tensor, scale, x_pad, y_pad = self._preprocess(img_bgr)

        # --- Pass 1: Normal inference ---
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        normal_boxes = self._nms_and_top_one(self._parse_detections(output, conf_threshold))

        # --- Pass 2: Flipped inference ---
        flipped_tensor = self._flip_tensor_vertically(tensor)
        output_flipped = self.session.run([self.output_name], {self.input_name: flipped_tensor})[0]
        flipped_boxes = self._nms_and_top_one(self._parse_detections(output_flipped, conf_threshold))

        # --- Build detections dict ---
        detections = {
            "bot_finger": [],
            "bot_toe": [],
            "up_finger": [],
            "up_toe": [],
            "scale": [],
            "id": [],
        }

        def _model_to_original_corners(box):
            """Convert model-space box to original image corners."""
            corners_model = self._get_obb_corners(
                box["x"], box["y"], box["w"], box["h"], box["angle"]
            )
            # Model space -> downsampled image space: undo padding and scale
            corners_ds = np.zeros_like(corners_model)
            corners_ds[:, 0] = (corners_model[:, 0] - x_pad) / scale
            corners_ds[:, 1] = (corners_model[:, 1] - y_pad) / scale
            # Downsampled image space -> original image space
            corners_orig = corners_ds * inv_scale
            return corners_orig

        def _model_to_flipped_original_corners(box):
            """Convert model-space box (from flipped pass) to flipped original image corners.

            Returns corners in the FLIPPED image coordinate space (not un-flipped).
            This matches what utils.py expects: it processes up_finger/up_toe
            on the flipped image directly.
            """
            corners_model = self._get_obb_corners(
                box["x"], box["y"], box["w"], box["h"], box["angle"]
            )
            # Model space -> downsampled flipped image space
            corners_ds = np.zeros_like(corners_model)
            corners_ds[:, 0] = (corners_model[:, 0] - x_pad) / scale
            corners_ds[:, 1] = (corners_model[:, 1] - y_pad) / scale
            # Downsampled -> original (still in flipped coordinate space)
            corners_orig = corners_ds * inv_scale
            return corners_orig

        def _flip_y(corners):
            out = np.copy(corners)
            out[:, 1] = h_img - 1 - out[:, 1]
            return out

        if self.schema == SCHEMA_H7_6CLASS:
            # Normal pass produces all 6 classes; flipped pass acts as fallback for missed up_*
            for box in normal_boxes:
                cls_name = self.class_names.get(box["class_id"], "unknown")
                corners = _model_to_original_corners(box)
                det = {"conf": box["conf"], "corners": corners}

                if cls_name in ("ruler", "scale"):
                    det["obb_wh"] = (box["w"] / scale * inv_scale, box["h"] / scale * inv_scale)
                    detections["scale"].append(det)
                elif cls_name == "bot_finger":
                    detections["bot_finger"].append(det)
                elif cls_name == "bot_toe":
                    detections["bot_toe"].append(det)
                elif cls_name == "up_finger":
                    detections["up_finger"].append({"conf": box["conf"], "corners": _flip_y(corners)})
                elif cls_name == "up_toe":
                    detections["up_toe"].append({"conf": box["conf"], "corners": _flip_y(corners)})
                elif cls_name == "id":
                    detections["id"].append(det)

            for box in flipped_boxes:
                cls_name = self.class_names.get(box["class_id"], "unknown")
                if cls_name == "bot_finger":
                    detections["up_finger"].append(
                        {"conf": box["conf"], "corners": _model_to_flipped_original_corners(box)}
                    )
                elif cls_name == "bot_toe":
                    detections["up_toe"].append(
                        {"conf": box["conf"], "corners": _model_to_flipped_original_corners(box)}
                    )

        else:  # SCHEMA_H11_4CLASS — pass routes class to bot/up
            for box in normal_boxes:
                cls_name = self.class_names.get(box["class_id"], "unknown")
                corners = _model_to_original_corners(box)
                det = {"conf": box["conf"], "corners": corners}

                if cls_name in ("ruler", "scale"):
                    det["obb_wh"] = (box["w"] / scale * inv_scale, box["h"] / scale * inv_scale)
                    detections["scale"].append(det)
                elif cls_name == "finger":
                    detections["bot_finger"].append(det)
                elif cls_name == "toe":
                    detections["bot_toe"].append(det)
                elif cls_name == "id":
                    detections["id"].append(det)

            for box in flipped_boxes:
                cls_name = self.class_names.get(box["class_id"], "unknown")
                # ruler/id deduped via normal pass; flipped pass only contributes upper toepads
                if cls_name == "finger":
                    detections["up_finger"].append(
                        {"conf": box["conf"], "corners": _model_to_flipped_original_corners(box)}
                    )
                elif cls_name == "toe":
                    detections["up_toe"].append(
                        {"conf": box["conf"], "corners": _model_to_flipped_original_corners(box)}
                    )

        # Keep only best (highest conf) per category
        for category in detections:
            if len(detections[category]) > 1:
                detections[category].sort(key=lambda d: d["conf"], reverse=True)
                detections[category] = [detections[category][0]]

        detected_classes = [k for k, v in detections.items() if v]
        print(f"[OrtYoloDetector] Detected: {', '.join(detected_classes)}")

        return detections
