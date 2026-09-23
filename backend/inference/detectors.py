import math
from typing import Dict, List

import cv2
import numpy as np


def _rotated_iou(first: dict, second: dict) -> float:
    first_obb = first["obb"]
    second_obb = second["obb"]
    first_rect = (
        (float(first_obb[0]), float(first_obb[1])),
        (max(0.0, float(first_obb[2])), max(0.0, float(first_obb[3]))),
        float(first_obb[4]),
    )
    second_rect = (
        (float(second_obb[0]), float(second_obb[1])),
        (max(0.0, float(second_obb[2])), max(0.0, float(second_obb[3]))),
        float(second_obb[4]),
    )
    _, intersection_points = cv2.rotatedRectangleIntersection(first_rect, second_rect)
    intersection = (
        float(abs(cv2.contourArea(intersection_points)))
        if intersection_points is not None
        else 0.0
    )
    first_area = first_rect[1][0] * first_rect[1][1]
    second_area = second_rect[1][0] * second_rect[1][1]
    denominator = first_area + second_area - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _class_aware_nms(detections: List[dict], iou_threshold: float) -> List[dict]:
    kept = []
    for detection in sorted(detections, key=lambda item: item["conf"], reverse=True):
        if all(
            detection["class_id"] != existing["class_id"]
            or _rotated_iou(detection, existing) <= iou_threshold
            for existing in kept
        ):
            kept.append(detection)
    return kept


class GenericOrtYoloOBBDetector:
    """Single-pass Ultralytics YOLO-OBB ONNX inference with dynamic class names."""

    def __init__(self, model_path: str, class_names: List[str], iou_threshold: float = 0.45):
        import onnxruntime as ort

        if not class_names:
            raise ValueError("Detector requires at least one class name.")
        self.class_names = class_names
        self.iou_threshold = iou_threshold
        providers = [
            provider
            for provider in (
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            )
            if provider in ort.get_available_providers()
        ] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.output_name = self.session.get_outputs()[0].name
        shape = input_info.shape
        self.input_height = int(shape[-2]) if isinstance(shape[-2], int) else 1024
        self.input_width = int(shape[-1]) if isinstance(shape[-1], int) else 1024

    def _preprocess(self, image: np.ndarray):
        height, width = image.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height))
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = tensor.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor[np.newaxis, ...], scale, pad_x, pad_y

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, list]:
        if image is None or image.size == 0:
            raise ValueError("Detector input image is empty.")
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        data = output[0] if output.ndim == 3 else output
        expected_channels = len(self.class_names) + 5
        if data.shape[0] != expected_channels and data.shape[1] == expected_channels:
            data = data.T
        if data.shape[0] != expected_channels:
            raise ValueError(
                f"Detector output has {data.shape[0]} channels; expected {expected_channels} "
                f"for {len(self.class_names)} classes."
            )

        class_scores = data[4 : 4 + len(self.class_names)]
        class_ids = np.argmax(class_scores, axis=0)
        confidences = np.max(class_scores, axis=0)
        candidates = []
        for index in np.where(confidences >= conf_threshold)[0]:
            cx = (float(data[0, index]) - pad_x) / scale
            cy = (float(data[1, index]) - pad_y) / scale
            width = float(data[2, index]) / scale
            height = float(data[3, index]) / scale
            if not all(np.isfinite(value) for value in (cx, cy, width, height)):
                continue
            if width <= 0 or height <= 0:
                continue
            angle_degrees = math.degrees(float(data[-1, index]))
            corners = cv2.boxPoints(((cx, cy), (width, height), angle_degrees))
            candidates.append(
                {
                    "class_id": int(class_ids[index]),
                    "conf": float(confidences[index]),
                    "corners": corners,
                    "obb": [cx, cy, width, height, angle_degrees],
                }
            )

        detections = {class_name: [] for class_name in self.class_names}
        for candidate in _class_aware_nms(candidates, self.iou_threshold):
            class_id = candidate["class_id"]
            if 0 <= class_id < len(self.class_names):
                detections[self.class_names[class_id]].append(candidate)
        return detections


class UltralyticsYoloOBBDetector:
    """Compatibility loader for existing .pt custom model runs."""

    def __init__(self, model_path: str, class_names: List[str]):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.class_names = class_names

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, list]:
        result = self.model.predict(image, conf=conf_threshold, verbose=False)[0]
        detections = {class_name: [] for class_name in self.class_names}
        if result.obb is None:
            return detections

        boxes = result.obb.xywhr.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()
        class_ids = result.obb.cls.cpu().numpy().astype(int)
        result_names = result.names
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            class_name = (
                result_names.get(class_id, str(class_id))
                if isinstance(result_names, dict)
                else result_names[class_id]
            )
            if class_name not in detections and 0 <= class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            if class_name not in detections:
                continue
            cx, cy, width, height, angle_radians = [float(value) for value in box]
            angle_degrees = math.degrees(angle_radians)
            corners = cv2.boxPoints(((cx, cy), (width, height), angle_degrees))
            detections[class_name].append(
                {
                    "class_id": class_id,
                    "conf": float(confidence),
                    "corners": corners,
                    "obb": [cx, cy, width, height, angle_degrees],
                }
            )
        return detections
