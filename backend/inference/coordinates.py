"""Albert's original ViT crop contract, exported to CPU ONNX for the pilot.

The original notebook normalizes BGR, not RGB. Do not silently change this.
This adapter supports bottom digit AABB crops, not flipped upper digits.
"""
import cv2
import numpy as np
import onnxruntime as ort


class CoordinatePredictor:
    def __init__(self, path, count):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        metadata = self.session.get_modelmeta().custom_metadata_map
        if (len(inputs) != 1 or inputs[0].shape != [1, 3, 224, 224]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1
                or outputs[0].shape != [1, count * 2]
                or metadata.get("lizard_contract") != "albert-vit-aabb-224-bgr-v1"):
            raise ValueError("Incompatible coordinate ONNX input/output contract")
        self.input_name = inputs[0].name
        self.count = count
        self._run(np.zeros((1, 3, 224, 224), dtype=np.float32))

    def _run(self, tensor):
        result = self.session.run(None, {self.input_name: tensor})[0].reshape(self.count, 2)
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
            raise ValueError("Coordinate predictor must return finite normalized coordinates")
        return result

    def predict(self, image, corners):
        height, width = image.shape[:2]
        corners = np.asarray(corners)
        x1 = max(0, int(corners[:, 0].min()))
        y1 = max(0, int(corners[:, 1].min()))
        x2 = min(width, int(corners[:, 0].max()))
        y2 = min(height, int(corners[:, 1].max()))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Empty landmark crop")
        crop = image[y1:y2, x1:x2]
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        scale = 224.0 / max(crop.shape[:2])
        new_w, new_h = round(crop.shape[1] * scale), round(crop.shape[0] * scale)
        resized = cv2.resize(crop, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_LINEAR)
        left, top = (224 - resized.shape[1]) // 2, (224 - resized.shape[0]) // 2
        canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        normalized = (canvas.astype(np.float32) / 255.0 - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        points = self._run(tensor) * 224.0
        return (points - np.array([left, top])) / scale + np.array([x1, y1])
