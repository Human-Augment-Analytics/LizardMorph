"""Behavioral regressions for research training and validation integrity."""
import sys
import types
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from backend.datasets.canonical import CanonicalDataset, CanonicalImage, CanonicalObject, LandmarkPoint
from backend.training.ml_morph_trainer import MLMorphTrainer
from backend.training.yolo_trainer import YoloOBBTrainer


def test_landmark_validation_keeps_source_images_together(tmp_path):
    images = []
    for index in range(3):
        path = tmp_path / f"source_{index}.png"
        cv2.imwrite(str(path), np.zeros((100, 100, 3), dtype=np.uint8))
        objects = [CanonicalObject(str(j), "digit", [50, 50, 40, 40, 0],
                   [LandmarkPoint("0", 50, 50)]) for j in range(4)]
        images.append(CanonicalImage(str(index), str(path), 100, 100, objects))
    trainer = MLMorphTrainer(str(tmp_path / "job"))
    xml = trainer.prepare_landmark_dataset(CanonicalDataset(images))
    train, test = trainer._split_dataset(xml, 0.5)
    # Generated crop filenames identify their original image, independently of
    # the split implementation. Every source must occur on only one side.
    def sources(path):
        return {node.get("file").split("_")[1]
                for node in ET.parse(path).findall(".//image")}
    assert sources(train).isdisjoint(sources(test))
    assert sources(train) | sources(test) == {"1", "2", "3"}
    assert len(ET.parse(train).findall(".//image")) + len(ET.parse(test).findall(".//image")) == 12


def test_yolo_default_model_can_train_and_export(tmp_path, monkeypatch):
    captured = []
    class FakeYOLO:
        def __init__(self, path):
            captured.append(str(path))
        def add_callback(self, *args):
            pass
        def train(self, **kwargs):
            weights = tmp_path / "yolo_obb_train" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"checkpoint")
            return types.SimpleNamespace(save_dir=str(weights.parent), results_dict={})
        def export(self, **kwargs):
            path = tmp_path / "export.onnx"
            path.write_bytes(b"export")
            return str(path)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    result = YoloOBBTrainer(str(tmp_path)).train("data.yaml", epochs=1)
    assert captured[0].endswith("yolov8n-obb.pt")
    assert result["detector_weights_pt"].endswith("best.pt")


def test_single_source_has_no_landmark_holdout(tmp_path):
    path = tmp_path / "source.png"
    cv2.imwrite(str(path), np.zeros((100, 100, 3), dtype=np.uint8))
    objects = [CanonicalObject(str(j), "digit", [50, 50, 40, 40, 0],
               [LandmarkPoint("0", 50, 50)]) for j in range(4)]
    dataset = CanonicalDataset([CanonicalImage("one", str(path), 100, 100, objects)])
    trainer = MLMorphTrainer(str(tmp_path / "job"))
    xml = trainer.prepare_landmark_dataset(dataset)
    assert trainer._split_dataset(xml, 0.5) == (xml, None)
    result = trainer.train(xml, custom_options={"tree_depth": 2, "cascade_depth": 2})
    assert result["status"] == "success"
    assert result["test_error"] is None
