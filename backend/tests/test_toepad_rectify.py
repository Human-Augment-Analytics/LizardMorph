"""Regression coverage for pretrained toepad crop geometry and both entry points."""
from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

import utils
import ort_inference


class RecordingPredictor:
    def __init__(self):
        self.calls = []

    def __call__(self, image, rect):
        self.calls.append((image.copy(), rect))
        return SimpleNamespace(parts=lambda: [SimpleNamespace(x=256, y=256)])


def test_rectify_backprojects_canvas_center_and_keeps_training_color_order():
    image = np.full((200, 300, 3), [20, 70, 150], dtype=np.uint8)
    corners = np.array([[80, 30], [140, 50], [120, 110], [60, 90]], np.float32)
    predictor = RecordingPredictor()
    result = utils._predict_toepad_crop(predictor, image, corners, 'ml_morph_best.dat')
    canvas, rect = predictor.calls[0]
    assert canvas.shape == (512, 512, 3)
    assert (rect.left(), rect.top(), rect.right(), rect.bottom()) == (0, 0, 512, 512)
    np.testing.assert_array_equal(canvas[256, 256], [20, 70, 150])
    # Pixel endpoints are w-1/h-1 while the training resize uses w/h.
    np.testing.assert_allclose(result[0], [100, 70], atol=1)


def test_legacy_predictor_retains_padded_rgb_crop():
    image = np.full((200, 300, 3), [20, 70, 150], dtype=np.uint8)
    corners = np.array([[80, 30], [140, 30], [140, 110], [80, 110]], np.float32)
    predictor = RecordingPredictor()
    result = utils._predict_toepad_crop(predictor, image, corners, 'toe_predictor_obb.dat')
    crop, rect = predictor.calls[0]
    assert crop.shape == (129, 97, 3)
    np.testing.assert_array_equal(crop[0, 0], [150, 70, 20])
    assert (rect.right(), rect.bottom()) == (97, 129)
    np.testing.assert_array_equal(result, [[318, 262]])


def test_rectify_rejects_degenerate_box():
    with pytest.raises(ValueError, match='Degenerate'):
        utils._predict_toepad_crop(RecordingPredictor(), np.zeros((10, 10, 3), np.uint8),
                                   np.zeros((4, 2)), 'ml_morph_best.dat')


def test_server_and_client_upper_digit_coordinate_parity(monkeypatch, tmp_path):
    image = np.full((200, 300, 3), 127, dtype=np.uint8)
    path = str(tmp_path / 'image.png')
    cv2.imwrite(path, image)
    corners = np.array([[80, 30], [140, 50], [120, 110], [60, 90]], np.float32)
    detector = object.__new__(ort_inference.OrtYoloDetector)
    monkeypatch.setattr(detector, 'detect', lambda *a, **k: {
        'up_toe': [{'conf': .95, 'corners': corners.copy()}]})
    original = corners.copy()
    original[:, 1] = 199 - original[:, 1]
    ann = {'bounding_boxes': [{'label': 'up_toe', 'confidence': .95,
           'obb_corners': [{'x': float(x), 'y': float(y)} for x, y in original]}]}
    kwargs = dict(image_path=path, toe_predictor_path='ml_morph_best.dat',
                  cached_dlib_predictors={'toe': RecordingPredictor()})
    server = tmp_path / 'server.xml'
    client = tmp_path / 'client.xml'
    assert utils.predictions_to_xml_single_with_yolo(
        **kwargs, output=str(server), cached_yolo_model=detector) == 1
    assert utils.predictions_to_xml_single_from_client_annotations(
        **kwargs, output=str(client), client_ann=ann) == 1
    assert ET.parse(server).find('.//part').attrib == ET.parse(client).find('.//part').attrib
    point = ET.parse(client).find('.//part')
    assert abs(int(point.get('x')) - 100) <= 1
    assert abs(int(point.get('y')) - 129) <= 1


def test_builtin_engine_uses_same_pretrained_geometry(monkeypatch):
    from backend.inference.engine import GenericPipelineEngine
    from backend.domain.models import Manifest, DetectorConfig, ClassConfig, LandmarkSchemaConfig

    engine = GenericPipelineEngine()
    corners = np.array([[80, 30], [140, 50], [120, 110], [60, 90]], np.float32)
    image = np.full((200, 300, 3), 127, np.uint8)
    predictor = RecordingPredictor()
    detector = SimpleNamespace(detect=lambda *a, **k: {
        'up_toe': [{'corners': corners.copy(), 'conf': .95}]})
    monkeypatch.setattr(engine, '_resolve_artifact_path', lambda path, bundle: path)
    monkeypatch.setattr(engine, '_get_detector', lambda *a, **k: detector)
    monkeypatch.setattr(engine, '_get_shape_predictor', lambda path: predictor)
    manifest = Manifest(
        schema_version=1, id='lizard-toepad-v1', name='Toepad', description='Test',
        detector=DetectorConfig(artifact='detector.onnx'),
        classes=[ClassConfig(id=0, name='up_toe', predictor='ml_morph_best.dat', landmark_schema='toe')],
        landmark_schemas={'toe': LandmarkSchemaConfig(points=['0'])})
    result = engine.predict(image, manifest)
    json.dumps(result)  # Public engine output must be safe for Flask's JSON response.
    point = result[0]['landmarks'][0]
    assert abs(point['x'] - 100) <= 1
    assert abs(point['y'] - 129) <= 1
    assert predictor.calls[0][0].shape == (512, 512, 3)


def test_transfer_manifest_preserves_protocol_after_rename_and_roundtrip(monkeypatch):
    from backend.domain.models import Manifest, DetectorConfig, ClassConfig, LandmarkSchemaConfig
    from backend.inference.engine import GenericPipelineEngine

    manifest = Manifest(
        schema_version=1, id='anolis-transfer-test', name='Transfer', description='Test',
        detector=DetectorConfig(artifact='renamed-detector.onnx', inference_protocol='toepad-dual-pass'),
        classes=[ClassConfig(id=1, name='up_toe', predictor='renamed-landmarks.dat',
                             landmark_schema='toe', preprocessing='toepad-rectify-512')],
        landmark_schemas={'toe': LandmarkSchemaConfig(points=['0'])})
    restored = Manifest.from_dict(manifest.to_dict())
    assert restored.detector.inference_protocol == 'toepad-dual-pass'
    assert restored.classes[0].preprocessing == 'toepad-rectify-512'
    corners = np.array([[80, 30], [140, 50], [120, 110], [60, 90]], np.float32)
    detector = SimpleNamespace(detect=lambda *a, **k: {
        'up_toe': [{'corners': corners, 'conf': .95}]})
    engine = GenericPipelineEngine()
    predictor = RecordingPredictor()
    monkeypatch.setattr(engine, '_resolve_artifact_path', lambda path, bundle: path)
    def get_detector(*args, **kwargs):
        assert kwargs['legacy_toepad'] is True
        return detector
    monkeypatch.setattr(engine, '_get_detector', get_detector)
    monkeypatch.setattr(engine, '_get_shape_predictor', lambda path: predictor)
    result = engine.predict(np.full((200, 300, 3), 127, np.uint8), restored)
    json.dumps(result)  # Public engine output must be safe for Flask's JSON response.
    point = result[0]['landmarks'][0]
    assert abs(point['x'] - 100) <= 1
    assert abs(point['y'] - 129) <= 1
    assert predictor.calls[0][0].shape == (512, 512, 3)


def test_unknown_model_protocols_are_rejected():
    from backend.domain.models import DetectorConfig, ClassConfig
    with pytest.raises(ValueError, match='detector protocol'):
        DetectorConfig(artifact='model.onnx', inference_protocol='typo')
    with pytest.raises(ValueError, match='landmark preprocessing'):
        ClassConfig(id=0, name='toe', preprocessing='typo')
