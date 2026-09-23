import io

import numpy as np
import cv2
import pytest
from backend.domain.models import (
    Manifest,
    DetectorConfig,
    ClassConfig,
    LandmarkSchemaConfig,
)
from backend.storage.db import DatabaseManager
from backend.storage.repository import ModelRegistryRepository
from backend.geometry import canonicalize_obb_for_crop
from backend.inference.engine import (
    GenericPipelineEngine,
    _shape_points_in_schema_order,
)


class _FakePart:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeShape:
    def __init__(self, points):
        self._points = [_FakePart(x, y) for x, y in points]
        self.num_parts = len(points)

    def part(self, index):
        return self._points[index]


def test_canonicalize_obb_swaps_equivalent_tall_representation():
    canonical = canonicalize_obb_for_crop(
        [344.6, 289.1, 249.6, 552.4, 90.2]
    )

    np.testing.assert_allclose(
        canonical,
        [344.6, 289.1, 552.4, 249.6, 0.2],
        atol=1e-6,
    )


def test_shape_points_are_restored_from_dlib_lexical_order():
    schema_names = [str(index) for index in range(12)]
    dlib_names = sorted(schema_names)
    point_for_name = {
        name: (float(index), float(index + 100))
        for index, name in enumerate(schema_names)
    }
    shape = _FakeShape([point_for_name[name] for name in dlib_names])

    ordered = _shape_points_in_schema_order(shape, schema_names)

    np.testing.assert_allclose(
        ordered,
        np.array([point_for_name[name] for name in schema_names], dtype=np.float32),
    )


def test_generic_predictions_to_annotations_accepts_numpy_corners():
    import app as app_mod

    coords, bounding_boxes = app_mod.generic_predictions_to_annotations(
        [
            {
                "class_name": "object",
                "obb": [20.0, 15.0, 10.0, 8.0, 12.0],
                "corners": np.array(
                    [[14.0, 10.0], [24.0, 10.0], [24.0, 18.0], [14.0, 18.0]],
                    dtype=np.float32,
                ),
                "confidence": 0.9,
                "landmarks": [{"x": 16.5, "y": 12.5}],
            }
        ]
    )

    assert coords == [{"x": 16.5, "y": 12.5, "id": 0}]
    assert bounding_boxes[0]["obb_corners"] == [
        {"x": 14.0, "y": 10.0},
        {"x": 24.0, "y": 10.0},
        {"x": 24.0, "y": 18.0},
        {"x": 14.0, "y": 18.0},
    ]


def test_crop_oriented_bbox_axis_aligned():
    db_mgr = DatabaseManager(":memory:")
    db_mgr.init_db()
    repo = ModelRegistryRepository(db_mgr)
    engine = GenericPipelineEngine(repo)

    # 100x100 RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255  # white square at center

    obb = [50.0, 50.0, 20.0, 20.0, 0.0]  # cx, cy, w, h, angle
    crop, M = engine.crop_oriented_bbox(img, obb, padding=0.2)

    # padded_w = 20 * 1.2 = 24, padded_h = 20 * 1.2 = 24
    assert crop.shape == (24, 24, 3)
    assert M.shape == (2, 3)

    # Test coordinate mapping: center (50, 50) in original image should map to (12, 12) in crop
    pt_orig = np.array([50.0, 50.0, 1.0], dtype=np.float32)
    pt_crop = M @ pt_orig
    np.testing.assert_allclose(pt_crop, [12.0, 12.0], atol=1e-5)


def test_reverse_coordinate_transform():
    db_mgr = DatabaseManager(":memory:")
    db_mgr.init_db()
    repo = ModelRegistryRepository(db_mgr)
    engine = GenericPipelineEngine(repo)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    obb = [100.0, 80.0, 40.0, 30.0, 30.0]  # rotated obb
    crop, M = engine.crop_oriented_bbox(img, obb, padding=0.2)

    M_rev = cv2.invertAffineTransform(M)

    # Center of crop
    padded_w = 40.0 * 1.2
    padded_h = 30.0 * 1.2
    crop_center = np.array([padded_w / 2.0, padded_h / 2.0, 1.0], dtype=np.float32)

    orig_pt = M_rev @ crop_center
    np.testing.assert_allclose(orig_pt, [100.0, 80.0], atol=1e-4)


def test_generic_pipeline_predict_fallback_bbox(tmp_path):
    db_mgr = DatabaseManager(":memory:")
    db_mgr.init_db()
    repo = ModelRegistryRepository(db_mgr, models_dir=str(tmp_path))
    engine = GenericPipelineEngine(repo, models_dir=str(tmp_path))

    manifest = Manifest(
        schema_version=1,
        id="test-model-v1",
        name="Test Model",
        description="Testing generic engine",
        detector=DetectorConfig(artifact="", geometry="obb"),
        classes=[
            ClassConfig(
                id=0,
                name="dorsal",
                landmark_schema="dorsal",
                predictor=None,
                crop_padding=0.2,
            )
        ],
        landmark_schemas={
            "dorsal": LandmarkSchemaConfig(points=["p1", "p2"])
        },
    )

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    results = engine.predict(img, manifest, bundle_dir=str(tmp_path))

    assert isinstance(results, list)
    assert len(results) == 1
    res = results[0]
    assert res["class_name"] == "dorsal"
    assert res["obb"] == [50.0, 50.0, 100.0, 100.0, 0.0]
    assert isinstance(res["landmarks"], list)


def test_api_predict_route(client, monkeypatch):
    import app as app_mod

    monkeypatch.setattr(
        app_mod.generic_pipeline_engine,
        "predict",
        lambda image, manifest, bundle_dir=None: [],
    )
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    # Test built-in legacy model resolution via view_type
    res = client.post(
        "/api/predict",
        data={
            "view_type": "dorsal",
            "image": (io.BytesIO(encoded.tobytes()), "test.png"),
        },
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["model_version_id"] == "lizard-dorsal-v1"
    assert "predictions" in data


def test_data_upload_uses_selected_generic_model(client, monkeypatch, tmp_path):
    import app as app_mod

    manifest = Manifest(
        schema_version=1,
        id="custom-v1",
        name="Custom Model",
        description="Test custom upload model",
        detector=DetectorConfig(artifact="", geometry="obb"),
        classes=[
            ClassConfig(
                id=0,
                name="object",
                landmark_schema="default",
                predictor=None,
            )
        ],
        landmark_schemas={"default": LandmarkSchemaConfig(points=["p1", "p2"])},
    )

    monkeypatch.setattr(
        app_mod,
        "resolve_inference_model",
        lambda model_id: (manifest, str(tmp_path)) if model_id == "custom-v1" else (None, None),
    )
    monkeypatch.setattr(
        app_mod.generic_pipeline_engine,
        "predict",
        lambda image, selected_manifest, bundle_dir=None: [
            {
                "class_name": "object",
                "obb": [20.0, 15.0, 10.0, 8.0, 0.0],
                "landmarks": [{"x": 12.5, "y": 9.5}, {"x": 18.0, "y": 14.0}],
            }
        ],
    )
    monkeypatch.setattr(app_mod.xray_preprocessing, "process_single_image", lambda *args: None)
    monkeypatch.setattr(
        app_mod.visual_individual_performance, "invert_single_image", lambda *args: None
    )

    image = np.zeros((30, 40, 3), dtype=np.uint8)
    encoded, image_bytes = cv2.imencode(".png", image)
    assert encoded

    response = client.post(
        "/data",
        data={
            "view_type": "custom",
            "model_id": "custom-v1",
            "image": (io.BytesIO(image_bytes.tobytes()), "specimen.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = response.get_json()[0]
    assert result["name"] == "specimen.jpg"
    assert result["coords"] == [
        {"id": 0, "x": 12.5, "y": 9.5},
        {"id": 1, "x": 18.0, "y": 14.0},
    ]
    assert result["bounding_boxes"][0]["landmark_count"] == 2
    assert result["bounding_boxes"][0]["landmark_start_index"] == 0


def test_data_upload_rejects_unknown_selected_model(client, monkeypatch):
    import app as app_mod

    monkeypatch.setattr(app_mod, "resolve_inference_model", lambda _model_id: (None, None))

    response = client.post(
        "/data",
        data={"view_type": "custom", "model_id": "missing-model"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 404
    assert response.get_json()["error"] == "Model 'missing-model' not found"
