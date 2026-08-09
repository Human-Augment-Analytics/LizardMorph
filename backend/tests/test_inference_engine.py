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
from backend.inference.engine import GenericPipelineEngine


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


def test_api_predict_route(client):
    # Test built-in legacy model resolution via view_type
    res = client.post(
        "/api/predict",
        data={"view_type": "dorsal"},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["model_version_id"] == "lizard-dorsal-v1"
    assert "predictions" in data
