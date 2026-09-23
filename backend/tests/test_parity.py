import importlib
import io
import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def stub_inference(monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(
        app_module.generic_pipeline_engine,
        "predict",
        lambda image, manifest, bundle_dir=None: [
            {
                "class_name": manifest.classes[0].name,
                "obb": [25.0, 25.0, 50.0, 50.0, 0.0],
                "landmarks": [],
            }
        ],
    )


def post_image(client, **fields):
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    return client.post(
        "/api/predict",
        data={**fields, "image": (io.BytesIO(encoded.tobytes()), "test.png")},
        content_type="multipart/form-data",
    )


def test_predict_parity_dorsal(client):
    """Verify that view_type='dorsal' yields identical model resolution as model_version_id='lizard-dorsal-v1'."""
    res_legacy = post_image(client, view_type="dorsal")
    assert res_legacy.status_code == 200
    data_legacy = res_legacy.get_json()

    res_modern = post_image(client, model_version_id="lizard-dorsal-v1")
    assert res_modern.status_code == 200
    data_modern = res_modern.get_json()

    assert data_legacy["success"] is True
    assert data_modern["success"] is True
    assert data_legacy["model_version_id"] == "lizard-dorsal-v1"
    assert data_modern["model_version_id"] == "lizard-dorsal-v1"
    assert data_legacy["predictions"] == data_modern["predictions"]


def test_predict_parity_lateral(client):
    """Verify that view_type='lateral' yields identical model resolution as model_version_id='lizard-lateral-v1'."""
    res_legacy = post_image(client, view_type="lateral")
    assert res_legacy.status_code == 200
    data_legacy = res_legacy.get_json()

    res_modern = post_image(client, model_version_id="lizard-lateral-v1")
    assert res_modern.status_code == 200
    data_modern = res_modern.get_json()

    assert data_legacy["success"] is True
    assert data_modern["success"] is True
    assert data_legacy["model_version_id"] == "lizard-lateral-v1"
    assert data_modern["model_version_id"] == "lizard-lateral-v1"
    assert data_legacy["predictions"] == data_modern["predictions"]


def test_predict_parity_toepad(client):
    """Verify that view_type='toepad' or 'toepads' yields identical model resolution as model_version_id='lizard-toepad-v1'."""
    res_toepad = post_image(client, view_type="toepad")
    res_toepads = post_image(client, view_type="toepads")
    res_modern = post_image(client, model_version_id="lizard-toepad-v1")

    assert res_toepad.status_code == 200
    assert res_toepads.status_code == 200
    assert res_modern.status_code == 200

    d_toepad = res_toepad.get_json()
    d_toepads = res_toepads.get_json()
    d_modern = res_modern.get_json()

    assert d_toepad["success"] is True
    assert d_toepads["success"] is True
    assert d_modern["success"] is True

    assert d_toepad["model_version_id"] == "lizard-toepad-v1"
    assert d_toepads["model_version_id"] == "lizard-toepad-v1"
    assert d_modern["model_version_id"] == "lizard-toepad-v1"

    assert d_toepad["predictions"] == d_modern["predictions"]
    assert d_toepads["predictions"] == d_modern["predictions"]


def test_predict_default_view_type_fallback(client):
    """Verify that omitting both view_type and model_version_id defaults to dorsal."""
    res = post_image(client)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["model_version_id"] == "lizard-dorsal-v1"


def test_predict_model_id_alias(client):
    """Verify that using model_id field works as an alias for model_version_id."""
    res = post_image(client, model_id="lizard-dorsal-v1")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["model_version_id"] == "lizard-dorsal-v1"


def test_predict_multipart_form_data(client):
    """Verify multipart form-data requests work for both view_type and model_version_id."""
    res_form_legacy = post_image(client, view_type="lateral")
    assert res_form_legacy.status_code == 200
    d_legacy = res_form_legacy.get_json()

    res_form_modern = post_image(client, model_version_id="lizard-lateral-v1")
    assert res_form_modern.status_code == 200
    d_modern = res_form_modern.get_json()

    assert d_legacy["success"] is True
    assert d_modern["success"] is True
    assert d_legacy["model_version_id"] == "lizard-lateral-v1"
    assert d_modern["model_version_id"] == "lizard-lateral-v1"
    assert d_legacy["predictions"] == d_modern["predictions"]


def test_predict_requires_uploaded_image(client, tmp_path):
    """Verify uploads work and arbitrary server-side image paths are rejected."""
    # 1. Create a dummy image
    img_array = np.zeros((50, 50, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".png", img_array)
    img_bytes = img_encoded.tobytes()

    # Post via multipart file upload
    res_upload = client.post(
        "/api/predict",
        data={
            "view_type": "dorsal",
            "image": (io.BytesIO(img_bytes), "test.png"),
        },
        content_type="multipart/form-data",
    )
    assert res_upload.status_code == 200
    d_upload = res_upload.get_json()
    assert d_upload["success"] is True
    assert d_upload["model_version_id"] == "lizard-dorsal-v1"
    assert isinstance(d_upload["predictions"], list)

    # 2. Save image to disk and post via image_path
    img_file = tmp_path / "specimen.png"
    cv2.imwrite(str(img_file), img_array)

    res_path = client.post(
        "/api/predict",
        json={
            "model_version_id": "lizard-dorsal-v1",
            "image_path": str(img_file),
        },
    )
    assert res_path.status_code == 400
    d_path = res_path.get_json()
    assert d_path["success"] is False
    assert "image upload" in d_path["error"]


def test_predict_invalid_model_version_id(client):
    """Verify that an invalid model_version_id returns 404 error response."""
    res = client.post("/api/predict", json={"model_version_id": "nonexistent-model-id"})
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False
    assert "not found" in data["error"]
