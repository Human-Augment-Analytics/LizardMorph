import io
import cv2
import numpy as np
import pytest


def test_predict_parity_dorsal(client):
    """Verify that view_type='dorsal' yields identical model resolution as model_version_id='lizard-dorsal-v1'."""
    res_legacy = client.post("/api/predict", json={"view_type": "dorsal"})
    assert res_legacy.status_code == 200
    data_legacy = res_legacy.get_json()

    res_modern = client.post("/api/predict", json={"model_version_id": "lizard-dorsal-v1"})
    assert res_modern.status_code == 200
    data_modern = res_modern.get_json()

    assert data_legacy["success"] is True
    assert data_modern["success"] is True
    assert data_legacy["model_version_id"] == "lizard-dorsal-v1"
    assert data_modern["model_version_id"] == "lizard-dorsal-v1"
    assert data_legacy["predictions"] == data_modern["predictions"]


def test_predict_parity_lateral(client):
    """Verify that view_type='lateral' yields identical model resolution as model_version_id='lizard-lateral-v1'."""
    res_legacy = client.post("/api/predict", json={"view_type": "lateral"})
    assert res_legacy.status_code == 200
    data_legacy = res_legacy.get_json()

    res_modern = client.post("/api/predict", json={"model_version_id": "lizard-lateral-v1"})
    assert res_modern.status_code == 200
    data_modern = res_modern.get_json()

    assert data_legacy["success"] is True
    assert data_modern["success"] is True
    assert data_legacy["model_version_id"] == "lizard-lateral-v1"
    assert data_modern["model_version_id"] == "lizard-lateral-v1"
    assert data_legacy["predictions"] == data_modern["predictions"]


def test_predict_parity_toepad(client):
    """Verify that view_type='toepad' or 'toepads' yields identical model resolution as model_version_id='lizard-toepad-v1'."""
    res_toepad = client.post("/api/predict", json={"view_type": "toepad"})
    res_toepads = client.post("/api/predict", json={"view_type": "toepads"})
    res_modern = client.post("/api/predict", json={"model_version_id": "lizard-toepad-v1"})

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
    res = client.post("/api/predict", json={})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["model_version_id"] == "lizard-dorsal-v1"


def test_predict_model_id_alias(client):
    """Verify that using model_id field works as an alias for model_version_id."""
    res = client.post("/api/predict", json={"model_id": "lizard-dorsal-v1"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["model_version_id"] == "lizard-dorsal-v1"


def test_predict_multipart_form_data(client):
    """Verify multipart form-data requests work for both view_type and model_version_id."""
    res_form_legacy = client.post(
        "/api/predict",
        data={"view_type": "lateral"},
        content_type="multipart/form-data",
    )
    assert res_form_legacy.status_code == 200
    d_legacy = res_form_legacy.get_json()

    res_form_modern = client.post(
        "/api/predict",
        data={"model_version_id": "lizard-lateral-v1"},
        content_type="multipart/form-data",
    )
    assert res_form_modern.status_code == 200
    d_modern = res_form_modern.get_json()

    assert d_legacy["success"] is True
    assert d_modern["success"] is True
    assert d_legacy["model_version_id"] == "lizard-lateral-v1"
    assert d_modern["model_version_id"] == "lizard-lateral-v1"
    assert d_legacy["predictions"] == d_modern["predictions"]


def test_predict_image_upload_and_path(client, tmp_path):
    """Verify prediction with uploaded image file vs image_path."""
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
    assert res_path.status_code == 200
    d_path = res_path.get_json()
    assert d_path["success"] is True
    assert d_path["model_version_id"] == "lizard-dorsal-v1"
    assert isinstance(d_path["predictions"], list)


def test_predict_invalid_model_version_id(client):
    """Verify that an invalid model_version_id returns 404 error response."""
    res = client.post("/api/predict", json={"model_version_id": "nonexistent-model-id"})
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False
    assert "not found" in data["error"]
