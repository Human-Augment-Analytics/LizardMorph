import os
import numpy as np
import cv2
import pytest
import importlib

def test_process_existing_missing_filename(client):
    res = client.post("/process_existing")
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data
    assert "filename" in data["error"].lower()

def test_process_existing_file_not_found(client):
    res = client.post("/process_existing?filename=nonexistent.jpg")
    assert res.status_code == 404
    data = res.get_json()
    assert "error" in data
    assert "not found" in data["error"].lower()

def test_process_existing_free_mode(client, tmp_path, monkeypatch):
    import uuid
    app_mod = importlib.import_module("app")
    session_id = str(uuid.uuid4())
    headers = {"X-Session-ID": session_id}
    
    # Get session upload folder
    session_data = app_mod.get_session_folders(session_id)
    upload_folder = session_data["upload_folder"]
    os.makedirs(upload_folder, exist_ok=True)
    
    # Create a dummy image file
    img_filename = "dummy_lizard.jpg"
    img_path = os.path.join(upload_folder, img_filename)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)

    # Request process_existing with free view_type
    res = client.post(
        f"/process_existing?filename={img_filename}&view_type=free",
        headers=headers
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["name"] == img_filename
    assert data["view_type"] == "free"
    assert "coords" in data

def test_process_existing_fallback_when_models_missing(client, monkeypatch):
    import uuid
    app_mod = importlib.import_module("app")
    session_id = str(uuid.uuid4())
    headers = {"X-Session-ID": session_id}

    session_data = app_mod.get_session_folders(session_id)
    upload_folder = session_data["upload_folder"]
    os.makedirs(upload_folder, exist_ok=True)

    img_filename = "test_fallback.jpg"
    img_path = os.path.join(upload_folder, img_filename)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)

    # Point predictors to non-existent paths
    monkeypatch.setattr(app_mod, "DORSAL_PREDICTOR_FILE", "nonexistent_dorsal.dat")
    monkeypatch.setattr(app_mod, "SCALE_PREDICTOR_FILE", "nonexistent_scale.dat")

    res = client.post(
        f"/process_existing?filename={img_filename}&view_type=dorsal",
        headers=headers
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["name"] == img_filename
