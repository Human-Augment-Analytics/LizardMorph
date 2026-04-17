import os
os.environ["GITHUB_REPO"] = "org/repo"
os.environ["GITHUB_TOKEN"] = "ghp_test"

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def make_model_version(name, version, stage, run_id="run-abc123"):
    mv = MagicMock()
    mv.name = name
    mv.version = version
    mv.current_stage = stage
    mv.run_id = run_id
    return mv


def make_registered_model(name, versions):
    m = MagicMock()
    m.name = name
    m.latest_versions = versions
    return m


def test_list_models():
    mv = make_model_version("H8_obb_botonly", "3", "Production")
    mock_models = [make_registered_model("H8_obb_botonly", [mv])]
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_registered_models.return_value = mock_models
        response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["models"][0]["name"] == "H8_obb_botonly"
    assert data["models"][0]["stage"] == "Production"


def test_list_models_empty():
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_registered_models.return_value = []
        response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["models"] == []


MOCK_METADATA = {
    "version": "v1.0.0-obb",
    "trained": "2026-03-13",
    "author": "Test <test@example.com>",
    "architecture": "yolo11m-obb.pt",
    "task": "obb",
    "config": "H11_obb",
    "dataset": {"nc": 4, "names": ["finger", "toe", "ruler", "id"], "train_images": 679, "val_images": 170},
    "training": {"epochs": 300, "batch": 32, "imgsz": 1280, "patience": 50},
    "metrics": {"best_epoch": 175, "epochs_completed": 225, "mAP50": 0.972, "mAP50-95": 0.942, "precision": 0.966, "recall": 0.959},
    "assets": {"pt": "best.pt", "fp16": "best_fp16.onnx", "fp32": "best_fp32.onnx"},
}

MOCK_RELEASE = {
    "tag_name": "model/v1.0.0-obb",
    "assets": [
        {"name": "best_fp16.onnx", "id": 1001, "url": "https://api.github.com/repos/org/repo/releases/assets/1001", "browser_download_url": "https://github.com/org/repo/releases/download/model/v1.0.0-obb/best_fp16.onnx"},
        {"name": "best_fp32.onnx", "id": 1002, "url": "https://api.github.com/repos/org/repo/releases/assets/1002", "browser_download_url": "https://github.com/org/repo/releases/download/model/v1.0.0-obb/best_fp32.onnx"},
        {"name": "best.pt",        "id": 1003, "url": "https://api.github.com/repos/org/repo/releases/assets/1003", "browser_download_url": "https://github.com/org/repo/releases/download/model/v1.0.0-obb/best.pt"},
        {"name": "metadata.json",  "id": 1004, "url": "https://api.github.com/repos/org/repo/releases/assets/1004", "browser_download_url": "https://github.com/org/repo/releases/download/model/v1.0.0-obb/metadata.json"},
    ],
}


def make_mock_run(tags=None):
    run = MagicMock()
    run.data.tags = tags or {}
    return run


def mock_httpx_get(url, **kwargs):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if "releases/tags/" in url:
        resp.json.return_value = MOCK_RELEASE
    elif "releases/assets/" in url:
        resp.json.return_value = MOCK_METADATA
    else:
        resp.json.return_value = MOCK_METADATA
    return resp


def test_latest_model():
    mv = make_model_version("H8_obb_botonly", "3", "Production", run_id="run-abc123")
    mock_run = make_mock_run(tags={"github_release_tag": "model/v1.0.0-obb"})
    with patch("mlflow.MlflowClient") as MockClient, \
         patch("httpx.get", side_effect=mock_httpx_get):
        MockClient.return_value.search_registered_models.return_value = [
            make_registered_model("H8_obb_botonly", [mv])
        ]
        MockClient.return_value.get_latest_versions.return_value = [mv]
        MockClient.return_value.get_run.return_value = mock_run
        response = client.get("/model/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.0.0-obb"
    assert data["metrics"]["mAP50"] == 0.972
    assert data["assets"]["fp16"].endswith("best_fp16.onnx")
    assert data["assets"]["fp32"].endswith("best_fp32.onnx")
    assert data["assets"]["pt"].endswith("best.pt")


def test_latest_model_no_production():
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_registered_models.return_value = [
            make_registered_model("H8_obb_botonly", [])
        ]
        MockClient.return_value.get_latest_versions.return_value = []
        response = client.get("/model/latest")
    assert response.status_code == 404
    assert "Production" in response.json()["detail"]


def test_latest_model_no_release_tag():
    mv = make_model_version("H8_obb_botonly", "3", "Production", run_id="run-abc123")
    mock_run = make_mock_run(tags={})  # no github_release_tag
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_registered_models.return_value = [
            make_registered_model("H8_obb_botonly", [mv])
        ]
        MockClient.return_value.get_latest_versions.return_value = [mv]
        MockClient.return_value.get_run.return_value = mock_run
        response = client.get("/model/latest")
    assert response.status_code == 404
    assert "github_release_tag" in response.json()["detail"]


def test_latest_model_no_metadata_asset():
    mv = make_model_version("H8_obb_botonly", "3", "Production", run_id="run-abc123")
    mock_run = make_mock_run(tags={"github_release_tag": "model/v1.0.0-obb"})
    release_no_metadata = {
        "tag_name": "model/v1.0.0-obb",
        "assets": [
            {"name": "best_fp16.onnx", "id": 1001, "url": "https://api.github.com/repos/org/repo/releases/assets/1001", "browser_download_url": "https://github.com/org/repo/releases/download/model/v1.0.0-obb/best_fp16.onnx"},
        ],
    }
    def mock_get_no_meta(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = release_no_metadata
        return resp

    with patch("mlflow.MlflowClient") as MockClient, \
         patch("httpx.get", side_effect=mock_get_no_meta):
        MockClient.return_value.search_registered_models.return_value = [
            make_registered_model("H8_obb_botonly", [mv])
        ]
        MockClient.return_value.get_latest_versions.return_value = [mv]
        MockClient.return_value.get_run.return_value = mock_run
        response = client.get("/model/latest")
    assert response.status_code == 404
    assert "metadata.json" in response.json()["detail"]
