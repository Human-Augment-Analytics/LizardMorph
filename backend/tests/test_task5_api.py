import json
import pytest
from backend.app import app, model_registry_repo, register_built_in_legacy_models


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_models_endpoint(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert isinstance(data["models"], list)
    # Check that built-in models are returned
    model_ids = [m["id"] for m in data["models"]]
    assert "lizard-dorsal-v1" in model_ids
    assert "lizard-lateral-v1" in model_ids
    assert "lizard-toepad-v1" in model_ids
    manifests = {model["id"]: model["manifest"] for model in data["models"]}
    assert len(manifests["lizard-dorsal-v1"]["landmark_schemas"]["dorsal"]["points"]) == 34
    assert len(manifests["lizard-lateral-v1"]["landmark_schemas"]["lateral"]["points"]) == 9
    assert len(manifests["lizard-toepad-v1"]["landmark_schemas"]["toepad"]["points"]) == 9


def test_api_projects_endpoints(client):
    # Test POST /api/projects
    create_res = client.post(
        "/api/projects",
        json={"name": "Anolis Morph Project", "organism": "Anolis carolinensis"},
    )
    assert create_res.status_code == 201
    create_data = create_res.get_json()
    assert create_data["success"] is True
    proj = create_data["project"]
    assert proj["name"] == "Anolis Morph Project"
    assert proj["organism"] == "Anolis carolinensis"
    assert "id" in proj

    # Test GET /api/projects
    list_res = client.get("/api/projects")
    assert list_res.status_code == 200
    list_data = list_res.get_json()
    assert list_data["success"] is True
    assert isinstance(list_data["projects"], list)
    proj_ids = [p["id"] for p in list_data["projects"]]
    assert proj["id"] in proj_ids


def test_api_train_and_status_endpoints(client, monkeypatch):
    import backend.app as app_mod

    dataset = {
        "images": [
            {
                "image_id": f"img_{index}",
                "file_path": f"image_{index}.jpg",
                "width": 100,
                "height": 100,
                "objects": [{
                    "object_id": f"obj_{index}",
                    "class_name": "object",
                    "obb": [50, 50, 40, 40, 0],
                    "landmarks": [{"name": "0", "x": 50, "y": 50}],
                }],
            }
            for index in (1, 2)
        ]
    }
    monkeypatch.setattr(app_mod.training_orchestrator, "submit_job", lambda **kwargs: "job_test123")
    monkeypatch.setattr(
        app_mod.training_orchestrator,
        "get_status",
        lambda job_id: {"status": "running", "stage": "Training", "progress": 0.5, "metrics": {}},
    )
    # Test POST /api/train
    train_res = client.post(
        "/api/train",
        json={
            "project_id": "test_proj_1",
            "dataset": dataset,
            "config": {"epochs": 1, "model_type": "ml_morph"},
        },
    )
    assert train_res.status_code == 200
    train_data = train_res.get_json()
    assert train_data["success"] is True
    assert "job_id" in train_data
    job_id = train_data["job_id"]

    # Test GET /api/train/<job_id>
    status_res = client.get(f"/api/train/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.get_json()
    assert status_data["success"] is True
    assert status_data["job_id"] == job_id
    assert "status" in status_data
    assert "stage" in status_data
    assert "progress" in status_data
    assert "metrics" in status_data


def test_api_derive_boxes_endpoint(client):
    tps_sample = """LM=2
10.0 20.0
30.0 40.0
IMAGE=specimen1.jpg
ID=Anolis_001
"""
    res = client.post(
        "/api/dataset/derive-boxes",
        json={"tps_content": tps_sample, "padding": 0.25},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["padding"] == 0.25
    assert len(data["images"]) == 1
    assert data["images"][0]["file_path"] == "specimen1.jpg"
    objs = data["images"][0]["objects"]
    assert len(objs) == 1
    assert len(objs[0]["obb"]) == 5


def test_api_cancel_training_endpoint(client, monkeypatch):
    import backend.app as app_mod

    job_id = "job_cancel123"
    monkeypatch.setattr(app_mod.training_orchestrator, "cancel_job", lambda requested: requested == job_id)

    # Cancel job
    cancel_res = client.post(f"/api/train/{job_id}/cancel")
    assert cancel_res.status_code == 200
    cancel_data = cancel_res.get_json()
    assert cancel_data["success"] is True
    assert cancel_data["job_id"] == job_id
