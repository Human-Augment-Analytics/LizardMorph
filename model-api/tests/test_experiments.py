import os
os.environ["GITHUB_REPO"] = "org/repo"
os.environ["GITHUB_TOKEN"] = "ghp_test"

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def make_experiment(exp_id, name):
    e = MagicMock()
    e.experiment_id = exp_id
    e.name = name
    e.lifecycle_stage = "active"
    return e


def make_run(run_id, name, metrics, params):
    r = MagicMock()
    r.info.run_id = run_id
    r.info.run_name = name
    r.info.status = "FINISHED"
    r.data.metrics = metrics
    r.data.params = params
    return r


def test_list_experiments():
    mock_exps = [make_experiment("1", "lizard-toepad-detection")]
    with patch("mlflow.search_experiments", return_value=mock_exps):
        response = client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert data["experiments"][0]["experiment_id"] == "1"
    assert data["experiments"][0]["name"] == "lizard-toepad-detection"


def test_list_experiments_empty():
    with patch("mlflow.search_experiments", return_value=[]):
        response = client.get("/experiments")
    assert response.status_code == 200
    assert response.json()["experiments"] == []


def test_get_runs():
    mock_runs = [make_run("abc123", "H8_fp16", {"mAP50": 0.91}, {"epochs": "100"})]
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_runs.return_value = mock_runs
        response = client.get("/experiments/1/runs")
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "1"
    assert data["runs"][0]["run_id"] == "abc123"
    assert data["runs"][0]["metrics"]["mAP50"] == 0.91


def test_get_runs_empty():
    with patch("mlflow.MlflowClient") as MockClient:
        MockClient.return_value.search_runs.return_value = []
        response = client.get("/experiments/99/runs")
    assert response.status_code == 200
    assert response.json()["runs"] == []
