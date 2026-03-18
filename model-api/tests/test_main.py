import os
os.environ["GITHUB_REPO"] = "org/repo"
os.environ["GITHUB_TOKEN"] = "ghp_test"

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint_exists():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "mlflow_reachable" in data


def test_unknown_route_returns_404():
    response = client.get("/nonexistent")
    assert response.status_code == 404
