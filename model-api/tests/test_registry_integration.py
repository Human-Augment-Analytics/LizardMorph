"""Real SQLite registry exercised through the application's HTTP routes."""
import mlflow
from fastapi.testclient import TestClient

import config
from main import app


def test_promote_and_rollback_real_registry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path}/registry.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    previous = mlflow.get_tracking_uri()
    try:
        with TestClient(app) as http:
            registry = mlflow.MlflowClient()
            experiment = registry.create_experiment("local integration")
            run = registry.create_run(experiment, tags={"github_release_tag": "model/v1.1.0-obb"})
            registry.create_registered_model("H12_obb")
            first = registry.create_model_version("H12_obb", source="https://github.com/Human-Augment-Analytics/Lizard_Toepads/releases/tag/model/v1.1.0-obb", run_id=run.info.run_id)
            second = registry.create_model_version("H12_obb", source=first.source, run_id=run.info.run_id)
            for version in [first.version, second.version, first.version]:
                response = http.post(f"/models/H12_obb/versions/{version}/promote", json={"alias": "champion"})
                assert response.status_code == 200
                assert registry.get_model_version_by_alias("H12_obb", "champion").version == version
                assert registry.get_model_version("H12_obb", version).current_stage == "Production"
                versions = http.get("/models").json()["models"]
                assert len(versions) == 2
                assert next(v for v in versions if v["version"] == str(version))["aliases"] == ["champion"]
                assert sum(v["stage"] == "Production" for v in versions) == 1
            response = http.post("/models/H12_obb/versions/99999/promote", json={"alias": "champion"})
            assert response.status_code == 404
            assert registry.get_model_version_by_alias("H12_obb", "champion").version == first.version
    finally:
        mlflow.set_tracking_uri(previous)
