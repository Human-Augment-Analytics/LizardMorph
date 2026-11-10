import hashlib

import mlflow
from fastapi.testclient import TestClient
import pytest

import config
from main import app


def test_real_registry_named_bundle_download_and_digest_rejection(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'MLFLOW_TRACKING_URI', f'sqlite:///{tmp_path}/mlflow.db')
    monkeypatch.chdir(tmp_path)
    previous = mlflow.get_tracking_uri()
    try:
        with TestClient(app) as http:
            registry = mlflow.MlflowClient()
            experiment = registry.create_experiment('bundles', artifact_location=(tmp_path / 'artifacts').as_uri())
            run = registry.create_run(experiment)
            path = tmp_path / 'bundle.zip'; path.write_bytes(b'bundle fixture')
            registry.log_artifact(run.info.run_id, str(path))
            for name in ['toepad-dlib', 'toepad-vit']:
                registry.create_registered_model(name)
                version = registry.create_model_version(name, source=f'runs:/{run.info.run_id}/bundle.zip', run_id=run.info.run_id,
                    tags={'bundle_format': 'lizard-pipeline-v1', 'bundle_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                response = http.get(f'/bundles/{name}/versions/{version.version}/archive')
                assert response.status_code == 200
                assert response.content == b'bundle fixture'
                assert response.headers['X-Bundle-SHA256'] == hashlib.sha256(response.content).hexdigest()
                registry.set_model_version_tag(name, version.version, 'bundle_sha256', 'wrong')
                assert http.get(f'/bundles/{name}/versions/{version.version}/archive').status_code == 502
            assert http.get('/bundles/missing/versions/1/archive').status_code == 404
    finally:
        mlflow.set_tracking_uri(previous)
