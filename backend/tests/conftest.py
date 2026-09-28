import importlib
import os
import tempfile
from unittest.mock import patch

import pytest


def pytest_configure(config):
    # Configure writable stores before test collection imports the Flask app.
    # Never register test models or recover jobs in a researcher's real stores.
    config._research_test_dir = tempfile.TemporaryDirectory(prefix="automorph-tests-")
    root = config._research_test_dir.name
    config._research_test_env = patch.dict(os.environ, {
        "DB_PATH": os.path.join(root, "test.db"),
        "RUNS_DIR": os.path.join(root, "runs"),
        "PREDICTOR_LIBRARY_DIR": os.path.join(root, "predictors"),
        "SESSION_DIR": os.path.join(root, "sessions"),
    })
    config._research_test_env.start()


def pytest_unconfigure(config):
    if hasattr(config, "_research_test_env"):
        config._research_test_env.stop()
        config._research_test_dir.cleanup()


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """
    Flask test client with predictor library paths redirected to tmp.

    Notes:
    - backend/app.py currently exposes a module-level `app`.
    - Endpoints will read PREDICTOR_LIBRARY_* globals; tests monkeypatch them.
    """
    app_mod = importlib.import_module("app")
    flask_app = getattr(app_mod, "app")

    base = tmp_path / "custom_predictors"
    index_path = base / "predictors.json"
    files_dir = base / "files"
    os.makedirs(files_dir, exist_ok=True)

    monkeypatch.setattr(app_mod, "PREDICTOR_LIBRARY_DIR", str(base), raising=False)
    monkeypatch.setattr(app_mod, "PREDICTOR_LIBRARY_INDEX", str(index_path), raising=False)
    monkeypatch.setattr(app_mod, "PREDICTOR_LIBRARY_FILES", str(files_dir), raising=False)

    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c

