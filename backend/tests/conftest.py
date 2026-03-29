import importlib
import os

import pytest


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

