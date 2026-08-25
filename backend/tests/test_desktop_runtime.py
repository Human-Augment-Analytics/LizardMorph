import json
import os

import pytest

import app as app_module
from session_manager import SessionManager


def test_bind_host_is_loopback_when_supervised_by_the_tauri_sidecar(monkeypatch):
    monkeypatch.setattr(app_module.sys, "frozen", False, raising=False)
    monkeypatch.setenv("AUTOMORPH_PARENT_PID", "4242")
    assert app_module.get_bind_host() == "127.0.0.1"


def test_bind_host_is_loopback_for_a_frozen_bundle_without_a_supervisor_token(monkeypatch):
    monkeypatch.setattr(app_module.sys, "frozen", True, raising=False)
    monkeypatch.delenv("AUTOMORPH_PARENT_PID", raising=False)
    assert app_module.get_bind_host() == "127.0.0.1"


def test_bind_host_stays_wide_open_for_hosted_deployments(monkeypatch):
    monkeypatch.setattr(app_module.sys, "frozen", False, raising=False)
    monkeypatch.delenv("AUTOMORPH_PARENT_PID", raising=False)
    assert app_module.get_bind_host() == "0.0.0.0"


def test_health_echoes_the_supervisor_token(monkeypatch):
    monkeypatch.setenv("AUTOMORPH_PARENT_PID", "4242")
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = json.loads(response.data)
    assert payload["status"] == "ok"
    assert payload["supervisor_pid"] == "4242"


def test_health_reports_an_empty_supervisor_token_when_unsupervised(monkeypatch):
    monkeypatch.delenv("AUTOMORPH_PARENT_PID", raising=False)
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        response = client.get("/health")

    payload = json.loads(response.data)
    assert payload["status"] == "ok"
    assert payload["supervisor_pid"] == ""


def test_frozen_cleanup_removes_the_onefile_extraction_directory(monkeypatch, tmp_path):
    meipass = tmp_path / "_MEIabc123"
    meipass.mkdir()
    (meipass / "libfoo.so").write_bytes(b"payload")

    monkeypatch.setattr(app_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app_module.sys, "_MEIPASS", str(meipass), raising=False)

    assert app_module.cleanup_frozen_temp_dir() is True
    assert not meipass.exists()


def test_frozen_cleanup_leaves_a_onedir_bundle_alone(monkeypatch, tmp_path):
    bundle = tmp_path / "AutoMorph.app"
    bundle.mkdir()
    (bundle / "libfoo.so").write_bytes(b"payload")

    monkeypatch.setattr(app_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app_module.sys, "_MEIPASS", str(bundle), raising=False)

    assert app_module.cleanup_frozen_temp_dir() is False
    assert (bundle / "libfoo.so").exists()


def test_frozen_cleanup_is_a_no_op_when_running_from_source(monkeypatch, tmp_path):
    meipass = tmp_path / "_MEIabc123"
    meipass.mkdir()

    monkeypatch.setattr(app_module.sys, "frozen", False, raising=False)
    monkeypatch.setattr(app_module.sys, "_MEIPASS", str(meipass), raising=False)

    assert app_module.cleanup_frozen_temp_dir() is False
    assert meipass.exists()


@pytest.fixture()
def runtime_root(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    elsewhere = tmp_path / "elsewhere"
    (root / "sessions").mkdir(parents=True)
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return root, elsewhere


def test_clear_session_sweeps_the_runtime_root_not_the_process_cwd(runtime_root):
    root, elsewhere = runtime_root
    manager = SessionManager(str(root / "sessions"))
    session_id = manager.create_session()

    stray_in_root = root / "output_1.xml"
    stray_in_root.write_text("<x/>")
    stray_in_cwd = elsewhere / "output_1.xml"
    stray_in_cwd.write_text("<x/>")

    result = manager.clear_session(session_id)

    assert result["success"] is True
    assert not stray_in_root.exists()
    assert stray_in_cwd.exists()


TAURI_ORIGIN = "http://tauri.localhost"


def _preflight(client, path, method):
    return client.options(
        path,
        headers={
            "Origin": TAURI_ORIGIN,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "x-session-id",
        },
    )


def _allowed_methods(response):
    allowed = response.headers.get("Access-Control-Allow-Methods", "")
    return {method.strip().upper() for method in allowed.split(",") if method.strip()}


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/predictors/some-id", "DELETE"),
        ("/predictors", "POST"),
        ("/data", "POST"),
        ("/list_uploads", "GET"),
    ],
)
def test_cross_origin_preflight_allows_every_method_the_desktop_frontend_uses(path, method):
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        response = _preflight(client, path, method)

    assert response.status_code in (200, 204)
    assert response.headers.get("Access-Control-Allow-Origin") in (TAURI_ORIGIN, "*")
    assert method in _allowed_methods(response)


def test_cross_origin_delete_response_is_readable_by_the_desktop_webview():
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        response = client.delete(
            "/predictors/does-not-exist",
            headers={"Origin": TAURI_ORIGIN, "X-Session-ID": "session"},
        )

    assert response.headers.get("Access-Control-Allow-Origin") in (TAURI_ORIGIN, "*")
