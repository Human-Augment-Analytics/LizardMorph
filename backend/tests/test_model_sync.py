import hashlib
import importlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest


@pytest.fixture()
def registry(monkeypatch):
    fixture_root = os.environ.get("TEST_FIXTURES")
    if not fixture_root:
        pytest.skip("TEST_FIXTURES must point to real published ONNX and stale H7 fixtures")
    fixture_dir = Path(fixture_root)
    state = {
        "body": (fixture_dir / "release-v1.1.0/best_fp16.onnx").read_bytes(),
        "dataset": {"nc": 4, "names": ["finger", "toe", "ruler", "id"]},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/model/latest":
                body = json.dumps({
                    "version": "v1.1.0-obb", "task": "obb", "dataset": state["dataset"],
                    "assets": {"fp16": f"http://127.0.0.1:{self.server.server_port}/model.onnx"},
                }).encode()
            else:
                body = state["body"]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("MODEL_API_URL", f"http://127.0.0.1:{server.server_port}")
    yield state
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.fixture()
def active_model(monkeypatch, tmp_path):
    module = importlib.import_module("app")
    model = tmp_path / "active.onnx"
    model.write_bytes((Path(os.environ["TEST_FIXTURES"]) / "release-v1.1.0/best_fp16.onnx").read_bytes())
    monkeypatch.setattr(module, "TOEPAD_YOLO_MODEL", str(model))
    monkeypatch.setattr(module, "ID_EXTRACTOR_MODEL", str(model))
    monkeypatch.setattr(module, "_cached_yolo_model", None)
    monkeypatch.setattr(module, "_cached_id_model", None)
    monkeypatch.setenv("USE_ORT_QUANTIZED", "true")
    return module, model


def test_sync_replaces_stale_int8_and_shared_caches(client, active_model, registry):
    module, model = active_model
    sibling = model.with_name("active_int8.onnx")
    sibling.write_bytes((Path(os.environ["TEST_FIXTURES"]) / "incoming/yolo_obb_6class_h7_int8.onnx").read_bytes())
    assert module.get_cached_yolo_model().num_classes == 6
    response = client.post("/api/sync-model")
    assert response.status_code == 200
    assert module.get_cached_yolo_model().num_classes == 4
    assert module.get_cached_yolo_model().schema == "h11_4class"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == "ed95b2ba77ed968ae17fc7198e7626e2371f90ad28544c463510ac8268834f02"
    assert module.get_cached_id_model() is module.get_cached_yolo_model()
    module._cached_yolo_model = None
    assert module.get_cached_yolo_model().num_classes == 4


@pytest.mark.parametrize("failure", ["invalid_onnx", "wrong_schema"])
def test_failed_sync_preserves_working_file_and_detector(client, active_model, registry, failure):
    module, model = active_model
    previous_bytes = model.read_bytes()
    detector = module.get_cached_yolo_model()
    if failure == "invalid_onnx":
        registry["body"] = b"interrupted or invalid download"
    else:
        registry["dataset"] = {"nc": 6, "names": ["up_finger", "up_toe", "bot_finger", "bot_toe", "ruler", "id"]}
    response = client.post("/api/sync-model")
    assert response.status_code == 502
    assert model.read_bytes() == previous_bytes
    assert module.get_cached_yolo_model() is detector
    assert sorted(p.name for p in model.parent.glob("*.onnx")) == ["active.onnx"]
