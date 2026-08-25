import sys
import threading
import types

import app


def _stub_model_file(path):
    path.write_bytes(b"stub")
    return str(path)


def test_a_second_caller_never_sees_a_half_loaded_predictor_set(monkeypatch, tmp_path):
    dorsal = _stub_model_file(tmp_path / "dorsal.dat")
    scale = _stub_model_file(tmp_path / "scale.dat")
    toe = _stub_model_file(tmp_path / "toe.dat")
    finger = _stub_model_file(tmp_path / "finger.dat")

    loading_scale = threading.Event()
    finish_loading = threading.Event()

    def fake_shape_predictor(path):
        if path == scale:
            loading_scale.set()
            finish_loading.wait(timeout=10)
        return f"predictor::{path}"

    monkeypatch.setitem(
        sys.modules, "dlib", types.SimpleNamespace(shape_predictor=fake_shape_predictor)
    )
    monkeypatch.setattr(app, "DORSAL_PREDICTOR_FILE", dorsal)
    monkeypatch.setattr(app, "SCALE_PREDICTOR_FILE", scale)
    monkeypatch.setattr(app, "TOEPAD_TOE_PREDICTOR", toe)
    monkeypatch.setattr(app, "TOEPAD_FINGER_PREDICTOR", finger)
    monkeypatch.setattr(app, "_cached_dlib_predictors", None)

    observed = {}

    def observe():
        observed["seen"] = dict(app.get_cached_dlib_predictors())

    loader = threading.Thread(target=app.get_cached_dlib_predictors)
    observer = threading.Thread(target=observe)

    loader.start()
    loading_scale.wait(timeout=10)
    observer.start()
    observer.join(timeout=1)
    finish_loading.set()
    loader.join(timeout=10)
    observer.join(timeout=10)

    assert set(observed["seen"]) == {"dorsal", "scale", "toe", "finger"}
    assert set(app.get_cached_dlib_predictors()) == {"dorsal", "scale", "toe", "finger"}


def test_concurrent_callers_build_the_yolo_model_once(monkeypatch, tmp_path):
    model_path = _stub_model_file(tmp_path / "toepad.pt")
    constructed = []
    building = threading.Event()
    finish_building = threading.Event()

    class FakeYolo:
        def __init__(self, path, task=None):
            constructed.append(path)
            building.set()
            finish_building.wait(timeout=10)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    monkeypatch.setenv("USE_ORT_QUANTIZED", "false")
    monkeypatch.setattr(app, "TOEPAD_YOLO_MODEL", model_path)
    monkeypatch.setattr(app, "_cached_yolo_model", None)

    first = threading.Thread(target=app.get_cached_yolo_model)
    second = threading.Thread(target=app.get_cached_yolo_model)

    first.start()
    building.wait(timeout=10)
    second.start()
    second.join(timeout=1)
    constructed_while_loading = list(constructed)
    finish_building.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert constructed_while_loading == [model_path]
    assert constructed == [model_path]
    assert isinstance(app.get_cached_yolo_model(), FakeYolo)


def test_a_model_load_does_not_block_an_unrelated_cache(monkeypatch, tmp_path):
    model_path = _stub_model_file(tmp_path / "toepad.pt")
    building = threading.Event()
    finish_building = threading.Event()

    class FakeYolo:
        def __init__(self, path, task=None):
            building.set()
            finish_building.wait(timeout=10)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYolo))
    monkeypatch.setenv("USE_ORT_QUANTIZED", "false")
    monkeypatch.setattr(app, "TOEPAD_YOLO_MODEL", model_path)
    monkeypatch.setattr(app, "_cached_yolo_model", None)
    monkeypatch.setattr(app, "_cached_dlib_predictors", None)
    monkeypatch.setattr(app, "_load_dlib_predictors", lambda: {"toe": "predictor"})

    observed = {}
    yolo_loader = threading.Thread(target=app.get_cached_yolo_model)
    predictor_caller = threading.Thread(
        target=lambda: observed.update(app.get_cached_dlib_predictors())
    )

    yolo_loader.start()
    building.wait(timeout=10)
    predictor_caller.start()
    predictor_caller.join(timeout=5)
    predictors_ready_during_yolo_load = not predictor_caller.is_alive()
    observed_during_yolo_load = dict(observed)
    yolo_still_loading = yolo_loader.is_alive()
    finish_building.set()
    yolo_loader.join(timeout=10)
    predictor_caller.join(timeout=10)

    assert yolo_still_loading
    assert predictors_ready_during_yolo_load
    assert observed_during_yolo_load == {"toe": "predictor"}


def test_an_empty_predictor_set_is_cached_instead_of_reprobed(monkeypatch):
    attempts = []

    def counting_load():
        attempts.append(1)
        return {}

    monkeypatch.setattr(app, "_cached_dlib_predictors", None)
    monkeypatch.setattr(app, "_load_dlib_predictors", counting_load)

    first = app.get_cached_dlib_predictors()
    second = app.get_cached_dlib_predictors()

    assert first == {} and second == {}
    assert attempts == [1]
