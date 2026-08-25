import threading

import numpy as np

import ort_inference


class _TensorInfo:
    name = "images"


class _FakeSession:
    def __init__(self, output, providers):
        self.output = output
        self.providers = providers

    def get_inputs(self):
        return [_TensorInfo()]

    def get_outputs(self):
        return [_TensorInfo()]

    def get_providers(self):
        return self.providers

    def run(self, *_args, **_kwargs):
        return [self.output]


def test_detector_retries_non_finite_accelerator_output_on_cpu(monkeypatch):
    bad_output = np.full((1, 11, 1), np.nan, dtype=np.float32)
    good_output = np.zeros((1, 11, 1), dtype=np.float32)
    sessions = []

    def make_session(_model_path, providers):
        output = bad_output if providers != ["CPUExecutionProvider"] else good_output
        session = _FakeSession(output, providers)
        sessions.append(session)
        return session

    monkeypatch.setattr(ort_inference, "_get_execution_providers", lambda: ["CoreMLExecutionProvider"])
    monkeypatch.setattr(ort_inference.ort, "InferenceSession", make_session)

    detector = ort_inference.OrtYoloDetector("model.onnx")
    result = detector._run(np.zeros((1, 3, 4, 4), dtype=np.float32))

    assert np.array_equal(result, good_output)
    assert [session.providers for session in sessions] == [
        ["CoreMLExecutionProvider"],
        ["CPUExecutionProvider"],
    ]


def test_a_second_caller_still_falls_back_after_another_thread_swapped_the_session(monkeypatch):
    bad_output = np.full((1, 11, 1), np.nan, dtype=np.float32)
    good_output = np.zeros((1, 11, 1), dtype=np.float32)
    sessions = []
    latecomer_entered_inference = threading.Event()
    latecomer_may_finish_inference = threading.Event()

    class _AcceleratorSession(_FakeSession):
        def run(self, *_args, **_kwargs):
            if threading.current_thread().name == "latecomer":
                latecomer_entered_inference.set()
                assert latecomer_may_finish_inference.wait(10)
            return [self.output]

    def make_session(_model_path, providers):
        if providers == ["CPUExecutionProvider"]:
            session = _FakeSession(good_output, providers)
        else:
            session = _AcceleratorSession(bad_output, providers)
        sessions.append(session)
        return session

    monkeypatch.setattr(
        ort_inference, "_get_execution_providers", lambda: ["CoreMLExecutionProvider"]
    )
    monkeypatch.setattr(ort_inference.ort, "InferenceSession", make_session)

    detector = ort_inference.OrtYoloDetector("model.onnx")
    tensor = np.zeros((1, 3, 4, 4), dtype=np.float32)
    outcomes = {}

    def call(name):
        try:
            outcomes[name] = detector._run(tensor)
        except Exception as error:  # noqa: BLE001 - recorded for the assertion below
            outcomes[name] = error

    latecomer = threading.Thread(target=call, args=("latecomer",), name="latecomer")
    latecomer.start()
    assert latecomer_entered_inference.wait(10)

    first = threading.Thread(target=call, args=("first",), name="first")
    first.start()
    first.join(10)

    latecomer_may_finish_inference.set()
    latecomer.join(10)

    assert isinstance(outcomes["first"], np.ndarray)
    assert np.array_equal(outcomes["first"], good_output)
    assert isinstance(outcomes["latecomer"], np.ndarray)
    assert np.array_equal(outcomes["latecomer"], good_output)
    assert [session.providers for session in sessions] == [
        ["CoreMLExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
