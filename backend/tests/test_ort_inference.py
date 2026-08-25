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
