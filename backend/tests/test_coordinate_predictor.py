import numpy as np
from backend.inference.coordinates import CoordinatePredictor


def test_notebook_letterbox_bgr_normalization_and_inverse_transform():
    predictor = CoordinatePredictor.__new__(CoordinatePredictor)
    tensors = []
    def run(tensor):
        tensors.append(tensor)
        return np.full((9, 2), .5, dtype=np.float32)
    predictor._run = run
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[20:120, 30:80] = [255, 0, 128]  # deliberately different BGR channels
    points = predictor.predict(image, [[30,20],[80,20],[80,120],[30,120]])
    np.testing.assert_allclose(points, [[55, 70]] * 9)
    # Center pixel matches original BGR channel order, rather than RGB conversion.
    expected = (np.array([1,0,128/255], dtype=np.float32) - [.485,.456,.406]) / [.229,.224,.225]
    np.testing.assert_allclose(tensors[0][0,:,112,112], expected, atol=1e-6)
    assert tensors[0].dtype == np.float32
    assert tensors[0].shape == (1,3,224,224)


def test_switching_models_evicts_old_detector_before_new_allocation(monkeypatch):
    import backend.inference.engine as module
    monkeypatch.setenv('MODEL_DETECTOR_CACHE_LIMIT', '1')
    engine = module.GenericPipelineEngine()
    observed = []
    def create(path, names, **kwargs):
        observed.append(len(engine._detector_cache))
        return object()
    monkeypatch.setattr(module, 'GenericOrtYoloOBBDetector', create)
    first = engine._get_detector('first.onnx', ['digit'])
    second = engine._get_detector('second.onnx', ['digit'])
    assert first is not second
    assert observed == [0, 0]
    assert len(engine._detector_cache) == 1
    assert engine._get_detector('second.onnx', ['digit']) is second
