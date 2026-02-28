# Backend Utility Scripts

Run all scripts with the `lizard` conda environment:
```bash
conda run -n lizard python scripts/<script>.py
```

| Script | Purpose |
|--------|---------|
| `predict_landmarks_flip.py` | Run dlib landmark prediction with flip inference on a batch of images |
| `toepad_preprocessing.py` | Preprocess toepad images (resize, normalize) before prediction |
| `test_api_extract_id.py` | Integration test: hit the `/extract_id` Flask endpoint and verify response |
| `do_export.py` | Export a YOLO `.pt` model to ONNX with the correct opset/IR version |
| `export_helper.py` | Helper to set up an isolated venv for ONNX export (avoids env conflicts) |
| `inspect_onnx.py` | Inspect an ONNX model: print input/output shapes, run a dummy inference |
| `get_yolo_classes.py` | Print the class name mapping from a YOLO `.pt` model |
| `compare_models.py` | Compare landmark output between `.pt` and `.onnx` versions of a model |
