# Backend Utility Scripts

Run all scripts with the `lizard` conda environment:
```bash
conda run -n lizard python scripts/<script>.py --help
```

| Script | Purpose |
|--------|---------|
| `check_env.py` | Verify all required packages are importable in the current env |
| `run_yolo_inference.py` | Run YOLO OBB inference on a single image, print detections |
| `run_yolo_bulk.py` | Run YOLO OBB on a directory of images, report id/ruler detection |
| `run_yolo_legacy.py` | Run the legacy (non-OBB) bounding-box YOLO model on an image |
| `visualize_diff.py` | Visual diff of .pt vs .onnx landmark output side-by-side |
| `compare_models.py` | Numerical comparison of .pt vs .onnx landmark coordinates |
| `do_export.py` | Export a YOLO `.pt` model to ONNX (IR=8, opset=17) |
| `export_helper.py` | Set up an isolated venv for ONNX export to avoid env conflicts |
| `inspect_onnx.py` | Inspect ONNX model input/output shapes and run a dummy inference |
| `get_yolo_classes.py` | Print class name mapping from a YOLO `.pt` model |
| `predict_landmarks_flip.py` | Batch landmark prediction with vertical-flip inference |
| `toepad_preprocessing.py` | Preprocess toepad images before prediction |
| `test_api_extract_id.py` | Integration test for the `/extract_id` Flask endpoint |

