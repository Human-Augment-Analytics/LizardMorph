# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from importlib.util import find_spec


project_dir = os.environ["AUTOMORPH_PROJECT_DIR"]
backend_dir = os.path.join(project_dir, "backend")
sidecar_name = os.environ["AUTOMORPH_SIDECAR_NAME"]

hiddenimports = ["native_ocr", "onnxruntime", "pytesseract"]
if sys.platform == "darwin":
    hiddenimports += [
        module_name
        for module_name in ("Vision", "Quartz", "objc")
        if find_spec(module_name) is not None
    ]
elif sys.platform == "win32":
    hiddenimports += ["asyncio", "winocr"]

datas = []
model_candidates = [
    ("models/lizard-x-ray/dorsal_predictor_clahe_best.dat", "models/lizard-x-ray"),
    ("models/lizard-x-ray/scale_predictor_clahe.dat", "models/lizard-x-ray"),
    ("models/lizard-x-ray/lateral_predictor_auto.dat", "models/lizard-x-ray"),
    ("models/lizard-toe-pad/yolo_obb_6class_h7_int8.onnx", "models/lizard-toe-pad"),
    ("models/lizard-toe-pad/yolo_obb_6class_h7.onnx", "models/lizard-toe-pad"),
    ("models/lizard-toe-pad/ml_morph_best.dat", "models/lizard-toe-pad"),
    ("models/lizard-toe-pad/toe_predictor_obb.dat", "models/lizard-toe-pad"),
    ("models/lizard-toe-pad/finger_predictor_obb.dat", "models/lizard-toe-pad"),
    ("models/lizard-toe-pad/lizard_scale.dat", "models/lizard-toe-pad"),
]
for relative_source, destination in model_candidates:
    source = os.path.join(project_dir, relative_source)
    if os.path.isfile(source):
        datas.append((source, destination))

a = Analysis(
    [os.path.join(backend_dir, "app.py")],
    pathex=[project_dir, backend_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython", "boto3", "botocore", "easyocr", "jax", "jaxlib",
        "jupyter", "librosa", "nbformat", "openpyxl", "openvino", "pytest",
        "sklearn", "sqlalchemy", "tensorflow", "tkinter", "torch",
        "torchaudio", "transformers", "ultralytics",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name=sidecar_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
