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
manifest_path = os.path.join(project_dir, "src-tauri", "desktop-models.txt")
with open(manifest_path, encoding="utf-8") as manifest:
    for raw_entry in manifest:
        entry = raw_entry.split("#", 1)[0].split()
        if not entry:
            continue
        requirement, relative_source, destination = entry
        source = os.path.join(project_dir, relative_source)
        if os.path.isfile(source):
            datas.append((source, destination))
        elif requirement == "required":
            raise SystemExit(f"Required desktop model is missing: {relative_source}")

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
