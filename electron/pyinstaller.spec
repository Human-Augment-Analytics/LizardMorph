# -*- mode: python ; coding: utf-8 -*-
import os

project_dir = os.path.abspath(os.path.join(SPECPATH, '..'))
backend_dir = os.path.join(project_dir, 'backend')
models_dir = os.path.join(project_dir, 'models')

import sys

base_hiddenimports = [
    'flask',
    'flask_cors',
    'werkzeug',
    'dlib',
    'cv2',
    'numpy',
    'pandas',
    'PIL',
    'onnxruntime',
    'native_ocr',
]

if sys.platform == 'darwin':
    base_hiddenimports.extend([
        'Vision',
        'Quartz',
        'objc',
    ])
elif sys.platform == 'win32':
    base_hiddenimports.extend([
        'winocr',
        'asyncio',
    ])

a = Analysis(
    [os.path.join(backend_dir, 'app.py')],
    pathex=[backend_dir],
    binaries=[],
    datas=[
        (os.path.join(models_dir, 'lizard-x-ray', 'better_predictor_auto.dat'), 'models/lizard-x-ray'),
        (os.path.join(models_dir, 'lizard-x-ray', 'lateral_predictor_auto.dat'), 'models/lizard-x-ray'),
        (os.path.join(models_dir, 'lizard-toe-pad', 'yolo_obb_6class_h7.onnx'), 'models/lizard-toe-pad'),
        (os.path.join(models_dir, 'lizard-toe-pad', 'toe_predictor_obb.dat'), 'models/lizard-toe-pad'),
        (os.path.join(models_dir, 'lizard-toe-pad', 'finger_predictor_obb.dat'), 'models/lizard-toe-pad'),
        (os.path.join(models_dir, 'lizard-toe-pad', 'lizard_scale.dat'), 'models/lizard-toe-pad'),
    ],
    hiddenimports=base_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'easyocr',
        'torch',
        'torchvision',
        'ultralytics',
        # matplotlib is needed by visual_individual_performance.py
        'tkinter',
        'tensorflow',
        'tflite',
        'tflite_runtime',
        'keras',
        'jax',
        'jaxlib',
        'tensorstore',
        'pyarrow',
        'grpc',
        'grpcio',
        'h5py',
        'tensorboard',
        'openvino',
        'sklearn',
        'skimage',
        'shapely',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='backend',
)
