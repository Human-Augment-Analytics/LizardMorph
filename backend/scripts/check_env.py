"""
check_env.py
------------
Verify that all required Python packages and modules are importable.
Run this after setting up the conda environment to confirm everything is installed.

Usage:
    conda run -n lizard python scripts/check_env.py
"""
import sys

print(f"Python: {sys.version}\n")

checks = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("cv2 (OpenCV)", "cv2"),
    ("dlib", "dlib"),
    ("matplotlib", "matplotlib"),
    ("PIL (Pillow)", "PIL"),
    ("flask", "flask"),
    ("flask_cors", "flask_cors"),
    ("gunicorn", "gunicorn"),
    ("ultralytics (YOLO)", "ultralytics"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("easyocr", "easyocr"),
    ("onnxruntime", "onnxruntime"),
    ("utils (local)", "utils"),
]

ok = []
fail = []
for name, module in checks:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"  ✓ {name} ({ver})")
        ok.append(name)
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        fail.append(name)

print(f"\n{len(ok)} ok, {len(fail)} failed")
if fail:
    print("  Missing:", ", ".join(fail))
    sys.exit(1)
