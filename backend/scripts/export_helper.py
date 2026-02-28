import os
import subprocess
import sys

# Create venv
subprocess.run([sys.executable, "-m", "venv", "venv_export"])
# Install ultralytics
pip_cmd = os.path.join("venv_export", "bin", "pip")
subprocess.run([pip_cmd, "install", "ultralytics", "onnx"])

# Run export
py_cmd = os.path.join("venv_export", "bin", "python")
export_script = """
from ultralytics import YOLO
import onnx
import sys

model_path = '/Users/leyangloh/dev/LizardMorph/models/lizard-toe-pad/yolo_obb_6class.pt'
model = YOLO(model_path)
model.export(format='onnx', imgsz=1024, opset=12) # or imgsz=640?
print("Export complete")
"""
with open("do_export.py", "w") as f:
    f.write(export_script)

subprocess.run([py_cmd, "do_export.py"])
