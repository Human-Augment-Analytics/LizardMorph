
from ultralytics import YOLO
import onnx
import sys

model_path = '/Users/leyangloh/dev/LizardMorph/models/lizard-toe-pad/yolo_obb_6class.pt'
model = YOLO(model_path)
model.export(format='onnx', imgsz=1024, opset=12) # or imgsz=640?
print("Export complete")
