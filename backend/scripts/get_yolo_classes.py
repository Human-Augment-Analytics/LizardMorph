from ultralytics import YOLO
import sys

try:
    model = YOLO('models/lizard-toe-pad/yolo_obb_6class.pt')
    print("NAMES:", model.names)
except Exception as e:
    print("ERROR:", e)
sys.exit(0)
