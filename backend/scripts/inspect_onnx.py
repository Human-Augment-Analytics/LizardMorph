import onnxruntime as ort
import numpy as np

# Create a random dummy image
dummy_input = np.random.randn(1, 3, 1024, 1024).astype(np.float32)

session = ort.InferenceSession("/Users/leyangloh/dev/LizardMorph/models/lizard-toe-pad/yolo_obb_6class.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: dummy_input})[0]

# Shape is [1, 11, 21504]
print(f"Output shape: {output.shape}")

# Look at the first anchor box (just to see value ranges)
box = output[0, :, 0]
print("First anchor output:")
for i, val in enumerate(box):
    print(f"  Channel {i}: {val}")
