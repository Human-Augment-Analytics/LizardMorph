#!/usr/bin/env python3
"""
Optimize existing ONNX model to accept variable input sizes with direct resize
"""

import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, numpy_helper
from onnx import version_converter

def add_preprocessing_to_onnx(input_model_path, output_model_path):
    """
    Add preprocessing layer to existing ONNX model
    """
    print(f"Loading ONNX model: {input_model_path}")
    model = onnx.load(input_model_path)
    
    # Get the original input
    original_input = model.graph.input[0]
    print(f"Original input: {original_input.name}, shape: {[d.dim_value for d in original_input.type.tensor_type.shape.dim]}")
    
    # Create new input that accepts variable sizes
    new_input = helper.make_tensor_value_info(
        'raw_images',
        onnx.TensorProto.FLOAT,
        [1, 3, 'height', 'width']  # Variable height and width
    )
    
    # Create preprocessing nodes
    # Node 1: Get image dimensions
    get_shape = helper.make_node(
        'Shape',
        inputs=['raw_images'],
        outputs=['image_shape']
    )
    
    # Node 2: Extract height and width
    get_height = helper.make_node(
        'Gather',
        inputs=['image_shape', 'height_idx'],
        outputs=['height']
    )
    
    get_width = helper.make_node(
        'Gather', 
        inputs=['image_shape', 'width_idx'],
        outputs=['width']
    )
    
    # Node 3: Calculate scale factor
    calculate_scale = helper.make_node(
        'Div',
        inputs=['target_size', 'max_dim'],
        outputs=['scale']
    )
    
    # Node 4: Resize image
    resize_image = helper.make_node(
        'Resize',
        inputs=['raw_images', '', '', 'scale'],
        outputs=['resized_images'],
        mode='linear',
        coordinate_transformation_mode='half_pixel'
    )
    
    # Node 5: Convert to grayscale
    # This is complex in ONNX, so we'll use a simpler approach
    
    # Create constants
    target_size_const = helper.make_tensor('target_size', onnx.TensorProto.INT64, [1], [640])
    height_idx_const = helper.make_tensor('height_idx', onnx.TensorProto.INT64, [1], [2])
    width_idx_const = helper.make_tensor('width_idx', onnx.TensorProto.INT64, [1], [3])
    
    # Create new graph
    new_nodes = [
        get_shape, get_height, get_width, calculate_scale, resize_image
    ]
    
    # Update the model
    model.graph.input[0] = new_input
    model.graph.node.extend(new_nodes)
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"✅ Saved optimized model: {output_model_path}")
    
    return True

def test_optimized_model():
    """
    Test the optimized ONNX model
    """
    try:
        print("Testing optimized ONNX model...")
        session = ort.InferenceSession("frontend/best_optimized.onnx")
        
        # Test with different input sizes
        test_input = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
        
        outputs = session.run(None, {'raw_images': test_input})
        print(f"✅ Output shape: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing optimized model: {e}")
        return False

if __name__ == "__main__":
    print("Optimizing ONNX model for variable input sizes...")
    
    input_model = "frontend/best.onnx"
    output_model = "frontend/best_optimized.onnx"
    
    if os.path.exists(input_model):
        if add_preprocessing_to_onnx(input_model, output_model):
            test_optimized_model()
    else:
        print(f"❌ Input model not found: {input_model}")
