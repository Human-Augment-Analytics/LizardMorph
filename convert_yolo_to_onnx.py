#!/usr/bin/env python3
"""
Create a coordinate conversion utility to match YOLO preprocessing with ONNX letterbox
"""

import numpy as np
import json
from PIL import Image
import os

def create_coordinate_converter():
    """
    Create a coordinate conversion utility that maps ONNX letterbox coordinates
    to YOLO direct resize coordinates
    """
    
    def convert_onnx_to_yolo_coords(onnx_bbox, original_width, original_height, 
                                   onnx_input_size=640, yolo_target_size=1024):
        """
        Convert ONNX letterbox coordinates to YOLO direct resize coordinates
        
        Args:
            onnx_bbox: [x1, y1, x2, y2] in ONNX letterbox coordinate system
            original_width, original_height: Original image dimensions
            onnx_input_size: ONNX model input size (640)
            yolo_target_size: YOLO target size (1024)
        
        Returns:
            [x1, y1, x2, y2] in YOLO coordinate system
        """
        x1, y1, x2, y2 = onnx_bbox
        
        # Step 1: Convert from ONNX letterbox to original image coordinates
        # Calculate ONNX letterbox parameters
        onnx_scale = min(onnx_input_size / original_width, onnx_input_size / original_height)
        onnx_scaled_width = original_width * onnx_scale
        onnx_scaled_height = original_height * onnx_scale
        onnx_pad_x = (onnx_input_size - onnx_scaled_width) / 2
        onnx_pad_y = (onnx_input_size - onnx_scaled_height) / 2
        
        # Convert from letterbox coordinates to original image coordinates
        orig_x1 = (x1 - onnx_pad_x) / onnx_scale
        orig_y1 = (y1 - onnx_pad_y) / onnx_scale
        orig_x2 = (x2 - onnx_pad_x) / onnx_scale
        orig_y2 = (y2 - onnx_pad_y) / onnx_scale
        
        # Step 2: Convert from original coordinates to YOLO direct resize coordinates
        # Calculate YOLO direct resize parameters
        yolo_scale = min(yolo_target_size / original_width, yolo_target_size / original_height)
        yolo_new_width = int(original_width * yolo_scale)
        yolo_new_height = int(original_height * yolo_scale)
        
        # Convert to YOLO coordinate system
        yolo_x1 = orig_x1 * yolo_scale
        yolo_y1 = orig_y1 * yolo_scale
        yolo_x2 = orig_x2 * yolo_scale
        yolo_y2 = orig_y2 * yolo_scale
        
        return [yolo_x1, yolo_y1, yolo_x2, yolo_y2]
    
    return convert_onnx_to_yolo_coords

def create_coordinate_conversion_script():
    """
    Create a JavaScript/TypeScript coordinate conversion function
    """
    js_converter = """
// Coordinate conversion utility for ONNX to YOLO preprocessing
export function convertOnnxToYoloCoords(
    onnxBbox: number[],
    originalWidth: number,
    originalHeight: number,
    onnxInputSize: number = 640,
    yoloTargetSize: number = 1024
): number[] {
    const [x1, y1, x2, y2] = onnxBbox;
    
    // Step 1: Convert from ONNX letterbox to original image coordinates
    const onnxScale = Math.min(onnxInputSize / originalWidth, onnxInputSize / originalHeight);
    const onnxScaledWidth = originalWidth * onnxScale;
    const onnxScaledHeight = originalHeight * onnxScale;
    const onnxPadX = (onnxInputSize - onnxScaledWidth) / 2;
    const onnxPadY = (onnxInputSize - onnxScaledHeight) / 2;
    
    const origX1 = (x1 - onnxPadX) / onnxScale;
    const origY1 = (y1 - onnxPadY) / onnxScale;
    const origX2 = (x2 - onnxPadX) / onnxScale;
    const origY2 = (y2 - onnxPadY) / onnxScale;
    
    // Step 2: Convert from original coordinates to YOLO direct resize coordinates
    const yoloScale = Math.min(yoloTargetSize / originalWidth, yoloTargetSize / originalHeight);
    
    const yoloX1 = origX1 * yoloScale;
    const yoloY1 = origY1 * yoloScale;
    const yoloX2 = origX2 * yoloScale;
    const yoloY2 = origY2 * yoloScale;
    
    return [yoloX1, yoloY1, yoloX2, yoloY2];
}
"""
    
    # Save the JavaScript converter
    with open("frontend/src/utils/coordinateConverter.ts", "w") as f:
        f.write(js_converter)
    
    print("✅ Created coordinate conversion utility: frontend/src/utils/coordinateConverter.ts")
    return True

def test_coordinate_conversion():
    """
    Test the coordinate conversion with sample data
    """
    print("Testing coordinate conversion...")
    
    # Sample test case
    original_width, original_height = 2048, 1024
    onnx_bbox = [100, 200, 300, 400]  # ONNX letterbox coordinates
    onnx_input_size = 640
    yolo_target_size = 1024
    
    converter = create_coordinate_converter()
    yolo_bbox = converter(onnx_bbox, original_width, original_height, 
                         onnx_input_size, yolo_target_size)
    
    print(f"Original image: {original_width}x{original_height}")
    print(f"ONNX bbox (letterbox): {onnx_bbox}")
    print(f"YOLO bbox (direct resize): {[round(x, 2) for x in yolo_bbox]}")
    
    return True

if __name__ == "__main__":
    print("Creating coordinate conversion utility...")
    
    if create_coordinate_conversion_script():
        test_coordinate_conversion()
        print("✅ Coordinate conversion utility created successfully!")
    else:
        print("❌ Failed to create coordinate conversion utility!")
