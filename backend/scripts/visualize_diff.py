import sys
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def parse_xml_to_dict(xml_path):
    if not os.path.exists(xml_path):
        return {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = {}
    for image in root.findall('.//image'):
        for box in image.findall('box'):
            label = box.get('label', 'unknown')
            if label not in data:
                data[label] = []
            
            box_data = {
                 'box': {
                     'top': int(box.get('top', 0)),
                     'left': int(box.get('left', 0)),
                     'width': int(box.get('width', 0)),
                     'height': int(box.get('height', 0))
                 },
                 'parts': []
            }
            
            for part in box.findall('part'):
                box_data['parts'].append({
                    'name': part.get('name', ''),
                    'x': int(part.get('x', 0)),
                    'y': int(part.get('y', 0))
                })
            
            data[label].append(box_data)
    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_diff.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    pt_xml = "output_pt.xml"
    onnx_xml = "output_onnx.xml"
    
    pt_data = parse_xml_to_dict(pt_xml)
    onnx_data = parse_xml_to_dict(onnx_xml)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)
        
    # Draw ONNX logic (Red)
    for label, objs in onnx_data.items():
        if label == "unknown": # Fingers/Toes
            for obj in objs:
                parts = obj.get('parts', [])
                for part in parts:
                    x, y = part['x'], part['y']
                    cv2.circle(img, (int(x), int(y)), 10, (0, 0, 255), -1) # Red for ONNX
                    
    # Draw PT logic (Green)
    for label, objs in pt_data.items():
        if label == "unknown": # Fingers/Toes
            for obj in objs:
                parts = obj.get('parts', [])
                for part in parts:
                    x, y = part['x'], part['y']
                    cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1) # Green for PT
                    
    # Add legend
    cv2.putText(img, "ONNX Points (Red radius 10)", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    cv2.putText(img, "PyTorch Points (Green radius 6)", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

    # Crop to the second unknown object (Landmark 2 from previous output)
    crop_img = img
    if "unknown" in pt_data and len(pt_data["unknown"]) > 1:
        target_obj = pt_data["unknown"][1]
        box = target_obj['box']
        
        # Add a 800px padding around the bounding box
        padding = 800
        y1 = max(0, box['top'] - padding)
        y2 = min(img.shape[0], box['top'] + box['height'] + padding)
        x1 = max(0, box['left'] - padding)
        x2 = min(img.shape[1], box['left'] + box['width'] + padding)
        
        crop_img = img[y1:y2, x1:x2]

    output_path = "landmark_comparison.jpg"
    cv2.imwrite(output_path, crop_img)
    print(f"Comparison image saved to {os.path.abspath(output_path)}")
    
if __name__ == "__main__":
    main()
