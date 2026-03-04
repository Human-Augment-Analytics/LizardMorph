import os
import xml.etree.ElementTree as ET
import argparse

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

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

    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX models")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--pt_model", default="../models/lizard-toe-pad/yolo_obb_6class_h7.pt")
    parser.add_argument("--onnx_model", default="../models/lizard-toe-pad/yolo_obb_6class_h7.onnx")
    parser.add_argument("--toe_pred", default="../models/lizard-toe-pad/toe_predictor_obb.dat")
    parser.add_argument("--finger_pred", default="../models/lizard-toe-pad/finger_predictor_obb.dat")
    args = parser.parse_args()
    
    image_path = args.image
    pt_model_path = args.pt_model
    onnx_model_path = args.onnx_model
    
    toe_predictor = args.toe_pred
    finger_predictor = args.finger_pred
    
    pt_xml = "output_pt.xml"
    onnx_xml = "output_onnx.xml"
    
    print(f"Running PyTorch model ({pt_model_path})...")
    utils.predictions_to_xml_single_with_yolo(
        image_path, pt_xml, pt_model_path,
        toe_predictor_path=toe_predictor,
        finger_predictor_path=finger_predictor,
        target_predictor_type='toe'
    )
    
    print(f"\nRunning ONNX model ({onnx_model_path})...")
    utils.predictions_to_xml_single_with_yolo(
        image_path, onnx_xml, onnx_model_path,
        toe_predictor_path=toe_predictor,
        finger_predictor_path=finger_predictor,
        target_predictor_type='toe'
    )
    
    print("\n--- COMPARISON ---")
    pt_data = parse_xml_to_dict(pt_xml)
    onnx_data = parse_xml_to_dict(onnx_xml)
    
    pt_count = sum(len(v) for v in pt_data.values())
    onnx_count = sum(len(v) for v in onnx_data.values())
    
    print(f"PyTorch objects detected: {pt_count}")
    print(f"ONNX objects detected: {onnx_count}")
    
    all_labels = set(pt_data.keys()).union(set(onnx_data.keys()))
    for label in sorted(all_labels):
        pt_objs = pt_data.get(label, [])
        onnx_objs = onnx_data.get(label, [])
        print(f"  {label}: PyTorch={len(pt_objs)}, ONNX={len(onnx_objs)}")
        
        # Compare parts if same count
        if len(pt_objs) > 0 and len(pt_objs) == len(onnx_objs):
            for i in range(len(pt_objs)):
                pt_parts = pt_objs[i]['parts']
                onnx_parts = onnx_objs[i]['parts']
                
                if pt_parts and onnx_parts:
                    diffs = []
                    for p1, p2 in zip(pt_parts, onnx_parts):
                        diffs.append(((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)**0.5)
                    
                    if diffs:
                        avg_diff = sum(diffs) / len(diffs)
                        max_diff = max(diffs)
                        print(f"    {label} [{i}] Average Landmark Diff: {avg_diff:.2f}px, Max Diff: {max_diff:.2f}px")
                else:
                    # Compare bounding box
                    p1 = pt_objs[i]['box']
                    p2 = onnx_objs[i]['box']
                    diff = ((p1['left'] - p2['left'])**2 + (p1['top'] - p2['top'])**2 + (p1['width'] - p2['width'])**2 + (p1['height'] - p2['height'])**2)**0.5
                    print(f"    {label} [{i}] Box Diff: {diff:.2f}")

    pt_tps = "output_pt.tps"
    onnx_tps = "output_onnx.tps"
    
    utils.dlib_xml_to_tps(pt_xml)
    utils.dlib_xml_to_tps(onnx_xml)
    
    # Check that they exist
    import os
    if os.path.exists("output_pt.xml.tps"):
        os.rename("output_pt.xml.tps", pt_tps)
    if os.path.exists("output_onnx.xml.tps"):
        os.rename("output_onnx.xml.tps", onnx_tps)
        
    print("\n--- TPS COMPARISON ---")
    with open(pt_tps, 'r') as f:
        pt_lines = f.readlines()
    with open(onnx_tps, 'r') as f:
        onnx_lines = f.readlines()
        
    diff_count = 0
    max_tps_diff = 0
    tps_diffs = []
    
    # Naive line-by-line numerical comparison
    for p_line, o_line in zip(pt_lines, onnx_lines):
        if p_line != o_line:
            # Check if this is a coordinate line
            p_parts = p_line.strip().split()
            o_parts = o_line.strip().split()
            
            if len(p_parts) == 2 and len(o_parts) == 2:
                try:
                    px, py = float(p_parts[0]), float(p_parts[1])
                    ox, oy = float(o_parts[0]), float(o_parts[1])
                    
                    diff = ((px - ox)**2 + (py - oy)**2)**0.5
                    tps_diffs.append(diff)
                    if diff > max_tps_diff:
                        max_tps_diff = diff
                    diff_count += 1
                except ValueError:
                    pass

    if tps_diffs:
        avg_diff = sum(tps_diffs) / len(tps_diffs)
        print(f"Total TPS points differing: {diff_count}")
        print(f"Average point offset in TPS format: {avg_diff:.2f}px")
        print(f"Max point offset in TPS format: {max_tps_diff:.2f}px")
    else:
        print("TPS exports completely perfectly identical.")
        
    print("\n--- MILLIMETER TRANSLATION ---")
    
    # Try to calculate pixels per mm from ruler
    def get_ruler_distance(data):
        for label, objs in data.items():
            for obj in objs:
                parts = obj.get('parts', [])
                if len(parts) == 2:
                    p1 = parts[0]
                    p2 = parts[1]
                    dist = ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)**0.5
                    return dist
        return None
        
    pt_ruler_px = get_ruler_distance(pt_data)
    onnx_ruler_px = get_ruler_distance(onnx_data)
    
    if pt_ruler_px and onnx_ruler_px:
        # Assuming ruler is 10mm like in ml-morph logic
        pt_mm_per_px = 10.0 / pt_ruler_px
        onnx_mm_per_px = 10.0 / onnx_ruler_px
        
        print(f"PyTorch Scale: {pt_ruler_px:.2f}px = 10mm (1px = {pt_mm_per_px:.4f}mm)")
        print(f"ONNX Scale: {onnx_ruler_px:.2f}px = 10mm (1px = {onnx_mm_per_px:.4f}mm)")
        
        avg_diff_mm = avg_diff * pt_mm_per_px
        max_diff_mm = max_tps_diff * pt_mm_per_px
        
        print(f"\nAverage TPS Error in MM: {avg_diff_mm:.4f} mm")
        print(f"Maximum TPS Error in MM: {max_diff_mm:.4f} mm")
    else:
        print("Could not find a 2-point ruler structure in the XML data to convert pixels to mm.")
        
if __name__ == "__main__":
    main()
