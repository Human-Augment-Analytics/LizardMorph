import xml.etree.ElementTree as ET
import numpy as np
# matplotlib.pyplot lazy-loaded inside functions
import cv2
import os
import glob
from PIL import Image, ImageEnhance

def _init_matplotlib():
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    return plt

def parse_xml_for_frontend(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    all_data = []
    
    for image in root.findall('.//image'):
        image_file = image.get('file')
        image_name = os.path.basename(image_file)
        coords = []
        bounding_boxes = []

        # Extract bounding boxes from box elements
        for box in image.findall('.//box'):
            parts = box.findall('.//part')
            part_ids = [int(p.get('name', -1)) for p in parts]
            
            # Determine label from XML attribute or infer from structure
            label = box.get('label', '')
            if not label:
                if 17 in part_ids or 18 in part_ids:
                    label = 'ruler'
                elif len(parts) == 9:
                    label = 'toe/finger'
                elif len(parts) == 2 and 17 not in part_ids:
                    label = 'scale'
                elif len(parts) == 0:
                    label = 'id'
                else:
                    label = 'unknown'
            
            box_data = {
                "top": float(box.get('top', 0)),
                "left": float(box.get('left', 0)),
                "width": float(box.get('width', 0)),
                "height": float(box.get('height', 0)),
                "label": label
            }
            bounding_boxes.append(box_data)
            
            box_idx = len(bounding_boxes) - 1
            
            # Extract parts (landmarks) within this box
            for part in box.findall('.//part'):
                x = float(part.get('x'))
                y = float(part.get('y'))
                landmark_id = int(part.get('name', 0))
                # Create globally unique ID for D3 rendering to prevent overlap bugs
                unique_id = (box_idx * 100) + landmark_id
                coords.append({"id": unique_id, "x": x, "y": y, "box_idx": box_idx, "landmark_id": landmark_id})
        # Add this image data to the list
        all_data.append({
            'name': image_name, 
            "coords": coords,
            "bounding_boxes": bounding_boxes
        })
    
    # For backward compatibility, return only the first image data
    # Frontend expects an object, not an array
    if all_data:
        return all_data[0]
    return {'name': '', 'coords': [], 'bounding_boxes': []}


def read_tps_file(file_path):
    """Read a TPS file and return data as a list of x and y coordinate lists with their corresponding images."""
    data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('LM='):
            num_points = int(line.split('=')[1])
            x_coords = []
            y_coords = []
            
            for j in range(num_points):
                i += 1
                point_data = lines[i].strip().split()
                x_coords.append(float(point_data[0]))
                y_coords.append(float(point_data[1]))
            
            # Find IMAGE name
            while i < len(lines) and not lines[i].strip().startswith('IMAGE='):
                i += 1
            
            if i < len(lines):
                image_name = lines[i].strip().split('=')[1]
                data.append((image_name, x_coords, y_coords))
        i += 1
    
    return data

SOURCE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def resolve_source_image(image_name, source_folder):
    """Locate the source image a TPS ``IMAGE=`` entry names, whatever its extension."""
    if os.path.dirname(image_name) and os.path.isfile(image_name):
        return image_name

    if not source_folder:
        return None

    basename = os.path.basename(image_name)
    direct = os.path.join(source_folder, basename)
    if os.path.isfile(direct):
        return direct

    stem, ext = os.path.splitext(basename)
    wanted = {basename.lower()}
    if ext.lower() in SOURCE_IMAGE_EXTENSIONS:
        wanted.add(stem.lower())

    try:
        entries = sorted(os.listdir(source_folder))
    except OSError:
        return None

    for entry in entries:
        entry_stem, entry_ext = os.path.splitext(entry)
        if entry_stem.lower() in wanted and entry_ext.lower() in SOURCE_IMAGE_EXTENSIONS:
            candidate = os.path.join(source_folder, entry)
            if os.path.isfile(candidate):
                return candidate

    return None


def create_image(tps_file_path, output_folder, source_folder):
    """Create annotated images based on TPS file data."""
    plot_data = read_tps_file(tps_file_path)
    output_image_paths = []
    
    print(f"TPS file: {tps_file_path}")
    print(f"Found {len(plot_data)} datasets in TPS file")
    
    for i, (image_name, x_coords, y_coords) in enumerate(plot_data):
        try:
            image_path = resolve_source_image(image_name, source_folder)

            if not image_path:
                print(f"Warning: Image file not found for: {image_name}")
                continue

            print(f"Loading image: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
            height_pixels = image.shape[0]
            
            # TPS stores Y with 0 at bottom; flip Y for image coordinates
            y_coords_img = [height_pixels - y for y in y_coords]
            
            for x, y in zip(x_coords, y_coords_img):
                pt = (int(round(x)), int(round(y)))
                cv2.circle(image, pt, 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(image, pt, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            
            output_basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_folder, f"annotated_{output_basename}.png")
            
            print(f"Saving annotated image to: {output_path}")
            cv2.imwrite(output_path, image)
            output_image_paths.append(output_path)
            
        except Exception as e:
            print(f"Error creating annotated image for {image_name}: {str(e)}")
    
    return output_image_paths

def invert_single_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is not None:
        image = 255 - image
        cv2.imwrite(output_path, image)
    else:
        print(f"Warning: Could not load image for inversion: {input_path}")

def enhance_image(img, sharpness=4, contrast=1.3, blur=3):
    """Enhance image sharpness, contrast, and blur."""
    # Convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL Image
    pil_img = Image.fromarray(img)

    # Enhance the sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    img_enhanced = enhancer.enhance(sharpness)

    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(contrast)

    # Convert back to OpenCV image (numpy array)
    img_enhanced = np.array(img_enhanced)

    # Apply a small amount of Gaussian blur
    img_enhanced = cv2.GaussianBlur(img_enhanced, (blur, blur), 0)

    return cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_single_image(input_path, output_path, sharpness=4, contrast=1.3, blur=3, clip_limit=2.0, 
                       tile_grid_size=(8, 8), gamma=1.0):
    """Process a single image with all enhancements"""
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    image = enhance_image(image, sharpness, contrast, blur)
    image = clahe(image, clip_limit, tile_grid_size)
    image = gamma_correction(image, gamma)
    cv2.imwrite(output_path, image)