import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from PIL import Image, ImageEnhance

# Use a non-GUI backend
plt.switch_backend('Agg')

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
            box_data = {
                "top": float(box.get('top', 0)),
                "left": float(box.get('left', 0)),
                "width": float(box.get('width', 0)),
                "height": float(box.get('height', 0))
            }
            bounding_boxes.append(box_data)
            
            # Extract parts (landmarks) within this box
            for part in box.findall('.//part'):
                x = float(part.get('x'))
                y = float(part.get('y'))
                coords.append({"x": x, "y": y})
        
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
    current_lm = 0
    current_x = []
    current_y = []
    current_image = None
    reading_points = False

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('LM='):
                current_x = []  # Reset coordinates for the new entry
                current_y = []
                current_lm = int(line[3:])
                reading_points = True
                i += 1
                # Read the next current_lm lines as coordinates
                for j in range(current_lm):
                    if i + j < len(lines):
                        point_line = lines[i + j].strip()
                        # Skip if it's not a coordinate line
                        if point_line.startswith('IMAGE=') or point_line.startswith('ID='):
                            break
                        try:
                            parts = point_line.split()
                            if len(parts) >= 2:  # Ensure we have at least x and y
                                current_x.append(float(parts[0]))
                                current_y.append(float(parts[1]))
                        except Exception as e:
                            print(f"Error parsing point line: {point_line}, error: {str(e)}")
                i += current_lm - 1  # Skip the coordinates we just read
                reading_points = False
            elif line.startswith('IMAGE='):
                current_image = line[6:]
                # After reading image, add data if we have points
                if current_x and current_y:
                    data.append((current_image, current_x, current_y))
                    # Reset for next image
                    current_x = []
                    current_y = []
            i += 1
    
    return data

def create_image(tps_file_path, output_folder):
    """Create annotated images based on TPS file data."""
    plot_data = read_tps_file(tps_file_path)
    output_image_paths = []
    
    print(f"TPS file: {tps_file_path}")
    print(f"Found {len(plot_data)} datasets in TPS file")
    
    for i, (image_name, x_coords, y_coords) in enumerate(plot_data):
        try:
            # Ensure image name is correctly handled
            if image_name.endswith('.jpg'):
                image_path = os.path.join("upload", image_name)
            else:
                image_path = os.path.join("upload", f"{image_name}.jpg")
            
            print(f"Loading image: {image_path}")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
                
            image = plt.imread(image_path)
            height_pixels, width_pixels = image.shape[0], image.shape[1]
            
            dpi = 100  # Use a reasonable DPI value
            width_inches = width_pixels / dpi
            height_inches = height_pixels / dpi
            
            fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=dpi)
            ax.imshow(image)
            ax.axis('off')
            
            print(f"Plotting {len(x_coords)} points")
            
            # Plot points
            ax.scatter(x_coords, y_coords, s=50, color='red', marker='o', edgecolors='black')
            
            # Add point labels
            for j, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax.text(x + 5, y - 5, str(j + 1), color='white', fontsize=8, 
                       bbox=dict(facecolor='black', alpha=0.7, pad=1))
            
            # Save the image
            output_basename = os.path.splitext(os.path.basename(image_name))[0]
            output_path = os.path.join(output_folder, f"annotated_{output_basename}.jpg")
            
            print(f"Saving annotated image to: {output_path}")
            
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig)
            
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