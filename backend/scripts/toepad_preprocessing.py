"""
Lizard Toepad preprocessing module.
Uses YOLO for bounding box detection and dlib for landmark prediction.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import dlib
import os


def detect_bounding_box_yolo(image_path, yolo_model_path, conf_threshold=0.25):
    """
    Detect bounding boxes using YOLO model.
    
    Parameters:
    ----------
        image_path (str): Path to the input image
        yolo_model_path (str): Path to the YOLO model file (.pt)
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
    ----------
        list: List of detected bounding boxes as dlib rectangles, or None if no detections
    """
    if not os.path.exists(yolo_model_path):
        print(f"YOLO model not found at {yolo_model_path}")
        return None
    
    try:
        # Load YOLO model
        model = YOLO(yolo_model_path)
        
        # Run inference on CPU
        results = model(image_path, conf=conf_threshold, device='cpu')
        
        # Extract bounding boxes
        boxes = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Convert to dlib rectangle format (left, top, right, bottom)
                    dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                    boxes.append(dlib_rect)
        
        return boxes if boxes else None
        
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return None


def process_single_image(input_path, output_path, sharpness=4, contrast=1.3, blur=3, 
                        clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.0):
    """
    Process a single image with enhancements (similar to xray_preprocessing).
    This is optional preprocessing that can help with detection.
    
    Parameters:
    ----------
        input_path (str): Path to input image
        output_path (str): Path to save processed image
        sharpness (float): Sharpness enhancement level
        contrast (float): Contrast enhancement level
        blur (int): Gaussian blur kernel size
        clip_limit (float): CLAHE clip limit
        tile_grid_size (tuple): CLAHE tile grid size
        gamma (float): Gamma correction value
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for enhancement
    pil_img = Image.fromarray(image)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    img_enhanced = enhancer.enhance(sharpness)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(contrast)
    
    # Convert back to numpy array
    img_enhanced = np.array(img_enhanced)
    
    # Apply Gaussian blur
    img_enhanced = cv2.GaussianBlur(img_enhanced, (blur, blur), 0)
    
    # Apply CLAHE
    img_yuv = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_output = cv2.LUT(img_output, table)
    
    # Convert back to BGR for saving
    img_output = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_output)

