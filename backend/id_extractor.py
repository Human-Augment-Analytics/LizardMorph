import cv2
import easyocr
import numpy as np
import re
import math

reader = easyocr.Reader(["en"], gpu=False)

def crop_from_yolo_box(image, x_center, y_center, box_width, box_height, enhance=True, target_size=(128, 64)):
    """Convert YOLO coords to pixel coords and crop image with padding."""
    
    h, w = image.shape[:2]
    
    # YOLO coordinates are normalized (0-1), convert to pixels
    x_center_px = x_center * w
    y_center_px = y_center * h
    box_w_px = box_width * w
    box_h_px = box_height * h
    
    # Get top-left and bottom-right coordinates
    x_min = int(x_center_px - box_w_px / 2)
    x_max = int(x_center_px + box_w_px / 2)
    y_min = int(y_center_px - box_h_px / 2)
    y_max = int(y_center_px + box_h_px / 2)

    # Padding
    padding_ratio_x = -0.01
    padding_ratio_y = -0.01

    pad_x = box_w_px * padding_ratio_x
    pad_y = box_h_px * padding_ratio_y
    
    x_min = max(0, int(x_min - pad_x))
    x_max = min(w, int(x_max + pad_x))
    y_min = max(0, int(y_min - pad_y))
    y_max = min(h, int(y_max + pad_y))

    # Shrink bottom by 10% of box height
    shrink_ratio = 0.10
    y_max = int(y_max - box_h_px * shrink_ratio)

    # CROP FIRST, then process (much faster than processing full image)
    crop = image[y_min:y_max, x_min:x_max]

    if crop.size == 0:
        return None

    # Resize while maintaining aspect ratio
    target_w, target_h = target_size
    cropped_h, cropped_w = crop.shape[:2]
    
    if cropped_h == 0 or cropped_w == 0:
         return None

    scale_w = target_w / cropped_w
    scale_h = target_h / cropped_h
    scale = min(scale_w, scale_h)

    new_w = int(cropped_w * scale)
    new_h = int(cropped_h * scale)
    
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    crop = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[1, 1, 1])

    if enhance and crop.size > 0:
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Increase contrast and brightness
        alpha = 2.0  # Contrast control (1.0–3.0)
        beta = 10    # Brightness control (0–100)
        enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # Apply adaptive threshold to make digits stand out
        thresh = cv2.adaptiveThreshold(enhanced, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 27, 10)
        crop = thresh

    return crop

def detect_digits(image, conf_threshold=0.5):
    """Run EasyOCR on the image to extract digits."""
    
    best_text = ""
    best_conf = 0.0
    
    # Run EasyOCR once on the whole image (much faster than per-contour)
    results_0 = reader.readtext(image, detail=1, allowlist='0123456789')
    
    for (bbox, text, conf) in results_0:
        text = re.sub(r"\D", "", text)
        if not text:
            continue
        candidates = [text[i:i + 4] for i in range(len(text) - 3)] if len(text) > 4 else [text]

        valid_candidates = [c for c in candidates if c.isdigit() and 0 <= int(c) <= 2000]
        for c in valid_candidates[::-1]:
             if 3 <= len(c) <= 4 and conf > best_conf:
                best_conf = conf
                best_text = c

    # Only try 180° rotation if confidence is below threshold
    if best_conf < conf_threshold:
        rotated_img = cv2.rotate(image, cv2.ROTATE_180)
        results_180 = reader.readtext(rotated_img, detail=1, allowlist="0123456789")
        for (bbox, text, conf) in results_180:
            text = re.sub(r"\D", "", text)
            if not text:
                continue
            candidates = [text[i:i + 4] for i in range(len(text) - 3)] if len(text) > 4 else [text]
            valid_candidates = [c for c in candidates if c.isdigit() and 0 <= int(c) <= 2000]
            for c in valid_candidates:
                if 3 <= len(c) <= 4 and conf > best_conf:
                    best_conf = conf
                    best_text = c
    
    return best_text, best_conf

def extract_id_from_image(image_path, box_norm):
    """
    Main function to extract ID from an image given a YOLO bounding box.
    
    Args:
        image_path (str): Path to image file.
        box_norm (list/tuple): [x_center, y_center, width, height] normalized.
        
    Returns:
        dict: {'id': str, 'confidence': float}
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Failed to load image'}
            
        x_center, y_center, width, height = box_norm
        
        # Crop
        processed_crop = crop_from_yolo_box(image, x_center, y_center, width, height)
        
        if processed_crop is None:
             return {'error': 'Failed to crop image'}

        # Process to get ID
        text, conf = detect_digits(processed_crop)
        
        return {
            'id': text,
            'confidence': float(conf)
        }
        
    except Exception as e:
        return {'error': str(e)}
