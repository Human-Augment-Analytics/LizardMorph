#!/usr/bin/env python3
"""
Lizard Toepad Detection Script

This script uses the YOLO model from frontend/best.pt to detect lizard toepads
and draws bounding boxes using the same approach as the Lizard_Toepads repository.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
from ultralytics import YOLO
import yaml
from pathlib import Path

# Increase maximum image size limit
Image.MAX_IMAGE_PIXELS = None

def preprocess_image_for_inference(img_path, target_size=1024):
    """
    Preprocess image to match the training pipeline from Lizard_Toepads.
    This ensures the model sees the same type of input it was trained on.
    """
    with Image.open(img_path) as img:
        print(f"Original image: {img.size}, mode: {img.mode}")

        # Step 1: resize with same method as training
        original_width, original_height = img.size
        scale = min(target_size / original_width, target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # resize with same resampling method as training
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Step 2: convert to grayscale (matching training preprocessing)
        if resized_img.mode != 'L':
            resized_img = resized_img.convert('L')

        # Step 3: convert back to RGB for YOLO inference (3 channels expected)
        resized_img = resized_img.convert('RGB')

        print(f"Processed image: {resized_img.size}, mode: {resized_img.mode}")
        return resized_img, new_width, new_height, scale

def draw_bounding_boxes(image, results, class_names, confidence_threshold=0.25):
    """
    Draw bounding boxes on the image using the same approach as Lizard_Toepads.
    
    Args:
        image: PIL Image object
        results: YOLO detection results
        class_names: List of class names ['finger', 'toe', 'ruler']
        confidence_threshold: Minimum confidence for drawing boxes
    
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Create a copy for drawing
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to load a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Color mapping for different classes (matching Lizard_Toepads)
    class_colors = {
        'finger': 'blue',
        'toe': 'red', 
        'ruler': 'purple'
    }
    
    total_detections = 0
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Draw bounding box with class-specific color
                    color = class_colors.get(class_name, 'green')
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # Draw class label with confidence
                    label = f"{class_name}: {confidence:.3f}"
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Position label above the bounding box
                    label_y = max(0, y1 - text_height - 5)
                    draw.rectangle([x1, label_y, x1 + text_width + 4, label_y + text_height + 2], 
                                 fill=color, outline=color)
                    draw.text((x1 + 2, label_y + 1), label, fill='white', font=font)
                    
                    total_detections += 1
                    print(f"Detected {class_name}: confidence={confidence:.3f}, "
                          f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    
    print(f"Total detections above threshold: {total_detections}")
    return img_with_boxes, total_detections

def run_detection(model_path, image_path, output_path=None, confidence_threshold=0.25, 
                 image_size=1024, save_results=True):
    """
    Run lizard toepad detection on an image using the trained YOLO model.
    
    Args:
        model_path: Path to the YOLO model (.pt file)
        image_path: Path to the input image
        output_path: Path to save the result (optional)
        confidence_threshold: Minimum confidence for detections
        image_size: Image size for inference
        save_results: Whether to save the results
    
    Returns:
        PIL Image with bounding boxes drawn
    """
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"Processing image: {image_path}")
    
    # Preprocess image (resize and convert to grayscale then back to RGB)
    processed_img, new_width, new_height, scale = preprocess_image_for_inference(image_path, image_size)
    
    # Run inference
    print("Running inference...")
    results = model.predict(
        source=processed_img,
        conf=confidence_threshold,
        imgsz=image_size,
        verbose=False
    )
    
    # Define class names (matching Lizard_Toepads configuration)
    class_names = ['finger', 'toe', 'ruler']
    
    # Draw bounding boxes
    print("Drawing bounding boxes...")
    img_with_boxes, num_detections = draw_bounding_boxes(
        processed_img, results, class_names, confidence_threshold
    )
    
    # Save results if requested
    if save_results and output_path:
        img_with_boxes.save(output_path)
        print(f"Results saved to: {output_path}")
    
    return img_with_boxes, num_detections, results

def main():
    parser = argparse.ArgumentParser(description='Detect lizard toepads using YOLO model')
    parser.add_argument('--model', default='frontend/best.pt', 
                       help='Path to YOLO model file (default: frontend/best.pt)')
    parser.add_argument('--image', required=True, 
                       help='Path to input image')
    parser.add_argument('--output', 
                       help='Path to save output image (default: input_image_with_boxes.jpg)')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--imgsz', type=int, default=1024, 
                       help='Image size for inference (default: 1024)')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save results')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Set output path if not provided
    if args.output is None and not args.no_save:
        base_name = os.path.splitext(args.image)[0]
        args.output = f"{base_name}_with_boxes.jpg"
    
    # Run detection
    try:
        result_img, num_detections, results = run_detection(
            model_path=args.model,
            image_path=args.image,
            output_path=args.output,
            confidence_threshold=args.conf,
            image_size=args.imgsz,
            save_results=not args.no_save
        )
        
        print(f"\n=== Detection Summary ===")
        print(f"Image processed: {args.image}")
        print(f"Total detections: {num_detections}")
        print(f"Confidence threshold: {args.conf}")
        print(f"Image size: {args.imgsz}")
        
        if args.output:
            print(f"Output saved to: {args.output}")
            
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

