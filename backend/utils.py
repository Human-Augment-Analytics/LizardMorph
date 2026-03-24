# Part of the standard library
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import re
import glob
import ntpath

# Not part of the standard library
import numpy as np
import pandas as pd
import cv2
try:
    import dlib
except ImportError:
    dlib = None
    print("Warning: dlib not found, some features will be unavailable")
import os
import shutil
import random


# Dorsal landmark reorder mapping.
# The dlib predictor (new_landmarks_2025_predictor.dat) outputs 34 parts in its
# own internal order.  The reference dorsal annotations use a different order
# (1‑34, 0‑indexed below).  This array maps each reference position to the dlib
# part index that should fill it.
#   reference_landmark[i] = dlib_part[ DORSAL_LANDMARK_ORDER[i] ]
DORSAL_LANDMARK_ORDER = [
    0, 1, 12, 23, 28, 29, 30, 31, 32, 33,   # ref 1‑10
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11,          # ref 11‑20
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22,  # ref 21‑30
    24, 25, 26, 27,                            # ref 31‑34
]


# Tools for predicting objects and shapes in new images


def load_dlib_detector(detector_path):
    """
    Load a dlib fhog object detector from file.
    
    Parameters:
    ----------
        detector_path (str): Path to the detector file
        
    Returns:
    ----------
        detector: dlib.fhog_object_detector object or None if file doesn't exist
    """
    if detector_path and os.path.exists(detector_path):
        try:
            return dlib.fhog_object_detector(detector_path)
        except Exception as e:
            print(f"Error loading detector from {detector_path}: {e}")
            return None
    return None


def detect_bounding_box_dlib(image, detector, upsample_num_times=0, adjust_threshold=0):
    """
    Detect bounding boxes using dlib fhog object detector.
    
    Parameters:
    ----------
        image: RGB image (numpy array)
        detector: dlib.fhog_object_detector object
        upsample_num_times (int): Number of times to upsample the image
        adjust_threshold (float): Threshold adjustment for detection
        
    Returns:
    ----------
        dlib.rectangle: Detected bounding box or None if no detection
    """
    if detector is None:
        return None
        
    try:
        # Run the detector
        [boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run(
            detector, image, upsample_num_times=upsample_num_times, adjust_threshold=adjust_threshold)
        
        # Return the first detected box if any
        if len(boxes) > 0:
            return boxes[0]
        else:
            return None
    except Exception as e:
        print(f"Error during detection: {e}")
        return None


def initialize_xml():
    """
    Initializes the xml file for the predictions

    Parameters:
    ----------
        None

    Returns:
    ----------
        None (xml file written to disk)
    """
    root = ET.Element("dataset")
    root.append(ET.Element("name"))
    root.append(ET.Element("comment"))
    images_e = ET.Element("images")
    root.append(images_e)

    return root, images_e


def create_box(img_shape):
    """
    Creates a box around the image

    Parameters:
    ----------
        img_shape (tuple): shape of the image

    Returns:
    ----------
        box (Element): box element
    """
    box = ET.Element("box")
    box.set("top", str(int(1)))
    box.set("left", str(int(1)))
    box.set("width", str(int(img_shape[1] - 2)))
    box.set("height", str(int(img_shape[0] - 2)))

    return box


def create_part(x, y, id):
    """
    Creates a part element

    Parameters:
    ----------
        x (int): x coordinate of the part
        y (int): y coordinate of the part
        name (str): name of the part

    Returns:
    ----------
        part (Element): part element
    """
    part = ET.Element("part")
    part.set("name", str(int(id)))
    part.set("x", str(int(x)))
    part.set("y", str(int(y)))

    return part


def pretty_xml(elem, out):
    """
    Writes the xml file to disk

    Parameters:
    ----------
        elem (Element): root element
        out (str): name of the output file

    Returns:
    ----------
        None (xml file written to disk)
    """
    et = ET.ElementTree(elem)
    xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
    with open(out, "w") as f:
        f.write(xmlstr)


def predictions_to_xml(
    predictor_name: str, folder: str, ignore=None, output="output.xml"):
    """
    Generates dlib format xml files for model predictions. It uses previously trained models to
    identify objects in images and to predict their shape.

    Parameters:
    ----------
        predictor_name (str): shape predictor filename
        dir(str): name of the directory containing images to be predicted
        ratio (float): (optional) scaling factor for the image
        out_file (str): name of the output file (xml format)
        variance_threshold (float): threshold value to determine high variance images

    Returns:
    ----------
        None (out_file written to disk)
    """
    extensions = {".jpg", ".jpeg", ".tif", ".png", ".bmp"}
    scales = [0.25, 0.5, 1]
    files = glob.glob(f"{folder}/*")

    predictor = dlib.shape_predictor(predictor_name)

    root, images_e = initialize_xml()

    kernel = np.ones((7, 7), np.float32) / 49

    for f in sorted(files, key=str):
        error = 0
        ext = ntpath.splitext(f)[1]
        if ext.lower() in extensions:
            print(f"Processing image {f}")
            image_e = ET.Element("image")
            image_e.set("file", str(f))
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.filter2D(img, -1, kernel)
            img =  cv2.bilateralFilter(img, 9, 41, 21)
            w = img.shape[1]
            h = img.shape[0]
            landmarks = []
            for scale in scales:
                image = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                rect = dlib.rectangle(1, 1, int(w * scale) - 1, int(h * scale) - 1)
                shape = predictor(image, rect)
                landmarks.append(shape_to_np(shape) / scale)

            box = create_box(img.shape)
            part_length = range(0, shape.num_parts)
            
            for idx, i in enumerate(sorted(part_length, key=int)):
                x = np.median([landmark[idx][0] for landmark in landmarks])
                y = np.median([landmark[idx][1] for landmark in landmarks])
                if ignore is not None:
                    if i not in ignore:
                        part = create_part(x, y, i)
                        box.append(part)
                else:
                    part = create_part(x, y, i)
                    box.append(part)

                pos = np.array(landmarks)[:, idx]
                pos_x, pos_y = (
                    pos[:, 0],
                    pos[:, 1],
                )

                mean_x, mean_y = np.mean(pos_x), np.mean(pos_y)
                distances = np.sqrt((pos_x - mean_x) ** 2 + (pos_y - mean_y) ** 2)
                total_variance = np.mean(distances)
                error += total_variance
                


            box[:] = sorted(box, key=lambda child: (child.tag, float(child.get("name"))))
            image_e.append(box)
            image_e.set("error", str(error))
            images_e.append(image_e)
        
    images_e[:] = sorted(
            images_e, key=lambda child: (child.tag, float(child.get("error"))), reverse=True
    )

    pretty_xml(root, output)

def predictions_to_xml_single(predictor_name: str, image_path: str, output: str):
    """Generates dlib format xml file for a single image."""
    predictor = dlib.shape_predictor(predictor_name)
    root, images_e = initialize_xml()
    kernel = np.ones((7, 7), np.float32) / 49

    image_e = ET.Element('image')
    image_e.set('file', str(image_path))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.bilateralFilter(img, 9, 41, 21)

    scales = [0.25, 0.5, 1]
    w = img.shape[1]
    h = img.shape[0]
    landmarks = []

    for scale in scales:
        image = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        rect = dlib.rectangle(1, 1, int(w * scale) - 1, int(h * scale) - 1)
        shape = predictor(image, rect)
        landmarks.append(shape_to_np(shape) / scale)

    box = create_box(img.shape)
    num_parts = shape.num_parts

    # Reorder landmarks if this is a 34-landmark dorsal predictor
    if num_parts == len(DORSAL_LANDMARK_ORDER):
        order = DORSAL_LANDMARK_ORDER
    else:
        order = list(range(num_parts))

    for ref_idx, dlib_idx in enumerate(order):
        x = np.median([landmark[dlib_idx][0] for landmark in landmarks])
        y = np.median([landmark[dlib_idx][1] for landmark in landmarks])
        part = create_part(x, y, ref_idx)
        box.append(part)

    box[:] = sorted(box, key=lambda child: (child.tag, float(child.get("name"))))
    image_e.append(box)
    images_e.append(image_e)
    pretty_xml(root, output)


def predictions_to_xml_single_with_detector(predictor_name: str, image_path: str, output: str, detector_path: str = None):
    """
    Generates dlib format xml file for a single image using dlib fhog object detector for bounding box detection.
    This follows the approach from visualize_predictions.py.
    
    Parameters:
    ----------
        predictor_name (str): Path to the shape predictor file
        image_path (str): Path to the input image
        output (str): Path for the output XML file
        detector_path (str): Optional path to the dlib fhog object detector file
    """
    predictor = dlib.shape_predictor(predictor_name)
    root, images_e = initialize_xml()
    kernel = np.ones((7, 7), np.float32) / 49

    image_e = ET.Element('image')
    image_e.set('file', str(image_path))
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.bilateralFilter(img, 9, 41, 21)
    
    # Load detector if path is provided
    detector = None
    if detector_path:
        detector = load_dlib_detector(detector_path)
    
    scales = [0.25, 0.5, 1]
    w = img.shape[1]
    h = img.shape[0]
    landmarks = []
    
    for scale in scales:
        image = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        
        # Try to detect bounding box using detector, fallback to default rectangle
        detected_rect = None
        if detector is not None:
            detected_rect = detect_bounding_box_dlib(image, detector)
        
        if detected_rect is not None:
            # Use the detected rectangle directly (no scaling back needed since we're working at the scaled level)
            rect = detected_rect
        else:
            # Fallback to default rectangle
            rect = dlib.rectangle(1, 1, int(w * scale) - 1, int(h * scale) - 1)
        
        shape = predictor(image, rect)
        landmarks.append(shape_to_np(shape) / scale)

    box = create_box(img.shape)
    num_parts = shape.num_parts

    # Reorder landmarks if this is a 34-landmark dorsal predictor
    if num_parts == len(DORSAL_LANDMARK_ORDER):
        order = DORSAL_LANDMARK_ORDER
    else:
        order = list(range(num_parts))

    for ref_idx, dlib_idx in enumerate(order):
        x = np.median([landmark[dlib_idx][0] for landmark in landmarks])
        y = np.median([landmark[dlib_idx][1] for landmark in landmarks])
        part = create_part(x, y, ref_idx)
        box.append(part)

    box[:] = sorted(box, key=lambda child: (child.tag, float(child.get("name"))))
    image_e.append(box)
    images_e.append(image_e)
    pretty_xml(root, output)


def shape_to_np(shape):
    """
    Convert a dlib shape object to a NumPy array of (x, y)-coordinates.

    Parameters
    ----------
    shape : dlib.full_object_detection
        The dlib shape object to convert.

    Returns
    -------
    coords: np.ndarray
        A NumPy array of (x, y)-coordinates representing the landmarks in the input shape object.
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype="int")

    length = range(0, shape.num_parts)

    for i in sorted(length, key=int):
        coords[i] = [shape.part(i).x, shape.part(i).y]

    # return the list of (x, y)-coordinates
    return coords


def get_dlib_rect(corners, img_shape, padding_ratio=0.3):
    """
    Convert OBB corners (4x2 array) to an axis-aligned dlib.rectangle via cv2.boundingRect,
    with padding to match training (generate_yolo_bbox_xml.py expand_box).
    """
    x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
    img_h, img_w = img_shape[:2]
    px = int(w * padding_ratio)
    py = int(h * padding_ratio)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(img_w, x + w + px)
    y2 = min(img_h, y + h + py)
    return dlib.rectangle(int(x1), int(y1), int(x2), int(y2))


def compute_pca_angle(landmarks):
    """
    Compute rotation angle to align the principal axis of landmarks to vertical (90 degrees).

    Uses PCA: finds the eigenvector with the largest eigenvalue of the 2x2 covariance matrix,
    then computes the rotation needed to align that axis to vertical.

    Parameters:
        landmarks (np.ndarray): Nx2 array of (x, y) landmark coordinates
    Returns:
        float: Rotation angle in degrees (OpenCV convention for getRotationMatrix2D)
    """
    cov = np.cov(landmarks.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Principal axis = eigenvector with largest eigenvalue
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    theta = np.degrees(np.arctan2(principal[1], principal[0]))
    # Rotation to align principal axis to vertical (90 degrees)
    rotation = theta - 90.0
    return rotation


def ensure_tip_up(landmarks, angle, tip_index=[0, 1], base_indices=[8]):
    """
    Resolve PCA's 180-degree sign ambiguity by ensuring the toe tip points up after rotation.

    Simulates the rotation on tip and base landmarks using OpenCV's rotation convention.
    In image coordinates, Y increases downward, so "tip up" means tip_y < base_y.
    If the tip would end up below the base, adds 180 degrees.

    Parameters:
        landmarks (np.ndarray): Nx2 array of landmark coordinates
        angle (float): Rotation angle in degrees
        tip_index (int or list): Index/indices of tip landmark(s) to average (default: 8)
        base_indices (list): Indices of base landmarks to average (default: [0, 1])
    Returns:
        float: Corrected rotation angle
    """
    if base_indices is None:
        base_indices = [0, 1]

    if isinstance(tip_index, (list, tuple)):
        tip = np.mean(landmarks[tip_index], axis=0)
    else:
        tip = landmarks[tip_index]
    base = np.mean(landmarks[base_indices], axis=0)
    centroid = np.mean(landmarks, axis=0)

    # Simulate rotation using OpenCV convention:
    # M = [[cos(a), sin(a), tx], [-sin(a), cos(a), ty]]
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # Rotated Y coordinates (only need Y to check up/down)
    dx_tip, dy_tip = tip[0] - centroid[0], tip[1] - centroid[1]
    tip_y_rot = -sin_a * dx_tip + cos_a * dy_tip

    dx_base, dy_base = base[0] - centroid[0], base[1] - centroid[1]
    base_y_rot = -sin_a * dx_base + cos_a * dy_base

    # In image coords, "tip up" means tip_y < base_y
    if tip_y_rot > base_y_rot:
        return angle + 180.0
    return angle


def rotate_image_expanded(image, angle, center=None):
    """
    Rotate image around a center point with canvas expansion to prevent clipping.
    Empty regions are filled with white (255, 255, 255).

    Parameters:
        image (np.ndarray): Input image
        angle (float): Rotation angle in degrees (OpenCV convention)
        center (tuple): (x, y) rotation center. If None, uses image center.
    Returns:
        tuple: (rotated_image, affine_matrix) where affine_matrix is the 2x3 transformation matrix
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    rad = np.radians(angle)

    new_w = int(np.ceil(h * abs(np.sin(rad)) + w * abs(np.cos(rad))))
    new_h = int(np.ceil(h * abs(np.cos(rad)) + w * abs(np.sin(rad))))

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Adjust translation to center content on the expanded canvas
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    return rotated, M



def transform_landmarks(landmarks, matrix):
    """
    Apply a 2x3 affine transformation matrix to landmark coordinates.
    """
    # Augment landmarks with ones for matrix multiplication
    ones = np.ones((landmarks.shape[0], 1))
    points_ones = np.hstack([landmarks, ones])
    # Apply transformation: (3xN).T = Nx3. Matrix is 2x3. Result Nx2.
    transformed = matrix.dot(points_ones.T).T
    return transformed


def find_orientation_cv(image_crop):
    """
    Find the orientation of the primary object in the crop using Otsu thresholding + PCA.
    Returns the angle needed to rotate the object to vertical (90 degrees).
    """
    if image_crop is None or image_crop.size == 0:
        print(f"  [find_orientation_cv] Empty image, returning 0.0")
        return 0.0

    # Gray & Blur
    if len(image_crop.shape) == 3:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_crop

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu Threshold — assumes object is darker than background (BINARY_INV)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"  [find_orientation_cv] Otsu threshold: {ret:.1f}, image size: {gray.shape[1]}x{gray.shape[0]}")

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"  [find_orientation_cv] No contours found, returning 0.0")
        return 0.0

    # Max Area Contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area = gray.shape[0] * gray.shape[1]
    print(f"  [find_orientation_cv] Largest contour area: {area:.0f} ({100*area/total_area:.1f}% of image), total contours: {len(contours)}")

    if area < 100:  # Too small
        print(f"  [find_orientation_cv] Contour too small (<100px), returning 0.0")
        return 0.0

    # PCA Orientation
    sz = len(c)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = c[i, 0, 0]
        data_pts[i, 1] = c[i, 0, 1]

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    angle_pca_rad = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    angle_pca_deg = np.degrees(angle_pca_rad)

    # Target = 90 (vertical)
    rotation_needed = angle_pca_deg - 90.0

    # Normalize to [-90, 90] to resolve PCA's 180° ambiguity.
    # We only care about making the axis vertical, not which end points up
    # (tip direction is handled separately by ensure_tip_up after prediction).
    while rotation_needed > 90.0:
        rotation_needed -= 180.0
    while rotation_needed < -90.0:
        rotation_needed += 180.0

    print(f"  [find_orientation_cv] PCA eigenvalues: [{eigenvalues[0][0]:.0f}, {eigenvalues[1][0]:.0f}], ratio: {eigenvalues[0][0]/max(eigenvalues[1][0],1):.1f}")
    print(f"  [find_orientation_cv] Principal axis angle: {angle_pca_deg:.1f}deg, rotation needed: {rotation_needed:.1f}deg")

    return rotation_needed


def calc_shape_error(shape_points, average_points):
    """
    Calculate Procrustes-like shape error between predicted and average landmarks.
    Both shapes are centered and scale-normalized before computing MSE.

    Parameters:
        shape_points (np.ndarray): Nx2 predicted landmark coordinates
        average_points (np.ndarray): Nx2 average landmark coordinates
    Returns:
        float: Mean squared error after normalization
    """
    if average_points is None or len(shape_points) != len(average_points):
        return float('inf')

    shape_centered = shape_points - np.mean(shape_points, axis=0)
    avg_centered = average_points - np.mean(average_points, axis=0)

    shape_scale = np.sqrt(np.sum(shape_centered**2))
    avg_scale = np.sqrt(np.sum(avg_centered**2))

    shape_norm = shape_centered / shape_scale if shape_scale > 0 else shape_centered
    avg_norm = avg_centered / avg_scale if avg_scale > 0 else avg_centered

    return np.mean(np.sum((shape_norm - avg_norm)**2, axis=1))


def score_landmark_quality(landmarks):
    """Score how well landmarks form an elongated, ordered pattern.

    Used to select the best rotation angle during multi-angle prediction.
    Higher score = more elongated and consistently ordered landmarks.

    Returns PCA eigenvalue ratio * monotonicity correlation.
    """
    if len(landmarks) < 3:
        return 0.0

    cov = np.cov(landmarks.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]

    if eigenvalues[1] < 1e-6:
        return eigenvalues[0]  # Nearly degenerate

    ratio = eigenvalues[0] / eigenvalues[1]

    # Check sequential ordering along principal axis
    eigenvectors = np.linalg.eigh(cov)[1]
    principal = eigenvectors[:, np.argmax(np.linalg.eigvalsh(cov))]
    projections = landmarks @ principal
    indices = np.arange(len(landmarks))
    corr = np.corrcoef(projections, indices)[0, 1]
    if corr < 0:
        corr = -corr  # Direction doesn't matter, just ordering

    return ratio * max(0.0, corr)


def predictions_to_xml_single_with_yolo(image_path: str, output: str,
                                        yolo_model_path: str = None,
                                        toe_predictor_path: str = None,
                                        scale_predictor_path: str = None,
                                        finger_predictor_path: str = None,
                                        conf_threshold: float = 0.25,
                                        padding_ratio: float = 0.3,
                                        scale_bar_length_mm: float = 10.0,
                                        target_predictor_type: str = None,
                                        cached_yolo_model=None,
                                        cached_dlib_predictors: dict = None):
    """
    Generates dlib format xml file for a single image using YOLO for bounding box detection
    and dlib shape predictor for landmark prediction. YOLO detects multiple objects (toe, finger, scale)
    and each detection is processed with its corresponding predictor.
    
    For scale bars: Uses YOLO only (no ml-morph/dlib predictor). Creates two landmarks by removing
    1mm from the left and right edges of the YOLO bounding box.
    
    For all toe/finger detections: The image region is cropped to the YOLO bounding box
    before running the dlib predictor, then coordinates are transformed back.
    
    Parameters:
    ----------
        image_path (str): Path to the input image
        output (str): Path for the output XML file
        yolo_model_path (str): Path to the YOLO model file (.pt)
        toe_predictor_path (str): Path to the toe dlib shape predictor file (.dat)
        scale_predictor_path (str): Path to the scale dlib shape predictor file (.dat) - NOT USED for scale bars
        finger_predictor_path (str): Path to the finger dlib shape predictor file (.dat)
        conf_threshold (float): Confidence threshold for YOLO detections
        padding_ratio (float): Padding ratio for bounding boxes when using cropped predictors (default: 0.3 = 30%)
        scale_bar_length_mm (float): Length of the scale bar in mm (default: 10.0). Used to calculate 1mm offset.
        target_predictor_type (str): Optional filter to only process specific type ('toe', 'finger', 'scale'). 
                                     If None, processes all detections. If specified, only processes that type.
        cached_yolo_model: Optional pre-loaded YOLO model to avoid reloading
        cached_dlib_predictors (dict): Optional pre-loaded dlib predictors dict with keys like 'toe', 'finger'
    """
    # Use cached predictors if provided, otherwise load from paths
    predictors = {}

    if cached_dlib_predictors:
        # Use cached predictors
        if 'toe' in cached_dlib_predictors:
            predictors['toe'] = cached_dlib_predictors['toe']
        if 'finger' in cached_dlib_predictors:
            predictors['finger'] = cached_dlib_predictors['finger']
        print(f"Using cached dlib predictors: {list(predictors.keys())}")
    else:
        # Load predictors from paths (fallback when no cached predictors)
        if (toe_predictor_path or finger_predictor_path) and dlib is None:
            print("Error: dlib is not installed, but predictors were requested. Cannot load predictors.")
        else:
            if toe_predictor_path and os.path.exists(toe_predictor_path):
                predictors['toe'] = dlib.shape_predictor(toe_predictor_path)
            if finger_predictor_path and os.path.exists(finger_predictor_path):
                predictors['finger'] = dlib.shape_predictor(finger_predictor_path)
    
    # Note: Scale bars don't require a predictor (they use YOLO only)
    if not predictors:
        raise ValueError("At least one predictor (toe or finger) must be provided")
    
    root, images_e = initialize_xml()

    image_e = ET.Element('image')
    image_e.set('file', str(image_path))
    
    # Load image via PIL (same as YOLO training pipeline)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img_pil = Image.open(image_path)

    # Ensure image is in RGB mode for consistent handling
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    # Convert to BGR numpy array for OpenCV/dlib operations
    img_raw_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    h_img, w_img = img_raw_bgr.shape[:2]

    print(f"\n{'='*60}")
    print(f"Starting prediction pipeline for: {os.path.basename(image_path)}")
    print(f"Image size: {w_img}x{h_img}")
    print(f"Predictors loaded: {list(predictors.keys())}")
    print(f"Padding ratio: {padding_ratio}, Conf threshold: {conf_threshold}")
    print(f"{'='*60}")

    # Use cached YOLO model if provided, otherwise load from path
    obj_count = 0
    model = cached_yolo_model
    if model is None and yolo_model_path and os.path.exists(yolo_model_path):
        print(f"Loading YOLO model from: {yolo_model_path}")
        from ultralytics import YOLO
        model = YOLO(yolo_model_path, task="obb")

    if model is not None:
        try:
            # Check if model is an OrtYoloDetector (direct ONNX Runtime inference)
            from ort_inference import OrtYoloDetector
            _is_ort = isinstance(model, OrtYoloDetector)

            obj_count = 0
            inv_scale = 1.0  # Default for ORT path (corners already in original coords)

            if not _is_ort:
                # Ultralytics-specific: downsample, run model, get results object
                YOLO_MAX_DIM = 4096
                max_dim = max(h_img, w_img)
                if max_dim > YOLO_MAX_DIM:
                    ds_scale = YOLO_MAX_DIM / max_dim
                    small_bgr = cv2.resize(img_raw_bgr, (int(w_img * ds_scale), int(h_img * ds_scale)),
                                           interpolation=cv2.INTER_AREA)
                    print(f"Downsampled for YOLO: {w_img}x{h_img} -> {small_bgr.shape[1]}x{small_bgr.shape[0]} (scale={ds_scale:.3f})")
                else:
                    ds_scale = 1.0
                    small_bgr = img_raw_bgr
                inv_scale = 1.0 / ds_scale

                # Run YOLO on the downsampled image
                small_rgb_pil = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
                results = model(small_rgb_pil, conf=conf_threshold, imgsz=1280, device='cpu', verbose=False)
                print(f"YOLO model returned {len(results)} result(s)")

                class_names = model.names if hasattr(model, 'names') else {}
                print(f"YOLO class names: {class_names}")

                res = results[0]
                names = res.names

            def _iou_aabb(a_corners, b_corners):
                """Axis-aligned bounding box IoU between two sets of OBB corners."""
                ax, ay, aw, ah = cv2.boundingRect(a_corners.astype(np.int32))
                bx, by, bw, bh = cv2.boundingRect(b_corners.astype(np.int32))
                ix1 = max(ax, bx)
                iy1 = max(ay, by)
                ix2 = min(ax + aw, bx + bw)
                iy2 = min(ay + ah, by + bh)
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0.0
                inter = (ix2 - ix1) * (iy2 - iy1)
                return inter / (aw * ah + bw * bh - inter)

            def _are_adjacent_rulers(a_corners, b_corners, gap_ratio=0.5):
                """Check if two ruler OBBs are adjacent fragments of the same ruler.
                Returns True if the gap between their AABBs is small relative to their size."""
                ax, ay, aw, ah = cv2.boundingRect(a_corners.astype(np.int32))
                bx, by, bw, bh = cv2.boundingRect(b_corners.astype(np.int32))
                # Gap along x and y axes (negative means overlap)
                gap_x = max(0, max(ax, bx) - min(ax + aw, bx + bw))
                gap_y = max(0, max(ay, by) - min(ay + ah, by + bh))
                # Use the shorter dimension (ruler width) as the reference for gap tolerance
                short_a = min(aw, ah)
                short_b = min(bw, bh)
                ref = max(short_a, short_b)
                if ref == 0:
                    return False
                # Adjacent if gap is small relative to the ruler's short dimension
                # and they overlap along the perpendicular axis
                if aw > ah:  # horizontal ruler
                    return gap_x < ref * gap_ratio and gap_y < ref
                else:  # vertical ruler
                    return gap_y < ref * gap_ratio and gap_x < ref

            def _merge_ruler_detections(det_list):
                """Merge adjacent ruler fragments into a single detection.
                Returns a single merged detection with corners from the convex hull."""
                if len(det_list) <= 1:
                    return det_list
                det_list.sort(key=lambda x: x['conf'], reverse=True)
                merged = [det_list[0]]
                for det in det_list[1:]:
                    did_merge = False
                    for i, m in enumerate(merged):
                        if _iou_aabb(det['corners'], m['corners']) > 0.1 or \
                           _are_adjacent_rulers(det['corners'], m['corners']):
                            # Merge: combine all corner points and fit a minimum-area OBB
                            all_pts = np.vstack([m['corners'], det['corners']])
                            rect = cv2.minAreaRect(all_pts.astype(np.float32))
                            new_corners = cv2.boxPoints(rect).astype(np.float32)
                            w, h = rect[1]
                            merged[i] = {
                                'conf': max(m['conf'], det['conf']),
                                'corners': new_corners,
                                'obb_wh': (w, h),
                            }
                            did_merge = True
                            break
                    if not did_merge:
                        merged.append(det)
                return merged

            def _nms_best(det_list, iou_threshold=0.3, is_ruler=False):
                """NMS then top-1: suppress overlapping detections, return the highest-confidence survivor.
                For ruler class, merge adjacent fragments before NMS."""
                if not det_list:
                    return None
                if is_ruler:
                    det_list = _merge_ruler_detections(det_list)
                det_list.sort(key=lambda x: x['conf'], reverse=True)
                survivors = []
                for det in det_list:
                    if any(_iou_aabb(det['corners'], s['corners']) > iou_threshold for s in survivors):
                        continue
                    survivors.append(det)
                return survivors[0] if survivors else None

            def _get_padded_crop(img_bgr, corners_orig):
                """Crop padded bbox from full-res image, return (crop_rgb, x_off, y_off)."""
                x, y, bw, bh = cv2.boundingRect(corners_orig.astype(np.int32))
                ih, iw = img_bgr.shape[:2]
                px = int(bw * padding_ratio)
                py = int(bh * padding_ratio)
                x1 = max(0, x - px)
                y1 = max(0, y - py)
                x2 = min(iw, x + bw + px)
                y2 = min(ih, y + bh + py)
                crop = img_bgr[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.uint8)
                return crop_rgb, x1, y1

            def _predict_on_crop(predictor, img_bgr, corners_orig):
                """Run dlib predictor on a cropped region, return landmarks in original coords."""
                crop_rgb, x_off, y_off = _get_padded_crop(img_bgr, corners_orig)
                crop_h, crop_w = crop_rgb.shape[:2]
                rect = dlib.rectangle(0, 0, crop_w, crop_h)
                shape = predictor(crop_rgb, rect)
                points = []
                for k in range(shape.num_parts):
                    p = shape.part(k)
                    points.append((float(p.x + x_off), float(p.y + y_off)))
                return np.array(points, dtype=float)

            def _generate_landmark_xml(landmarks_global, label=None):
                """Generate XML box element from landmark coordinates."""
                min_lx, min_ly = np.min(landmarks_global, axis=0)
                max_lx, max_ly = np.max(landmarks_global, axis=0)
                bbox_w = max_lx - min_lx
                bbox_h = max_ly - min_ly

                box_xml = ET.Element('box')
                box_xml.set('top', str(int(min_ly)))
                box_xml.set('left', str(int(min_lx)))
                box_xml.set('width', str(int(bbox_w)))
                box_xml.set('height', str(int(bbox_h)))
                if label:
                    box_xml.set('label', label)

                for pt_i, point in enumerate(landmarks_global):
                    part = ET.SubElement(box_xml, 'part')
                    part.set('name', str(pt_i))
                    part.set('x', str(int(point[0])))
                    part.set('y', str(int(point[1])))
                return box_xml

            # --- Collect all detections first ---
            # Categorize detections
            detections = {'scale': [], 'bot_finger': [], 'bot_toe': [], 'up_finger': [], 'up_toe': [], 'id': []}

            if _is_ort:
                # ORT path: detector handles preprocessing, dual-pass, and NMS internally
                detections = model.detect(img_raw_bgr, conf_threshold=conf_threshold)
                print(f"ORT detections: {[k for k, v in detections.items() if v]}")
            else:
                # Ultralytics path: standard + flipped inference passes
                # Pass 1: Standard inference (bot_finger, bot_toe, scale, id, up)
                print(f"\n--- Standard inference pass ---")
                if res.obb is not None and len(res.obb) > 0:
                    print(f"OBB detections: {len(res.obb)}")
                    for idx in range(len(res.obb)):
                        cls_id = int(res.obb.cls[idx].item())
                        det_conf = float(res.obb.conf[idx].item())
                        if det_conf < conf_threshold:
                            continue
                        cls_name = names[cls_id]
                        corners_small = res.obb.xyxyxyxy[idx].cpu().numpy().astype(np.float32)
                        corners = corners_small * inv_scale

                        if 'ruler' in cls_name.lower() or 'scale' in cls_name.lower():
                            wh_small = res.obb.xywhr[idx].tolist()[2:4]
                            detections['scale'].append({'conf': det_conf, 'corners': corners, 'obb_wh': (wh_small[0] * inv_scale, wh_small[1] * inv_scale)})
                        elif 'bot_finger' in cls_name.lower():
                            detections['bot_finger'].append({'conf': det_conf, 'corners': corners})
                        elif 'bot_toe' in cls_name.lower():
                            detections['bot_toe'].append({'conf': det_conf, 'corners': corners})
                        elif 'up_finger' in cls_name.lower():
                            # Standard pass: corners are already in original image space, no flip needed
                            detections['up_finger'].append({'conf': det_conf, 'corners': corners})
                        elif 'up_toe' in cls_name.lower():
                            # Standard pass: corners are already in original image space, no flip needed
                            detections['up_toe'].append({'conf': det_conf, 'corners': corners})
                        elif cls_name.lower() == 'id':
                            detections['id'].append({'conf': det_conf, 'corners': corners})
                        elif cls_id == 0:
                            detections['bot_finger'].append({'conf': det_conf, 'corners': corners})
                        elif cls_id == 1:
                            detections['bot_toe'].append({'conf': det_conf, 'corners': corners})

                # Pass 2: Flipped inference (up_finger, up_toe)
                print(f"\n--- Flipped inference pass ---")
                small_flipped = cv2.flip(small_bgr, 0)
                small_flipped_pil = Image.fromarray(cv2.cvtColor(small_flipped, cv2.COLOR_BGR2RGB))
                results_flipped = model(small_flipped_pil, conf=conf_threshold, imgsz=1280, device='cpu', verbose=False)
                res_flipped = results_flipped[0]

                if res_flipped.obb is not None and len(res_flipped.obb) > 0:
                    print(f"Flipped OBB detections: {len(res_flipped.obb)}")
                    for idx in range(len(res_flipped.obb)):
                        cls_id = int(res_flipped.obb.cls[idx].item())
                        det_conf = float(res_flipped.obb.conf[idx].item())
                        if det_conf < conf_threshold:
                            continue
                        cls_name = names[cls_id]
                        corners_small = res_flipped.obb.xyxyxyxy[idx].cpu().numpy().astype(np.float32)
                        corners_flipped = corners_small * inv_scale # This is coords in FLIPPED image

                        if 'bot_finger' in cls_name.lower(): # bot_finger in flipped -> up_finger
                            detections['up_finger'].append({'conf': det_conf, 'corners': corners_flipped})
                        elif 'bot_toe' in cls_name.lower(): # bot_toe in flipped -> up_toe
                            detections['up_toe'].append({'conf': det_conf, 'corners': corners_flipped})
                        elif cls_id == 0:
                            detections['up_finger'].append({'conf': det_conf, 'corners': corners_flipped})
                        elif cls_id == 1:
                            detections['up_toe'].append({'conf': det_conf, 'corners': corners_flipped})

            # --- Process best detections ---
            flipped_bgr = None # Load lazily if needed

            for category, det_list in detections.items():
                if not det_list:
                    continue

                best_det = _nms_best(det_list, is_ruler=(category == 'scale'))
                if best_det is None:
                    continue

                print(f"Processing best {category} (conf={best_det['conf']:.3f}, {len(det_list)} candidates)")

                if category == 'scale':
                    # Process scale bar
                    corners = best_det['corners']
                    xs = [c[0] for c in corners]
                    ys = [c[1] for c in corners]
                    x1s, y1s, x2s, y2s = min(xs), min(ys), max(xs), max(ys)
                    detected_rect = dlib.rectangle(int(x1s), int(y1s), int(x2s), int(y2s))

                    box_xml = ET.Element("box")
                    box_xml.set("top", str(int(detected_rect.top())))
                    box_xml.set("left", str(int(detected_rect.left())))
                    box_xml.set("width", str(int(detected_rect.width())))
                    box_xml.set("height", str(int(detected_rect.height())))
                    box_xml.set("label", "ruler")
                    
                    obb_wh = best_det['obb_wh']
                    ruler_pixel_width = max(obb_wh[0], obb_wh[1])

                    # Physical ruler is 2mm longer than the marked scale bar (1mm margin each side)
                    total_length_mm = scale_bar_length_mm + 2.0
                    pixels_per_mm = ruler_pixel_width / total_length_mm
                    offset_pixels = pixels_per_mm * 1.0

                    # OBB corners are (4, 2). Find the short edges to get the long axis endpoints.
                    corners = best_det['corners']
                    dist01 = np.linalg.norm(corners[0] - corners[1])
                    dist12 = np.linalg.norm(corners[1] - corners[2])
                    
                    if dist01 < dist12:
                        mid1 = (corners[0] + corners[1]) / 2.0
                        mid2 = (corners[2] + corners[3]) / 2.0
                    else:
                        mid1 = (corners[1] + corners[2]) / 2.0
                        mid2 = (corners[3] + corners[0]) / 2.0

                    if mid1[0] > mid2[0]:
                        mid1, mid2 = mid2, mid1

                    vec = mid2 - mid1
                    length = np.linalg.norm(vec)
                    direction = vec / length if length > 0 else np.array([1.0, 0.0])

                    pt1 = mid1 + direction * offset_pixels
                    pt2 = mid2 - direction * offset_pixels

                    # Use IDs 0 and 1 for scale bar (visualized as 1 and 2 in the UI).
                    # When processed at the first box_idx=0, these become unique_id 0 and 1.
                    part0 = create_part(float(pt1[0]), float(pt1[1]), 0)
                    part1 = create_part(float(pt2[0]), float(pt2[1]), 1)
                    box_xml.append(part0)
                    box_xml.append(part1)
                    image_e.append(box_xml)
                    obj_count += 1

                elif category == 'id':
                    corners = best_det['corners']
                    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
                    detected_rect = dlib.rectangle(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                    box_xml = ET.Element("box")
                    box_xml.set("top", str(int(detected_rect.top())))
                    box_xml.set("left", str(int(detected_rect.left())))
                    box_xml.set("width", str(int(detected_rect.width())))
                    box_xml.set("height", str(int(detected_rect.height())))
                    box_xml.set("label", "id")
                    image_e.append(box_xml)
                    obj_count += 1
                    
                elif category in ['bot_finger', 'bot_toe']:
                     curr_predictor = predictors.get('finger') if 'finger' in category else predictors.get('toe')
                     print(f"DEBUG {category}: predictors={list(predictors.keys())}, curr_predictor={'FOUND' if curr_predictor else 'NONE'}")
                     if curr_predictor:
                          landmarks_global = _predict_on_crop(curr_predictor, img_raw_bgr, best_det['corners'])
                          image_e.append(_generate_landmark_xml(landmarks_global, label=category))
                          obj_count += 1
                     else:
                          print(f"WARNING: No predictor for {category}, SKIPPING!")

                elif category in ['up_finger', 'up_toe']:
                    # Load flipped image if not already loaded
                    if flipped_bgr is None:
                        flipped_bgr = cv2.flip(img_raw_bgr, 0)

                    curr_predictor = predictors.get('finger') if 'finger' in category else predictors.get('toe')
                    if curr_predictor:
                         # Corners may be in original image space (ORT path) or flipped space (Ultralytics flip-pass).
                         # For ORT path, we need to flip y coords to crop from flipped_bgr.
                         # For Ultralytics flip-pass, corners are already in flipped space.
                         if _is_ort:
                             corners_for_crop = np.copy(best_det['corners'])
                             corners_for_crop[:, 1] = h_img - 1 - corners_for_crop[:, 1]
                         else:
                             corners_for_crop = best_det['corners']
                         crop_rgb, x_off, y_off = _get_padded_crop(flipped_bgr, corners_for_crop)
                         crop_h, crop_w = crop_rgb.shape[:2]
                         rect = dlib.rectangle(0, 0, crop_w, crop_h)
                         shape = curr_predictor(crop_rgb, rect)
                         
                         # Map back
                         points = []
                         for k in range(shape.num_parts):
                             p = shape.part(k)
                             points.append((float(p.x + x_off), float(h_img - 1 - (p.y + y_off))))
                         
                         landmarks_global = np.array(points, dtype=float)
                         image_e.append(_generate_landmark_xml(landmarks_global, label=category))
                         obj_count += 1

            print(f"\nTotal detections: {obj_count}")
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"YOLO model path not available or doesn't exist: {yolo_model_path}")
    images_e.append(image_e)
    if output:
         # Write XML
         tree = ET.ElementTree(root)
         minidom_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
         with open(output, "w") as f:
             f.write(minidom_xml)
             
    return obj_count


# Importing to pandas tools


def natural_sort(l):
    """
    Internal function used by the dlib_xml_to_pandas. Performs the natural sorting of an array of XY
    coordinate names.

    Parameters:
    ----------
        l(array)=array to be sorted

    Returns:natural_sort_XY
    ----------
        l(array): naturally sorted array
    """
    convert = lambda text: int(text) if text.isdigit() else 0
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def dlib_xml_to_pandas(xml_file: str, parse=False):
    """
    Imports dlib xml data into a pandas dataframe. An optional file parsing argument is present
    for very specific applications. For most people, the parsing argument should remain as 'False'.

    Parameters:
    ----------
        xml_file(str):file to be imported (dlib xml format)

    Returns:
    ----------
        df(dataframe): returns a pandas dataframe containing the data in the xml_file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    landmark_list = []
    for images in root:
        for image in images:
            for boxes in image:
                box = (
                    boxes.attrib["top"]
                    + "_"
                    + boxes.attrib["left"]
                    + "_"
                    + boxes.attrib["width"]
                    + "_"
                    + boxes.attrib["height"]
                )
                for parts in boxes:
                    if parts.attrib["name"] is not None:
                        data = {
                            "id": image.attrib["file"],
                            "box_id": box,
                            "box_top": float(boxes.attrib["top"]),
                            "box_left": float(boxes.attrib["left"]),
                            "box_width": float(boxes.attrib["width"]),
                            "box_height": float(boxes.attrib["height"]),
                            "X" + parts.attrib["name"]: float(parts.attrib["x"]),
                            "Y" + parts.attrib["name"]: float(parts.attrib["y"]),
                        }

                    landmark_list.append(data)
    dataset = pd.DataFrame(landmark_list)
    if dataset.empty or "id" not in dataset.columns:
        basename = ntpath.splitext(xml_file)[0]
        pd.DataFrame().to_csv(f"{basename}.csv")
        return pd.DataFrame()
    df = dataset.groupby(["id", "box_id"], sort=False).max()
    df = df[natural_sort(df)]
    basename = ntpath.splitext(xml_file)[0]
    df.to_csv(f"{basename}.csv")
    return df


def dlib_xml_to_tps(xml_file: str):
    """
    Imports dlib xml data and converts it to tps format

    Parameters:
    ----------
        xml_file(str):file to be imported (dlib xml format)

    Returns:
    ----------
        tps (file): returns the dataset in tps format
    """
    basename = ntpath.splitext(xml_file)[0]
    tree = ET.parse(xml_file)
    root = tree.getroot()
    id = 0
    coordinates = []
    with open(f"{basename}.tps", "w") as f:
        wr = csv.writer(f, delimiter=" ")
        for images in root:
            for image in images:
                for boxes in image:
                    wr.writerows([["LM=" + str(int(len(boxes)))]])
                    for parts in boxes:
                        if parts.attrib["name"] is not None:
                            data = [
                                float(parts.attrib["x"]),
                                float(boxes.attrib["height"])
                                + 1
                                - float(parts.attrib["y"]),
                            ]
                        wr.writerows([data])
                    wr.writerows(
                        [["IMAGE=" + image.attrib["file"]], ["ID=" + str(int(id))]]
                    )
                    id += 1
def read_csv(input):
    '''
    This function reads a XY coordinate file (following the tpsDig coordinate system) containing several specimens(rows) 
    and any number of landmarks. It is generally assumed here that the file contains a header and no other 
    columns other than an id column (first column) and the X0 Y0 ...Xn Yn coordinates for n landmarks.It is also 
    assumed that the file contains no missing values.
        
    Parameters:
        input (str): The XY coordinate file (csv format)
    Returns:
        dict: dictionary containing two keys (im= image id, coords= array with 2D coordinates)
    
    '''
    csv_file = open(input, 'r') 
    csv =csv_file.read().splitlines()
    csv_file.close()
    im, coords_array = [], []
    
    for i, ln in enumerate(csv):
        if i > 0:
            im.append(ln.split(',')[0])
            coord_vec=ln.split(',')[1:]
            coords_mat = np.reshape(coord_vec, (int(len(coord_vec)/2),2))
            coords = np.array(coords_mat, dtype=float)
            coords_array.append(coords)
    return {'im': im, 'coords': coords_array}

def read_tps(input):
    '''
    This function reads a tps coordinate file containing several specimens and an arbitrary number of landmarks. 
    A single image file can contain as many specimens as wanted.
    It is generally assumed here that all specimens were landmarked in the same order.It is also  assumed that 
    the file contains no missing values.
    
    Parameters:
        input (str): The tps coordinate file
    Returns:
        dict: dictionary containing four keys 
        (lm= number of landmarks,im= image id, scl= scale, coords= array with 2D coordinates)
    
    '''
    tps_file = open(input, 'r') 
    tps = tps_file.read().splitlines()
    tps_file.close()
    lm, im, sc, coords_array = [], [], [], []

    for i, ln in enumerate(tps):
        if ln.startswith("LM"):
            lm_num = int(ln.split('=')[1])
            lm.append(lm_num)
            coords_mat = []
            for j in range(i + 1, i + 1 + lm_num):
                coords_mat.append(tps[j].split(' '))
            coords_mat = np.array(coords_mat, dtype=float)
            coords_array.append(coords_mat)

        if ln.startswith("IMAGE"):
            im.append(ln.split('=')[1])

        if ln.startswith("SCALE"):
            sc.append(ln.split('=')[1])
    return {'lm': lm, 'im': im, 'scl': sc, 'coords': coords_array}


#dlib xml tools


def add_part_element(bbox,num,sz):
    '''
    Internal function used by generate_dlib_xml. It creates a 'part' xml element containing the XY coordinates
    of an arbitrary number of landmarks. Parts are nested within boxes.
    
    Parameters:
        bbox (array): XY coordinates for a specific landmark
        num(int)=landmark id
        sz (int)=the image file's height in pixels
        
        
    Returns:
        part (xml tag): xml element containing the 2D coordinates for a specific landmark id(num)
    
    '''
    part = ET.Element('part')
    part.set('name',str(int(num)))
    part.set('x',str(int(bbox[0])))
    part.set('y',str(int(sz[0]-bbox[1])))
    return part

def add_bbox_element(bbox,sz,padding=0):
    '''
    Internal function used by generate_dlib_xml. It creates a 'bounding box' xml element containing the 
    four parameters that define the bounding box (top,left, width, height) based on the minimum and maximum XY 
    coordinates of its parts(i.e.,landmarks). An optional padding can be added to the bounding box.Boxes are 
    nested within images.
    
    Parameters:
        bbox (array): XY coordinates for all landmarks within a bounding box
        sz (int)= the image file's height in pixels
        padding(int)= optional parameter definining the amount of padding around the landmarks that should be 
                       used to define a bounding box, in pixels (int).
        
        
    Returns:
        box (xml tag): xml element containing the parameters that define a bounding box and its corresponding parts
    
    '''
    
    box = ET.Element('box')
    height = sz[0]-2
    width = sz[1]-2
    top = 1
    left = 1

    box.set('top', str(int(top)))
    box.set('left', str(int(left)))
    box.set('width', str(int(width)))
    box.set('height', str(int(height)))
    for i in range(0,len(bbox)):
        box.append(add_part_element(bbox[i,:],i,sz))
    return box

def add_image_element(image, coords, sz, path):
    '''
    Internal function used by generate_dlib_xml. It creates a 'image' xml element containing the 
    image filename and its corresponding bounding boxes and parts. 
    
    Parameters:
        image (str): image filename
        coords (array)=  XY coordinates for all landmarks within a bounding box
        sz (int)= the image file's height in pixels
        
        
    Returns:
        image (xml tag): xml element containing the parameters that define each image (boxes+parts)
    
    '''
    image_e = ET.Element('image')
    image_e.set('file', str(path))
    image_e.append(add_bbox_element(coords,sz))
    return image_e

def generate_dlib_xml(images,sizes,folder='train',out_file='output.xml'):
    '''
    Generates a dlib format xml file for training or testing of machine learning models. 
    
    Parameters:
        images (dict): dictionary output by read_tps or read_csv functions 
        sizes (dict)= dictionary of image file sizes output by the split_train_test function
        folder(str)= name of the folder containing the images 
        
        
    Returns:
        None (file written to disk)
    '''
    root = ET.Element('dataset')
    root.append(ET.Element('name'))
    root.append(ET.Element('comment'))

    images_e = ET.Element('images')
    root.append(images_e)

    for i in range(0,len(images['im'])):
        name=os.path.splitext(images['im'][i])[0]+'.jpg'
        path=os.path.join(folder,name)
        if os.path.isfile(path) is True: 
            present_tags=[]
            for img in images_e.findall('image'):
                present_tags.append(img.get('file'))   

            if path in present_tags:
                pos=present_tags.index(path)           
                images_e[pos].append(add_bbox_element(images['coords'][i],sizes[name]))

            else:    
                images_e.append(add_image_element(name,images['coords'][i],sizes[name],path))
            
    et = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
    with open(out_file, "w") as f:
        f.write(xmlstr)

#Directory preparation tools


def split_train_test(input_dir):
    '''
    Splits an image directory into 'train' and 'test' directories. The original image directory is preserved. 
    When creating the new directories, this function converts all image files to 'jpg'. The function returns
    a dictionary containing the image dimensions in the 'train' and 'test' directories.
    
    Parameters:
        input_dir(str)=original image directory
        
    Returns:
        sizes (dict): dictionary containing the image dimensions in the 'train' and 'test' directories.
    '''
    # Listing the filenames.Folders must contain only image files (extension can vary).Hidden files are ignored
    filenames = os.listdir(input_dir)
    filenames = [os.path.join(input_dir, f) for f in filenames if not f.startswith('.')]

    # Splitting the images into 'train' and 'test' directories (80/20 split)
    random.seed(845)
    filenames.sort()
    random.shuffle(filenames)
    split = int(0.8 * len(filenames))
    train_set = filenames[:split]
    test_set = filenames[split:]

    filenames = {'train':train_set,
                 'test': test_set}
    sizes={}
    for split in ['train','test']:
        sizes[split]={}
        if not os.path.exists(split):
            os.mkdir(split)
        else:
            print("Warning: the folder {} already exists. It's being replaced".format(split))
            shutil.rmtree(split)
            os.mkdir(split)

        for filename in filenames[split]:
            basename=os.path.basename(filename)
            name=os.path.splitext(basename)[0] + '.jpg'
            sizes[split][name]=image_prep(filename,name,split)
    return sizes

def image_prep(file, name, dir_path):
    '''
    Internal function used by the split_train_test function. Reads the original image files and, while 
    converting them to jpg, gathers information on the original image dimensions. 
    
    Parameters:
        file(str)=original path to the image file
        name(str)=basename of the original image file
        dir_path(str)= directory where the image file should be saved to
        
    Returns:
        file_sz(array): original image dimensions
    '''
    img = cv2.imread(file)
    if img is None:
        print('File {} was ignored'.format(file))
    else:
        file_sz= [img.shape[0],img.shape[1]]
        cv2.imwrite(os.path.join(dir_path,name), img)
    return file_sz


def natural_sort_XY(l): 
    '''
    Internal function used by the dlib_xml_to_pandas. Performs the natural sorting of an array of XY 
    coordinate names.
    
    Parameters:
        l(array)=array to be sorted
        
    Returns:
        l(array): naturally sorted array
    '''
    convert = lambda text: int(text) if text.isdigit() else 0 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def predictions_to_xml_single_from_client_annotations(image_path: str, output: str,
                                        client_ann: dict,
                                        toe_predictor_path: str = None,
                                        finger_predictor_path: str = None,
                                        scale_bar_length_mm: float = 10.0,
                                        target_predictor_type: str = None,
                                        cached_dlib_predictors: dict = None):
    import dlib
    import cv2
    import numpy as np
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    predictors = {}
    if cached_dlib_predictors:
        if 'toe' in cached_dlib_predictors:
            predictors['toe'] = cached_dlib_predictors['toe']
        if 'finger' in cached_dlib_predictors:
            predictors['finger'] = cached_dlib_predictors['finger']
    else:
        if toe_predictor_path and os.path.exists(toe_predictor_path):
            predictors['toe'] = dlib.shape_predictor(toe_predictor_path)
        if finger_predictor_path and os.path.exists(finger_predictor_path):
            predictors['finger'] = dlib.shape_predictor(finger_predictor_path)

    if not predictors:
        raise ValueError("At least one predictor (toe or finger) must be provided")

    root, images_e = initialize_xml()
    image_e = ET.Element('image')
    image_e.set('file', str(image_path))
    
    img_raw_bgr = cv2.imread(image_path)
    if img_raw_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    h_img, w_img = img_raw_bgr.shape[:2]

    flipped_bgr = None
    obj_count = 0
    padding_ratio = 0.3

    def _iou_aabb(a_corners, b_corners):
        """Axis-aligned bounding box IoU between two sets of OBB corners."""
        ax, ay, aw, ah = cv2.boundingRect(a_corners.astype(np.int32))
        bx, by, bw, bh = cv2.boundingRect(b_corners.astype(np.int32))
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / (aw * ah + bw * bh - inter)

    def _are_adjacent_rulers(a_corners, b_corners, gap_ratio=0.5):
        """Check if two ruler OBBs are adjacent fragments of the same ruler."""
        ax, ay, aw, ah = cv2.boundingRect(a_corners.astype(np.int32))
        bx, by, bw, bh = cv2.boundingRect(b_corners.astype(np.int32))
        gap_x = max(0, max(ax, bx) - min(ax + aw, bx + bw))
        gap_y = max(0, max(ay, by) - min(ay + ah, by + bh))
        short_a = min(aw, ah)
        short_b = min(bw, bh)
        ref = max(short_a, short_b)
        if ref == 0:
            return False
        if aw > ah:
            return gap_x < ref * gap_ratio and gap_y < ref
        else:
            return gap_y < ref * gap_ratio and gap_x < ref

    def _merge_ruler_detections(det_list):
        """Merge adjacent ruler fragments into a single detection."""
        if len(det_list) <= 1:
            return det_list
        det_list.sort(key=lambda x: x['conf'], reverse=True)
        merged = [det_list[0]]
        for det in det_list[1:]:
            did_merge = False
            for i, m in enumerate(merged):
                if _iou_aabb(det['corners'], m['corners']) > 0.1 or \
                   _are_adjacent_rulers(det['corners'], m['corners']):
                    all_pts = np.vstack([m['corners'], det['corners']])
                    rect = cv2.minAreaRect(all_pts.astype(np.float32))
                    new_corners = cv2.boxPoints(rect).astype(np.float32)
                    w, h = rect[1]
                    merged[i] = {
                        'conf': max(m['conf'], det['conf']),
                        'corners': new_corners,
                        'obb_wh': (w, h),
                    }
                    did_merge = True
                    break
            if not did_merge:
                merged.append(det)
        return merged

    def _nms_best(det_list, iou_threshold=0.3, is_ruler=False):
        """NMS then top-1, with ruler fragment merging."""
        if not det_list:
            return None
        if is_ruler:
            det_list = _merge_ruler_detections(det_list)
        det_list.sort(key=lambda x: x['conf'], reverse=True)
        survivors = []
        for det in det_list:
            if any(_iou_aabb(det['corners'], s['corners']) > iou_threshold for s in survivors):
                continue
            survivors.append(det)
        return survivors[0] if survivors else None

    def _get_padded_crop(img_bgr, corners_orig):
        """Crop padded bbox from full-res image, return (crop_rgb, x_off, y_off)."""
        x, y, bw, bh = cv2.boundingRect(corners_orig.astype(np.int32))
        ih, iw = img_bgr.shape[:2]
        px = int(bw * padding_ratio)
        py = int(bh * padding_ratio)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(iw, x + bw + px)
        y2 = min(ih, y + bh + py)
        crop = img_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_rgb = np.ascontiguousarray(crop_rgb, dtype=np.uint8)
        return crop_rgb, x1, y1

    def _predict_on_crop(curr_predictor, img_bgr, corners_orig):
        """Run dlib predictor on a cropped region, return landmarks in original coords."""
        crop_rgb, x_off, y_off = _get_padded_crop(img_bgr, corners_orig)
        crop_h, crop_w = crop_rgb.shape[:2]
        rect = dlib.rectangle(0, 0, crop_w, crop_h)
        shape = curr_predictor(crop_rgb, rect)
        points = []
        for k in range(shape.num_parts):
            p = shape.part(k)
            points.append((float(p.x + x_off), float(p.y + y_off)))
        return np.array(points, dtype=float)

    def _generate_landmark_xml(landmarks_global, label=None):
        """Generate XML box element from landmark coordinates."""
        min_lx, min_ly = np.min(landmarks_global, axis=0)
        max_lx, max_ly = np.max(landmarks_global, axis=0)
        bbox_w = max_lx - min_lx
        bbox_h = max_ly - min_ly

        box_xml = ET.Element('box')
        box_xml.set('top', str(int(min_ly)))
        box_xml.set('left', str(int(min_lx)))
        box_xml.set('width', str(int(bbox_w)))
        box_xml.set('height', str(int(bbox_h)))
        if label:
            box_xml.set('label', label)

        for pt_i, point in enumerate(landmarks_global):
            part = ET.SubElement(box_xml, 'part')
            part.set('name', str(pt_i))
            part.set('x', str(int(point[0])))
            part.set('y', str(int(point[1])))
        return box_xml

    bounding_boxes = client_ann.get("bounding_boxes", [])
    
    # Sort by confidence descending
    if any('confidence' in b for b in bounding_boxes):
        bounding_boxes.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
    # Categorize best boxes
    detections = {'scale': [], 'bot_finger': [], 'bot_toe': [], 'up_finger': [], 'up_toe': [], 'id': []}
    
    for box in bounding_boxes:
        label = box.get('label', '').lower()
        if 'obb_corners' not in box:
            continue
            
        obb = box['obb_corners']
        corners = np.array([[pt['x'], pt['y']] for pt in obb], dtype=np.float32)
        det_conf = box.get('confidence', 0)
        
        if 'ruler' in label or 'scale' in label:
            detections['scale'].append({'conf': det_conf, 'corners': corners, 'obb_wh': (box['width'], box['height'])})
        elif 'bot_finger' in label:
            detections['bot_finger'].append({'conf': det_conf, 'corners': corners})
        elif 'bot_toe' in label:
            detections['bot_toe'].append({'conf': det_conf, 'corners': corners})
        elif 'up_finger' in label:
            # NOTE: Frontend (OnnxService) converts up_finger corners BACK to original image space.
            # Do NOT re-flip here. We will flip at crop time.
            detections['up_finger'].append({'conf': det_conf, 'corners': corners})
        elif 'up_toe' in label:
            # NOTE: Frontend (OnnxService) converts up_toe corners BACK to original image space.
            # Do NOT re-flip here. We will flip at crop time.
            detections['up_toe'].append({'conf': det_conf, 'corners': corners})
        elif label == 'id':
            detections['id'].append({'conf': det_conf, 'corners': corners})

    for category, det_list in detections.items():
        if not det_list:
            continue

        best_det = _nms_best(det_list, is_ruler=(category == 'scale'))
        if best_det is None:
            continue
        try:
            if category == 'scale':
                corners = best_det['corners']
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                x1s, y1s, x2s, y2s = min(xs), min(ys), max(xs), max(ys)
                detected_rect = dlib.rectangle(int(x1s), int(y1s), int(x2s), int(y2s))

                box_xml = ET.Element("box")
                box_xml.set("top", str(int(detected_rect.top())))
                box_xml.set("left", str(int(detected_rect.left())))
                box_xml.set("width", str(int(detected_rect.width())))
                box_xml.set("height", str(int(detected_rect.height())))
                box_xml.set("label", "ruler")
                
                obb_wh = best_det['obb_wh']
                ruler_pixel_width = max(obb_wh[0], obb_wh[1])
                # Physical ruler is 2mm longer than the marked scale bar (1mm margin each side)
                total_length_mm = scale_bar_length_mm + 2.0
                pixels_per_mm = ruler_pixel_width / total_length_mm
                offset_pixels = pixels_per_mm * 1.0

                dist01 = np.linalg.norm(corners[0] - corners[1])
                dist12 = np.linalg.norm(corners[1] - corners[2])
                
                if dist01 < dist12:
                    mid1 = (corners[0] + corners[1]) / 2.0
                    mid2 = (corners[2] + corners[3]) / 2.0
                else:
                    mid1 = (corners[1] + corners[2]) / 2.0
                    mid2 = (corners[3] + corners[0]) / 2.0

                if mid1[0] > mid2[0]:
                    mid1, mid2 = mid2, mid1

                vec = mid2 - mid1
                length = np.linalg.norm(vec)
                direction = vec / length if length > 0 else np.array([1.0, 0.0])

                pt1 = mid1 + direction * offset_pixels
                pt2 = mid2 - direction * offset_pixels

                # Use IDs 0 and 1 for scale bar (visualized as 1 and 2 in the UI).
                part0 = create_part(float(pt1[0]), float(pt1[1]), 0)
                part1 = create_part(float(pt2[0]), float(pt2[1]), 1)
                box_xml.append(part0)
                box_xml.append(part1)
                image_e.append(box_xml)
                obj_count += 1

            elif category == 'id':
                corners = best_det['corners']
                xs, ys = [c[0] for c in corners], [c[1] for c in corners]
                detected_rect = dlib.rectangle(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                box_xml = ET.Element("box")
                box_xml.set("top", str(int(detected_rect.top())))
                box_xml.set("left", str(int(detected_rect.left())))
                box_xml.set("width", str(int(detected_rect.width())))
                box_xml.set("height", str(int(detected_rect.height())))
                box_xml.set("label", "id")
                image_e.append(box_xml)
                obj_count += 1
                
            elif category in ['bot_finger', 'bot_toe']:
                curr_predictor = predictors.get('finger') if 'finger' in category else predictors.get('toe')
                if curr_predictor:
                     landmarks_global = _predict_on_crop(curr_predictor, img_raw_bgr, best_det['corners'])
                     image_e.append(_generate_landmark_xml(landmarks_global, label=category))
                     obj_count += 1

            elif category in ['up_finger', 'up_toe']:
                if flipped_bgr is None:
                    flipped_bgr = cv2.flip(img_raw_bgr, 0)
                
                curr_predictor = predictors.get('finger') if 'finger' in category else predictors.get('toe')
                if curr_predictor:
                     # corners are in original image space (frontend already unflipped them).
                     # To crop from flipped_bgr, we need to flip y coords.
                     corners_for_crop = np.copy(best_det['corners'])
                     corners_for_crop[:, 1] = h_img - 1 - corners_for_crop[:, 1]
                     crop_rgb, x_off, y_off = _get_padded_crop(flipped_bgr, corners_for_crop)
                     crop_h, crop_w = crop_rgb.shape[:2]
                     rect = dlib.rectangle(0, 0, crop_w, crop_h)
                     shape = curr_predictor(crop_rgb, rect)
                     
                     points = []
                     for k in range(shape.num_parts):
                         p = shape.part(k)
                         points.append((float(p.x + x_off), float(h_img - 1 - (p.y + y_off))))
                     
                     landmarks_global = np.array(points, dtype=float)
                     image_e.append(_generate_landmark_xml(landmarks_global))
                     obj_count += 1

        except Exception as e:
            import traceback
            print(f"Error processing client_ann detection category={category}: {e}")
            traceback.print_exc()

    images_e.append(image_e)
    if output:
         tree = ET.ElementTree(root)
         minidom_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
         with open(output, "w") as f:
             f.write(minidom_xml)
             
    return obj_count

