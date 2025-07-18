"""
Utility functions for the LizardMorph MCP Server.
Contains only the functions needed by the MCP server.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import cv2
import dlib


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

    for i in sorted(length, key=str):
        coords[i] = [shape.part(i).x, shape.part(i).y]

    # return the list of (x, y)-coordinates
    return coords


def predictions_to_xml_single(predictor_name: str, image_path: str, output: str):
    """Generates dlib format xml file for a single image."""
    predictor = dlib.shape_predictor(predictor_name)
    root, images_e = initialize_xml()
    kernel = np.ones((7, 7), np.float32) / 49

    image_e = ET.Element("image")
    image_e.set("file", str(image_path))

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
    part_length = range(0, shape.num_parts)

    for item, i in enumerate(sorted(part_length, key=str)):
        x = np.median([landmark[item][0] for landmark in landmarks])
        y = np.median([landmark[item][1] for landmark in landmarks])
        part = create_part(x, y, i)
        box.append(part)

    box[:] = sorted(box, key=lambda child: (child.tag, float(child.get("name"))))
    image_e.append(box)
    images_e.append(image_e)
    pretty_xml(root, output)
