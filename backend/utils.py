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
import dlib
import os
import shutil
import random


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
            
            for item, i in enumerate(sorted(part_length, key=str)):
                x = np.median([landmark[item][0] for landmark in landmarks])
                y = np.median([landmark[item][1] for landmark in landmarks])
                if ignore is not None:
                    if i not in ignore:
                        part = create_part(x, y, i)
                        box.append(part)
                else:
                    part = create_part(x, y, i)
                    box.append(part)

                pos = np.array(landmarks)[:, item]
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
                                + 2
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
