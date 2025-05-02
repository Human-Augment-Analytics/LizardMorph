import xml.etree.ElementTree as ET
import numpy as np
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import os
import glob

# Use a non-GUI backend
plt.switch_backend('Agg')

def parse_xml_for_frontend(file_path):
    '''
    Parses an XML file to extract image filenames and their corresponding landmark coordinates.

    Parameters:
        file_path (str): Path to the XML file containing image data and landmarks.

    Returns:
        list: A list of dicts which are image filenames (without path) and values are dictionaries of landmarks, 
            with landmark names as keys and their x and y coordinates stored in separate arrays.
    '''
    tree = ET.parse(file_path)
    root = tree.getroot()
    plot_data = []
    coord_data = []
    
    
    for image in root.findall('.//image'):
        image_file = image.get('file')
        
        # Extract just the filename from the full file path
        image_name = os.path.basename(image_file)
        
        # Create arrays for x and y coordinates
        coords = []

        for part in image.findall('.//part'):
            x = float(part.get('x'))
            y = float(part.get('y'))
            
            # Store the coordinates in separate arrays
            coords.append({"x": x, "y": y})
        
        # Store the coordinates arrays in the dictionary for the current image
        data = {'name' : image_name,"coords" : coords}
        # print(data)
    
    return data
def read_tps_file(file_path):
    """Read a TPS file and return data as a list of x and y coordinate lists with their corresponding images."""
    data = []
    current_x = []
    current_y = []
    current_image = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('LM='):
                current_x = []  # Reset coordinates for the new entry
                current_y = []
                # print("New Entry")
            elif line.startswith('IMAGE='):
                current_image = line[6:]
            else:
                points = list(map(float, line.split()))
                current_x.append(points[0])
                current_y.append(points[1])
    
        if current_image is not None and current_x and current_y:
            data.append((current_image, current_x, current_y))

    return data

def create_image(output_xml, output_folder):
    file_path = output_xml
    plot_data = read_tps_file(file_path)
    directory = "tps_download"
    output_image_paths = []
    print(len(plot_data))
    
    
    
    # print(len(data['coords']))
    for i in range(len(plot_data)):
        image = plt.imread(os.path.join("upload",plot_data[i][0]+".jpg"))
        height_pixels, width_pixels = image.shape[0], image.shape[1]
        #print(image.shape)
        dpi = 500
        width_inches = width_pixels / dpi
        height_inches = height_pixels / dpi
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.imshow(image)
        ax.axis('off')
        ax.scatter(plot_data[i][1], plot_data[i][2], s = 5, color = 'red')
        output_path = os.path.join(output_folder, "annotated_"+plot_data[i][0])
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        output_image_paths.append(output_path)
        plt.close()
def invert_images(input_folder, output_folder):

    for root, dirs, files in os.walk(input_folder):
        jpg_files = glob.glob(os.path.join(root, '*.jpg'))
        for jpg_file in jpg_files:
            image = cv2.imread(jpg_file)
            image = 255 - image
            file_name = os.path.basename(jpg_file)
            output_path = os.path.join(output_folder, "inverted_" + file_name)
            cv2.imwrite(output_path, image)