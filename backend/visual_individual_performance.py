import xml.etree.ElementTree as ET
import numpy as np
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import os

def parse_xml(file_path):
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
    
    # Dictionary to store image filenames and their corresponding landmarks
    data = []
    
    for image in root.findall('.//image'):
        image_file = image.get('file')
        
        # Extract just the filename from the full file path
        image_name = os.path.basename(image_file)
        
        # Create arrays for x and y coordinates
        x_coords = []
        y_coords = []
        
        for part in image.findall('.//part'):
            x = float(part.get('x'))
            y = float(part.get('y'))
            
            # Store the coordinates in separate arrays
            x_coords.append(x)
            y_coords.append(y)
        
        # Store the coordinates arrays in the dictionary for the current image
        data.append({'name' : image_name,'x': x_coords,'y': y_coords})
    
    return data

def create_image(output_xml, output_folder):
    file_path = output_xml
    data = parse_xml(file_path)
    directory = "test_auto"
    output_image_paths = []
    
    for i in range(len(data)):
        image = plt.imread(os.path.join(directory,data[i]['name']))
        height_pixels, width_pixels = image.shape[0], image.shape[1]
        print(image.shape)
        dpi = 500
        width_inches = width_pixels / dpi
        height_inches = height_pixels / dpi
        print(image.shape)
        fig, ax = plt.subplots(figsize=(width_inches, height_inches))
        ax.imshow(image)
        ax.axis('off')
        ax.scatter(data[i]["x"], data[i]["y"], s = 5, color = 'red')

        output_path = os.path.join("outputs", "annotated_"+data[i]['name'])
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        output_image_paths.append(output_path)
    
    return output_image_paths


# import xml.etree.ElementTree as ET
# import numpy as np
# import argparse
# import numpy as np
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import cv2
# import os

# from PIL import Image

# # Use a non-GUI backend for Matplotlib
# plt.switch_backend('Agg')



# def parse_xml(file_path):
#     '''
#     Parses an XML file to extract image filenames and their corresponding landmark coordinates.

#     Parameters:
#         file_path (str): Path to the XML file containing image data and landmarks.

#     Returns:
#         dict: A dictionary where keys are image filenames and values are dictionaries of landmarks, 
#             with landmark names as keys and their (x, y) coordinates as values.
#             Note: Returns the first entry only.
#     '''
#     tree = ET.parse(file_path)
#     root = tree.getroot()
    
#     # Dictionary to store image filenames and their corresponding landmarks
#     data = {}
    
#     for image in root.findall('.//image'):
#         image_file = image.get('file')
#         parts = {}
#         for part in image.findall('.//part'):
#             part_name = int(part.get('name'))
#             x = float(part.get('x'))
#             y = float(part.get('y'))
#             parts[part_name] = (x, y)
        
#         data[image_file] = parts
#     return data

# def find_groundtruth(file_path, image_name):
#     '''
#     Finds the ground truth landmark coordinates for a specified image in an XML file.

#     Parameters:
#         file_path (str): Path to the XML file containing image data and landmarks.
#         image_name (str): The name of the image for which to retrieve landmark coordinates.

#     Returns:
#         list: A list containing two lists: 
#             - The first list contains the x-coordinates of the landmarks.
#             - The second list contains the y-coordinates of the landmarks.
#             If the image is not found, both lists will be empty.
#     '''

#     tree = ET.parse(file_path)
#     root = tree.getroot()
#     ground_truth = []
#     X = []
#     Y = []
#     for image in root.findall('.//image'):
#         image_file = image.get('file')
#         if image_file == image_name:
#             for part in image.findall('.//part'):
#                 x = int(part.get('x'))
#                 y = int(part.get('y'))
#                 X.append(x)
#                 Y.append(y)
#     return [X, Y]

# def find_output(file_path, image_name):
#     '''
#     Retrieves the landmark coordinates for a specified image from an XML file.

#     Parameters:
#         file_path (str): Path to the XML file containing image data and landmarks.
#         image_name (str): The name of the image for which to retrieve landmark coordinates.

#     Returns:
#         list: A list containing two lists: 
#             - The first list contains the x-coordinates of the landmarks.
#             - The second list contains the y-coordinates of the landmarks.
#             If the image is not found, both lists will be empty.

#     Note: The input `image_name` is modified to remove './' for comparison.
#     '''
#     tree = ET.parse(file_path)
#     root = tree.getroot()
#     print(file_path, image_name)
#     print('tree', tree)
#     ground_truth = []
#     X = []
#     Y = []
#     for image in root.findall('.//image'):
#         image_file = image.get('file')
#         image_file = image_file.replace('./', '')
#         # print("n:", image_name, "f:", image_file)
#         if image_file == image_name:
#             # print(image_name)
#             for part in image.findall('.//part'):
#                 x = int(part.get('x'))
#                 y = int(part.get('y'))
#                 X.append(x)
#                 Y.append(y)
#     return [X, Y]

# def get_image_name(lizard_number, groundtruth_xml):
#     '''
#     Retrieves the file name of an image corresponding to a specified lizard number 
#     from a groundtruth XML file.

#     Parameters:
#         lizard_number (int): The index of the lizard whose image file name is to be retrieved.
#         groundtruth_xml (str): The path to the XML file containing image metadata.

#     Returns:
#         str: The file name of the image corresponding to the given lizard number.
#     '''
#     tree = ET.parse(groundtruth_xml)
#     root = tree.getroot()
#     for i, image in enumerate(root.findall('.//image')):
#         if int(i) == int(lizard_number):
#             return image.get('file')

# def create_image(groundtruth_xml, output_xml, output_folder):
#     '''
#     Main function to process images and compare ground truth landmarks with model outputs.

#     This function creates an output folder, parses ground truth and output XML files, 
#     and generates visualizations of the ground truth and model output landmarks for each image.

#     Parameters:
#         groundtruth_xml (str): Path to the ground truth XML file containing landmark data.
#         output_xml (str): Path to the output XML file containing model predictions.
#         output_folder (str): Path to the folder where output images will be saved.

#     Returns:
#         None

#     Usage:
#         This function is typically called from the command line or another script.
#         It will create visualizations and save them in the specified output folder.
#     '''
#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Parse the groundtruth XML to find the number of images
#     tree = ET.parse(groundtruth_xml)
#     root = tree.getroot()
#     num_images = len(root.findall('.//image'))

#     for lizard_number in range(num_images):
#         name = get_image_name(lizard_number, groundtruth_xml)
#         groundtruth = find_groundtruth(groundtruth_xml, name)
#         output = find_output(output_xml, name)

#         # Load the background image using Pillow and rotate it
#         background_image = cv2.imread(name)

#         # Convert the rotated image to a NumPy array for Matplotlib
#         background_image_np = np.array(background_image)

#         # Get image dimensions
#         img_height, img_width, _ = background_image_np.shape

#         # Create a figure with dimensions matching the image
#         fig, ax = plt.subplots(figsize=(img_width / 100, img_height / 100), dpi=100)

#         # Display the rotated background image in the axes
#         ax.imshow(background_image_np, extent=(0, img_width, 0, img_height))

#         # Plot data on top of the background image
#         ax.scatter(groundtruth[0], groundtruth[1], s=10, color="lawngreen", label="Ground Truth")
#         ax.scatter(output[0], output[1], s=10, color="deeppink", label="Model Output", alpha=0.7)

#         # Adjust the axis limits to match the image dimensions
#         ax.set_xlim(0, img_width)
#         ax.set_ylim(img_height, 0)  # Invert y-axis for image coordinates

#         # Add legend
#         ax.legend(loc='upper right', fontsize=8)

#         # Remove axis ticks and labels for cleaner visualization
#         ax.axis('off')

#         # Save the figure to the specified output folder
#         output_path = os.path.join(output_folder, f"{lizard_number}_test_set.png")
#         plt.savefig(output_path, bbox_inches='tight', dpi=300)
#         plt.close()
