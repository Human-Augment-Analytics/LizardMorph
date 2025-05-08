import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import os
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from PIL import Image, ImageEnhance

def dcm_to_jpeg(dcm_file_path, jpeg_file_path):
    dicom = pydicom.dcmread(dcm_file_path)
    
    # If DICOM file has a VOI LUT (Value of Interest Look-Up Table), apply it
    if hasattr(dicom, 'VOILUTFunction'):
        image = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        image = dicom.pixel_array
    
    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255).astype(np.uint8)

    # Convert to RGB
    if len(image.shape) == 2:  # If grayscale, convert to RGB
        image = np.stack((image,) * 3, axis=-1)
    
    pil_image = Image.fromarray(image)
    pil_image.save(jpeg_file_path, 'JPEG')

# Function to enhance image sharpness, contrast and apply Gaussian blur
def enhance_image(img, sharpness=4, contrast=1.3, blur=3):
    """Enhance image sharpness, contrast, and blur.

    Args:
        img: Loaded cv2 image.
        output_path (str): Path to save the enhanced image.
        sharpness (float, optional): Sharpness level. Defaults to 4.
        contrast (float, optional): Contrast level. Defaults to 1.3.
        blur (int, optional): Blur level. Defaults to 3.
    """

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

    return img_enhanced

def clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def image_complement(image):
    img_complement = cv2.bitwise_not(image)
    return img_complement

def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Apply image enhancement techniques to improve XRAY image.')
#     parser.add_argument('input_folder', type=str, help='Path to the folder containing input images')
#     parser.add_argument('output_folder', type=str, help='Path to the folder to store output processed images')
#     parser.add_argument('--sharpness', type=float, default=4.0, help='Sharpness level')
#     parser.add_argument('--contrast', type=float, default=1.3, help='Contrast level')
#     parser.add_argument('--blur', type=int, default=3, help='Blur for Gaussian Blur')
#     parser.add_argument('--clip_limit', type=float, default=2.0, help='Clip limit for CLAHE')
#     parser.add_argument('--tile_grid_size', type=int, nargs=2, default=(8, 8), help='Tile grid size for CLAHE (two integers)')
#     parser.add_argument('--gamma', type=float, default=1.0, help='Gamma for gamma correction')
#     parser.add_argument('--shouldPlotEveryImage', type=bool, default=False, help='Whether to plot and display each image after processing')

#     args = parser.parse_args()
def process_images(input_folder, output_folder, sharpness=4, contrast=1.3, blur=3, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.0, should_plot=False):

    for root, dirs, files in os.walk(input_folder):
        jpg_files = glob.glob(os.path.join(root, '*.jpg'))
        for jpg_file in jpg_files:
            print("Processing " + jpg_file)
            image = cv2.imread(jpg_file)

            image = enhance_image(image)
            image = clahe(image, clip_limit, tile_grid_size)
            image = gamma_correction(image, gamma)

            if should_plot:
              plt.plot()
              plt.title("Processed Image") 
              plt.imshow(image)
              plt.show()

            file_name = os.path.basename(jpg_file)
            output_path = os.path.join(output_folder, "processed_" + file_name)
            cv2.imwrite(output_path, image)



def process_single_image(input_path, output_path, sharpness=4, contrast=1.3, blur=3, clip_limit=2.0, 
                        tile_grid_size=(8, 8), gamma=1.0):
    """Process a single image with all enhancements"""
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Failed to load image: {input_path}")
    image = enhance_image(image)
    image = clahe(image, clip_limit, tile_grid_size)
    image = gamma_correction(image, gamma)
    cv2.imwrite(output_path, image)