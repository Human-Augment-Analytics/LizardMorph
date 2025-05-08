import os
import shutil
import zipfile
import logging
import uuid
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExportHandler:
    """
    Handles the export of files to organized directories and creates zip archives of exports.
    """
    
    def __init__(self, base_dir='outputs'):
        """
        Initialize the export handler with the base output directory.
        
        Args:
            base_dir (str): Base directory for all outputs
        """
        self.base_dir = base_dir
        self.ensure_directory_exists(self.base_dir)
    
    def ensure_directory_exists(self, directory):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    def create_export_directory(self, image_name=None):
        """
        Create a timestamped directory for exports.
        
        Args:
            image_name (str, optional): Base name of the image to include in directory name
        
        Returns:
            str: Path to the created directory
        """
        # Create a timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a unique directory name
        if image_name:
            # Remove file extension and clean up the name
            base_name = os.path.splitext(image_name)[0]
            dir_name = f"{base_name}_{timestamp}"
        else:
            dir_name = f"export_{timestamp}_{uuid.uuid4().hex[:6]}"
        
        export_dir = os.path.join(self.base_dir, dir_name)
        self.ensure_directory_exists(export_dir)
        
        logger.info(f"Created export directory: {export_dir}")
        return export_dir
    
    def copy_file_to_export(self, source_path, export_dir, new_filename=None):
        """
        Copy a file to the export directory.
        
        Args:
            source_path (str): Path to the source file
            export_dir (str): Target export directory
            new_filename (str, optional): New filename to use instead of original
            
        Returns:
            str: Path to the copied file
        """
        if not os.path.exists(source_path):
            logger.error(f"Source file does not exist: {source_path}")
            return None
        
        filename = new_filename if new_filename else os.path.basename(source_path)
        destination = os.path.join(export_dir, filename)
        
        try:
            shutil.copy2(source_path, destination)
            logger.info(f"Copied file to: {destination}")
            return destination
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            return None
    
    def create_export_zip(self, export_dirs):
        """
        Create a zip file containing all export directories.
        
        Args:
            export_dirs (list): List of directories to include in the zip
            
        Returns:
            str: Path to the created zip file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"lizard_exports_{timestamp}.zip"
        zip_path = os.path.join(self.base_dir, zip_filename)
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for directory in export_dirs:
                    if not os.path.exists(directory):
                        logger.warning(f"Directory not found, skipping: {directory}")
                        continue
                    
                    dir_name = os.path.basename(directory)
                    for root, _, files in os.walk(directory):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.join(dir_name, os.path.relpath(file_path, directory))
                            zipf.write(file_path, arcname)
                            logger.info(f"Added to zip: {file_path} as {arcname}")
            
            logger.info(f"Created zip file: {zip_path}")
            return zip_path
        except Exception as e:
            logger.error(f"Error creating zip file: {str(e)}")
            return None
    
    def export_tps_file(self, coords, image_name, export_dir=None):
        """
        Create a TPS file from coordinates and save it to an export directory.
        
        Args:
            coords (list): List of coordinate dictionaries
            image_name (str): Name of the image
            export_dir (str, optional): Directory to save the TPS file
        
        Returns:
            str: Path to the TPS file
        """
        if export_dir is None:
            export_dir = self.create_export_directory(image_name)
        
        # Remove file extension and clean up the name
        base_name = os.path.splitext(image_name)[0]
        tps_filename = f"{base_name}.tps"
        tps_path = os.path.join(export_dir, tps_filename)
        
        try:
            with open(tps_path, 'w', encoding='utf-8', newline='\n') as tps_file:
                tps_file.write(f"LM={len(coords)}\n")
                
                # Write coordinates
                for point in coords:
                    if 'x' in point and 'y' in point:
                        x = float(point['x'])
                        y = float(point['y'])
                        tps_file.write(f"{x} {y}\n")
                    else:
                        logger.warning(f"Invalid point data: {point}")
                
                # Write image name
                tps_file.write(f'IMAGE={base_name}')
            
            logger.info(f"Created TPS file: {tps_path}")
            return tps_path
        except Exception as e:
            logger.error(f"Error creating TPS file: {str(e)}")
            return None
    
    def collect_all_export_directories(self):
        """
        List all export directories in the base directory.
        
        Returns:
            list: List of paths to all export directories
        """
        if not os.path.exists(self.base_dir):
            return []
        
        dirs = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                dirs.append(item_path)
        
        return dirs