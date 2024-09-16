import os
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_image(image_path):
    """
    Checks if the image is valid and not corrupted.
    
    Args:
    - image_path (str): Path to the image file.
    
    Returns:
    - bool: True if the image is valid, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it is an image
        # Reload the image to check if it can be processed (optional)
        with Image.open(image_path) as img:
            img.load()
        return True
    except (IOError, OSError, Image.DecompressionBombError) as e:
        logging.error(f"Image validation failed for {image_path}: {e}")
        return False

def delete_invalid_image(image_path):
    """
    Deletes the specified image file.
    
    Args:
    - image_path (str): Path to the image file to delete.
    """
    try:
        os.remove(image_path)
        logging.info(f"Deleted invalid or corrupted image: {image_path}")
    except Exception as e:
        logging.error(f"Error deleting image {image_path}: {e}")

def check_and_clean_images_in_directory(directory):
    """
    Loops through all images in the specified directory, checks if they are valid,
    and deletes them if they are found to be invalid.
    
    Args:
    - directory (str): Path to the directory containing images.
    """
    logging.info(f"Checking all images in directory: {directory}")
    
    # Loop through all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Include valid image file extensions
                image_path = os.path.join(root, file)
                if not validate_image(image_path):
                    delete_invalid_image(image_path)  # Delete the invalid image

if __name__ == "__main__":
    # Define the paths to your image directories
    train_image_folder = 'images/train'
    test_image_folder = 'images/test'
    
    # Check and clean all images in both directories
    check_and_clean_images_in_directory(train_image_folder)
    check_and_clean_images_in_directory(test_image_folder)
