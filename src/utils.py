import re
import constants
import os
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import urllib.request
from PIL import Image
import logging

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')  # Set logging level to WARNING

def common_mistake(unit):
    """
    Corrects common mistakes in units (e.g., different spellings or typos).
    """
    if unit in constants.allowed_units:
        return unit
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    return unit

def parse_string(s):
    """
    Parses and validates a prediction string, ensuring it follows the format 'number unit'.
    """
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError("Invalid format in {}".format(s))
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.allowed_units:
        raise ValueError("Invalid unit [{}] found in {}. Allowed units: {}".format(
            unit, s, constants.allowed_units))
    return number, unit

def create_placeholder_image(image_save_path):
    """
    Creates a black placeholder image if downloading fails.
    """
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
        logging.warning(f"Placeholder image created at {image_save_path}")
    except Exception as e:
        logging.error(f"Error creating placeholder image: {e}")

def download_image(image_link, save_folder, retries=5, delay=5):
    """
    Downloads an image from a given link. If the download fails after the specified retries, 
    a placeholder image is created.
    """
    if not isinstance(image_link, str) or not image_link.strip():  # Check for empty or None links
        logging.warning(f"Invalid or empty image link: {image_link}")
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return  # Skip logging for existing images

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)  # Correct use of urllib.request
            if validate_image(image_save_path):
                logging.info(f"Image successfully downloaded: {image_save_path}")
                return
            else:
                logging.warning(f"Invalid image detected, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            logging.error(f"Error downloading image {image_link}: {e}")
            time.sleep(delay)
    
    logging.error(f"Failed to download image after {retries} retries: {image_link}")
    create_placeholder_image(image_save_path)  # Create a black placeholder image for invalid links/images

def validate_image(image_path):
    """
    Checks if the downloaded image is valid and not corrupted.
    
    Args:
    - image_path (str): Path to the image file.
    
    Returns:
    - bool: True if the image is valid, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it is an image
        # Reload the image to check if it can be converted to an array (optional, can be skipped)
        img = Image.open(image_path)
        img.load()
        return True
    except Exception as e:
        logging.error(f"Image validation failed for {image_path}: {e}")
        return False

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses the image for model prediction.
    Includes resizing, normalization, etc.
    
    Args:
    - image_path (str): Path to the image.
    - target_size (tuple): Target size for resizing (width, height).
    
    Returns:
    - np.array: Preprocessed image ready for prediction.
    """
    try:
        img = Image.open(image_path).convert('RGB')  # Convert to RGB
        img = img.resize(target_size)
        img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        return img
    except Exception as e:
        logging.error(f"Error preprocessing image {image_path}: {e}")
        return None

def download_images(image_links, download_folder, allow_multiprocessing=True, max_workers=20):
    """
    Downloads multiple images using multiprocessing for efficiency.
    Adjusts the number of workers to avoid Windows handle limitations.
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        download_image_partial = partial(
            download_image, save_folder=download_folder, retries=5, delay=5)

        with multiprocessing.Pool(min(max_workers, 20)) as pool:  # Limit number of workers to avoid Windows handle limits
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=5, delay=5)

def load_and_download_images(csv_file_path, download_folder, sample_size=None):
    """
    Loads image links from a CSV file, samples a subset, and downloads them to the specified folder.
    
    Args:
    - csv_file_path (str): Path to the CSV file containing image links.
    - download_folder (str): Folder to download images to.
    - sample_size (int, optional): Number of samples to use for testing. Default is None (use full dataset).
    """
    df = pd.read_csv(csv_file_path)
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)  # Sample a subset for quick testing
    image_links = df['image_link'].tolist()
    
    # Ensure the folder exists
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Download images
    download_images(image_links, download_folder)

if __name__ == "__main__":
    # Example usage: download images from train and test datasets with sampling
    train_csv_path = 'dataset/train.csv'
    test_csv_path = 'dataset/test.csv'
    sample_test_csv_path = 'dataset/sample_test.csv'
    train_images_folder = 'images/train'
    sample_test_images_folder = 'images/sample_test'
    test_images_folder = 'images/test'

    # Download a small subset of images for testing
    sample_size_train = None
    sample_size_test = None

    # Load image links from CSV files and download them
    # load_and_download_images(train_csv_path, train_images_folder, sample_size=sample_size_train)
    load_and_download_images(test_csv_path, test_images_folder, sample_size=sample_size_test)
    # load_and_download_images(sample_test_csv_path, sample_test_images_folder, sample_size=None)
