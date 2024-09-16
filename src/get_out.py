import os
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import pickle
from urllib.parse import urlparse
import multiprocessing
from functools import partial
from tqdm import tqdm

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Paths to data and images
TEST_CSV_PATH = 'dataset/test.csv'
IMAGES_DIR = 'images/test'
MODEL_PATH = 'saved_models/ocr_model2_full.pkl'
OUTPUT_CSV_PATH = 'dataset/test_out2.csv'

# Load the OCR model components
with open(MODEL_PATH, 'rb') as f:
    model_components = pickle.load(f)
    entity_unit_map = model_components['entity_unit_map']

# Function to extract the image filename from the image link
def get_image_filename(image_link):
    parsed_url = urlparse(image_link)
    return os.path.basename(parsed_url.path)

# Function to standardize units
def standardize_unit(unit):
    unit_mapping = {
        'kgs': 'kilogram',
        'kg': 'kilogram',
        'kgs.': 'kilogram',
        'kilogram': 'kilogram',
        'kilograms': 'kilogram',
        'lbs': 'pound',
        'lb': 'pound',
        'pound': 'pound',
        'pounds': 'pound',
        'g': 'gram',
        'gram': 'gram',
        'grams': 'gram',
        'mg': 'milligram',
        'milligram': 'milligram',
        'milligrams': 'milligram',
        'µg': 'microgram',
        'microgram': 'microgram',
        'micrograms': 'microgram',
        'oz': 'ounce',
        'ounce': 'ounce',
        'ounces': 'ounce',
        'cm': 'centimetre',
        'centimetre': 'centimetre',
        'centimeter': 'centimetre',
        'centimeters': 'centimetre',
        'mm': 'millimetre',
        'millimetre': 'millimetre',
        'millimeter': 'millimetre',
        'millimeters': 'millimetre',
        'm': 'metre',
        'metre': 'metre',
        'meter': 'metre',
        'meters': 'metre',
        'in': 'inch',
        'inch': 'inch',
        'inches': 'inch',
        'ft': 'foot',
        'foot': 'foot',
        'feet': 'foot',
        'yd': 'yard',
        'yard': 'yard',
        'yards': 'yard',
        'kv': 'kilovolt',
        'kilovolt': 'kilovolt',
        'kilovolts': 'kilovolt',
        'mv': 'millivolt',
        'millivolt': 'millivolt',
        'millivolts': 'millivolt',
        'v': 'volt',
        'volt': 'volt',
        'volts': 'volt',
        'w': 'watt',
        'watt': 'watt',
        'watts': 'watt',
        'kw': 'kilowatt',
        'kilowatt': 'kilowatt',
        'kilowatts': 'kilowatt',
        'l': 'litre',
        'litre': 'litre',
        'litres': 'litre',
        'ml': 'millilitre',
        'millilitre': 'millilitre',
        'millilitres': 'millilitre',
        'cl': 'centilitre',
        'centilitre': 'centilitre',
        'centilitres': 'centilitre',
        'dl': 'decilitre',
        'decilitre': 'decilitre',
        'decilitres': 'decilitre',
        'µl': 'microlitre',
        'microlitre': 'microlitre',
        'microlitres': 'microlitre',
        'pt': 'pint',
        'pint': 'pint',
        'pints': 'pint',
        'qt': 'quart',
        'quart': 'quart',
        'quarts': 'quart',
        'gal': 'gallon',
        'gallon': 'gallon',
        'gallons': 'gallon',
        'c': 'cup',
        'cup': 'cup',
        'cups': 'cup',
        # Add more mappings as needed
    }
    return unit_mapping.get(unit.lower(), unit.lower())

# Enhanced image preprocessing
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')

    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    
    # Resize the image to make it larger
    # new_width = int(image.width * 1.5)
    # new_height = int(image.height * 1.5)
    # image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Apply median filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter())
    return image

# Function to extract entity value from OCR text
def extract_entity_value(ocr_text, entity_name):
    allowed_units = entity_unit_map.get(entity_name, set())
    if not allowed_units:
        return ""
    # Create units pattern with capturing group
    units_pattern = '(' + '|'.join(re.escape(unit) for unit in allowed_units) + ')'
    # Regex pattern to find numbers followed or preceded by units
    pattern = r'(?:([0-9]+(?:[.,][0-9]+)?)\s*' + units_pattern + r'\b)|(?:\b' + units_pattern + r'\s*([0-9]+(?:[.,][0-9]+)?))'
    matches = re.findall(pattern, ocr_text, flags=re.IGNORECASE)
    if matches:
        for match in matches:
            number_str = match[0] if match[0] else match[2]
            unit = match[1] if match[1] else match[3]
            if not number_str or not unit:
                continue
            number_str = number_str.replace(',', '.').replace(' ', '')
            try:
                number = float(number_str)
                unit = standardize_unit(unit)
                return f"{number} {unit}"
            except ValueError:
                continue
    return ""

# Function to process a single sample (for multiprocessing)
def process_row(row, images_dir):
    index = row['index']
    image_link = row['image_link']
    entity_name = row['entity_name']

    # Get image filename
    image_filename = get_image_filename(image_link)
    image_path = os.path.join(images_dir, image_filename)

    # Check if image exists
    if not os.path.exists(image_path):
        return index, ""

    # Load image
    try:
        image = Image.open(image_path)
        # Preprocess image
        image = preprocess_image(image)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return index, ""

    # Use OCR to extract text with custom configurations
    ocr_config = r'--oem 3 --psm 6'
    ocr_text = pytesseract.image_to_string(image, config=ocr_config)

    # Normalize OCR text
    ocr_text = ocr_text.replace('|', '1').replace('l', '1').replace('O', '0').replace('S', '5')

    # Extract entity value from OCR text
    predicted_value = extract_entity_value(ocr_text, entity_name)

    return index, predicted_value

# Function to process samples and extract predictions using multiprocessing
def process_samples(df, images_dir):
    num_processes = max(1, multiprocessing.cpu_count() - 2)  # Leave some cores free
    with multiprocessing.Pool(processes=num_processes) as pool:
        func = partial(process_row, images_dir=images_dir)
        results = list(tqdm(pool.imap(func, [row for _, row in df.iterrows()]), total=df.shape[0]))
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['index', 'prediction'])
    return results_df

# Main block to execute the script
if __name__ == '__main__':
    # Read the test data
    test_df = pd.read_csv(TEST_CSV_PATH)

    # Ensure 'index' column exists
    if 'index' not in test_df.columns:
        test_df.reset_index(inplace=True)

    # Process the test samples
    predictions_df = process_samples(test_df, IMAGES_DIR)

    # Merge predictions with the original indices
    output_df = test_df[['index']].merge(predictions_df, on='index', how='left')

    # Save to CSV
    output_df[['index', 'prediction']].to_csv(OUTPUT_CSV_PATH, index=False)
