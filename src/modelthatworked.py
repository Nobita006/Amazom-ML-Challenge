import os
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from tqdm import tqdm
import pickle
from urllib.parse import urlparse
import multiprocessing
from functools import partial
import cv2
import tempfile

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Load entity-unit mappings with abbreviations and plural forms
entity_unit_map = {
    'width': {'centimetre', 'cm', 'centimeter', 'centimeters', 'foot', 'ft', 'feet', 'inch', 'in', 'inches', 'metre', 'm',
              'meter', 'meters', 'millimetre', 'mm', 'millimeter', 'millimeters', 'yard', 'yd', 'yards'},
    'depth': {'centimetre', 'cm', 'centimeter', 'centimeters', 'foot', 'ft', 'feet', 'inch', 'in', 'inches', 'metre', 'm',
              'meter', 'meters', 'millimetre', 'mm', 'millimeter', 'millimeters', 'yard', 'yd', 'yards'},
    'height': {'centimetre', 'cm', 'centimeter', 'centimeters', 'foot', 'ft', 'feet', 'inch', 'in', 'inches', 'metre', 'm',
               'meter', 'meters', 'millimetre', 'mm', 'millimeter', 'millimeters', 'yard', 'yd', 'yards'},
    'item_weight': {'gram', 'g', 'grams', 'kilogram', 'kg', 'kilograms', 'microgram', 'µg', 'micrograms', 'milligram',
                    'mg', 'milligrams', 'ounce', 'oz', 'ounces', 'pound', 'lb', 'pounds', 'ton', 'tons'},
    'maximum_weight_recommendation': {'gram', 'g', 'grams', 'kilogram', 'kg', 'kilograms', 'microgram',
                                      'µg', 'micrograms', 'milligram', 'mg', 'milligrams', 'ounce', 'oz',
                                      'ounces', 'pound', 'lb', 'pounds', 'ton', 'tons'},
    'voltage': {'kilovolt', 'kv', 'kilovolts', 'millivolt', 'mv', 'millivolts', 'volt', 'v', 'volts'},
    'wattage': {'kilowatt', 'kw', 'kilowatts', 'watt', 'w', 'watts'},
    'item_volume': {'centilitre', 'cl', 'centilitres', 'cubic foot', 'ft3', 'cubic feet', 'cubic inch', 'in3',
                    'cubic inches', 'cup', 'c', 'cups', 'decilitre', 'dl', 'decilitres', 'fluid ounce', 'fl oz',
                    'fluid ounces', 'gallon', 'gal', 'gallons', 'litre', 'l', 'litres', 'microlitre', 'µl',
                    'microlitres', 'millilitre', 'ml', 'millilitres', 'pint', 'pt', 'pints', 'quart', 'qt', 'quarts'}
}

# Set the number of samples to use
NUM_SAMPLES = 10000  # Change this value as needed

# Paths to data and images
TRAIN_CSV_PATH = 'dataset/cleaned_train2.csv'
IMAGES_DIR = 'images/train'
MODEL_SAVE_PATH = 'saved_models/ocr_model3.pkl'

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
        'kilograms': 'kilogram',
        'lbs': 'pound',
        'lb': 'pound',
        'pounds': 'pound',
        'g': 'gram',
        'grams': 'gram',
        'mg': 'milligram',
        'milligrams': 'milligram',
        'µg': 'microgram',
        'micrograms': 'microgram',
        'oz': 'ounce',
        'ounces': 'ounce',
        'cm': 'centimetre',
        'centimeters': 'centimetre',
        'mm': 'millimetre',
        'millimeters': 'millimetre',
        'm': 'metre',
        'meters': 'metre',
        'in': 'inch',
        'inches': 'inch',
        'ft': 'foot',
        'feet': 'foot',
        'yd': 'yard',
        'yards': 'yard',
        'kv': 'kilovolt',
        'kilovolts': 'kilovolt',
        'mv': 'millivolt',
        'millivolts': 'millivolt',
        'v': 'volt',
        'volts': 'volt',
        'w': 'watt',
        'watts': 'watt',
        'kw': 'kilowatt',
        'kilowatts': 'kilowatt',
        'l': 'litre',
        'litres': 'litre',
        'ml': 'millilitre',
        'millilitres': 'millilitre',
        'cl': 'centilitre',
        'centilitres': 'centilitre',
        'dl': 'decilitre',
        'decilitres': 'decilitre',
        'µl': 'microlitre',
        'microlitres': 'microlitre',
        'pt': 'pint',
        'pints': 'pint',
        'qt': 'quart',
        'quarts': 'quart',
        'gal': 'gallon',
        'gallons': 'gallon',
        'c': 'cup',
        'cups': 'cup',
        # Add more mappings as needed
    }
    return unit_mapping.get(unit.lower(), unit.lower())

def preprocess_image(image):
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)

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
    # Create units pattern with capturing group
    units_pattern = '(' + '|'.join(re.escape(unit) for unit in allowed_units) + ')'
    # Regex pattern to find numbers followed or preceded by units
    pattern = r'([0-9]+(?:[.,][0-9]+)?)\s*(?:' + units_pattern + r')\b|\b' + units_pattern + r'\s*([0-9]+(?:[.,][0-9]+)?)'
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
    image_link = row['image_link']
    entity_name = row['entity_name']

    # Get image filename
    image_filename = get_image_filename(image_link)
    image_path = os.path.join(images_dir, image_filename)

    # Check if image exists
    if not os.path.exists(image_path):
        return ""

    # Load image
    try:
        image = Image.open(image_path)
        # Preprocess image
        image = preprocess_image(image)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return ""

    # Use OCR to extract text with custom configurations
    ocr_config = r'--oem 3 --psm 12'
    ocr_text = pytesseract.image_to_string(image, config=ocr_config)

    # Normalize OCR text
    ocr_text = ocr_text.replace('|', '1').replace('l', '1').replace('O', '0').replace('S', '5')

    # Extract entity value from OCR text
    predicted_value = extract_entity_value(ocr_text, entity_name)

    return predicted_value

# Function to process samples and extract predictions using multiprocessing
def process_samples(df, images_dir):
    num_processes = max(1, multiprocessing.cpu_count() - 2)  # Leave some cores free
    with multiprocessing.Pool(processes=num_processes) as pool:
        func = partial(process_row, images_dir=images_dir)
        predictions = list(tqdm(pool.imap(func, [row for _, row in df.iterrows()]), total=df.shape[0]))
    return predictions

# Function to compute evaluation metrics
def compute_f1_score(df):
    y_true = df['entity_value'].fillna('').str.strip().str.lower()
    y_pred = df['predicted_entity_value'].fillna('').str.strip().str.lower()
    TP = FP = FN = TN = 0
    for true, pred in zip(y_true, y_pred):
        if pred != '' and true != '':
            if pred == true:
                TP += 1
            else:
                FP += 1
        elif pred != '' and true == '':
            FP += 1
        elif pred == '' and true != '':
            FN += 1
        elif pred == '' and true == '':
            TN += 1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Main block to execute the script
if __name__ == '__main__':
    # Read the training data
    df = pd.read_csv(TRAIN_CSV_PATH)
    df = df.head(NUM_SAMPLES)  # Use only NUM_SAMPLES entries

    # Process the training samples
    df['predicted_entity_value'] = process_samples(df, IMAGES_DIR)

    # Compute metrics on training data
    precision, recall, f1 = compute_f1_score(df)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Save the OCR model components (if any) to the saved_models folder
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump({
            'entity_unit_map': entity_unit_map,
        }, f)