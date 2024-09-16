import os
import pandas as pd
import numpy as np
# Choose between Tesseract and EasyOCR
USE_EASYOCR = True

if USE_EASYOCR:
    import easyocr
else:
    import pytesseract
    # Set the path to the Tesseract executable
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

from PIL import Image, ImageEnhance, ImageFilter
import re
from tqdm import tqdm
import pickle
from urllib.parse import urlparse
import multiprocessing
from functools import partial
import cv2
import spacy
from pint import UnitRegistry

# Load spaCy model for NER
nlp = spacy.load('en_core_web_sm')

# Initialize EasyOCR reader if used
if USE_EASYOCR:
    reader = easyocr.Reader(['en'])

# Initialize Unit Registry
ureg = UnitRegistry()

# Load entity-unit mappings with abbreviations and plural forms
entity_unit_map = {
    'width': {'centimetre', 'cm', 'centimeter', 'centimeters', 'foot', 'ft', 'feet', 'inch', 'in', 'inches',
              'metre', 'm', 'meter', 'meters', 'millimetre', 'mm', 'millimeter', 'millimeters', 'yard',
              'yd', 'yards', '″', '"', '’', "'"},
    'depth': {'centimetre', 'cm', 'centimeter', 'centimeters', 'foot', 'ft', 'feet', 'inch', 'in', 'inches',
              'metre', 'm', 'meter', 'meters', 'millimetre', 'mm', 'millimeter', 'millimeters', 'yard',
              'yd', 'yards', '″', '"', '’', "'"},
    'height': {'centimetre', 'cm', 'centimeter', 'centimeters', 'foot', 'ft', 'feet', 'inch', 'in', 'inches',
               'metre', 'm', 'meter', 'meters', 'millimetre', 'mm', 'millimeter', 'millimeters', 'yard',
               'yd', 'yards', '″', '"', '’', "'"},
    'item_weight': {'gram', 'g', 'grams', 'kilogram', 'kg', 'kilograms', 'microgram', 'µg', 'micrograms',
                    'milligram', 'mg', 'milligrams', 'ounce', 'oz', 'ounces', 'pound', 'lb', 'pounds',
                    'ton', 'tons'},
    'maximum_weight_recommendation': {'gram', 'g', 'grams', 'kilogram', 'kg', 'kilograms', 'microgram',
                                      'µg', 'micrograms', 'milligram', 'mg', 'milligrams', 'ounce', 'oz',
                                      'ounces', 'pound', 'lb', 'pounds', 'ton', 'tons'},
    'voltage': {'kilovolt', 'kv', 'kilovolts', 'millivolt', 'mv', 'millivolts', 'volt', 'v', 'volts'},
    'wattage': {'kilowatt', 'kw', 'kilowatts', 'watt', 'w', 'watts'},
    'item_volume': {'centilitre', 'cl', 'centilitres', 'cubic foot', 'ft3', 'cubic feet', 'cubic inch', 'in3',
                    'cubic inches', 'cup', 'c', 'cups', 'decilitre', 'dl', 'decilitres', 'fluid ounce',
                    'fl oz', 'fluid ounces', 'gallon', 'gal', 'gallons', 'litre', 'l', 'litres', 'microlitre',
                    'µl', 'microlitres', 'millilitre', 'ml', 'millilitres', 'pint', 'pt', 'pints', 'quart',
                    'qt', 'quarts'}
}

# Entity keywords to improve matching
entity_keywords = {
    'width': ['width', 'w'],
    'height': ['height', 'h'],
    'depth': ['depth', 'd'],
    'item_weight': ['weight', 'wt'],
    'maximum_weight_recommendation': ['max weight', 'maximum weight'],
    'voltage': ['voltage', 'volt'],
    'wattage': ['wattage', 'watt'],
    'item_volume': ['volume', 'capacity'],
}

# Set the number of samples to use
NUM_SAMPLES = 1000  # Change this value as needed

# Paths to data and images
TRAIN_CSV_PATH = 'dataset/cleaned_train2.csv'
IMAGES_DIR = 'images/train'
MODEL_SAVE_PATH = 'saved_models/ocr_model_updated.pkl'

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
        '″': 'inch',
        '"': 'inch',
        '’': 'foot',
        "'": 'foot',
        # Add more mappings as needed
    }
    return unit_mapping.get(unit.lower(), unit.lower())

# Function to normalize numbers (handle fractions)
def normalize_number(number_str):
    number_str = number_str.replace(',', '.').replace(' ', '')
    if '/' in number_str:
        try:
            num, denom = number_str.split('/')
            return float(num) / float(denom)
        except:
            return None
    else:
        try:
            return float(number_str)
        except:
            return None

# Enhanced image preprocessing
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize the image based on its dimensions
    # if image.width < 800:
    #     new_width = int(image.width * 1.5)
    #     new_height = int(image.height * 1.5)
    #     image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    
    # Convert to numpy array for OpenCV processing
    # image_np = np.array(image)
    
    # Apply adaptive thresholding
    # image_np = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    #  cv2.THRESH_BINARY, 31, 2)
    
    # Apply Gaussian blur to reduce noise
    # image_np = cv2.GaussianBlur(image_np, (3, 3), 0)
    
    # Convert back to PIL Image
    # image = Image.fromarray(image_np)
    
    return image

# Cache for OCR results
ocr_cache = {}

# Function to extract entity value from OCR text
def extract_entity_value(ocr_text, entity_name):
    allowed_units = entity_unit_map.get(entity_name, set())
    if not allowed_units:
        return ""
    
    # Include entity keywords in regex
    keywords = entity_keywords.get(entity_name, [])
    keywords_pattern = '|'.join(re.escape(keyword) for keyword in keywords)
    
    # Create units pattern with capturing group
    units_pattern = '(' + '|'.join(re.escape(unit) for unit in allowed_units) + ')'
    
    # Regex pattern to find numbers with units
    pattern = rf'''
        (?:(?:{keywords_pattern})\s*[:\-]?\s*)?
        (?:
            ([0-9]+(?:[.,][0-9]+)?|\d+/\d+)\s*{units_pattern}\b   # Number followed by unit
        )|
        (?:
            \b{units_pattern}\s*([0-9]+(?:[.,][0-9]+)?|\d+/\d+)   # Unit followed by number
        )
    '''
    matches = re.findall(pattern, ocr_text, flags=re.IGNORECASE | re.VERBOSE)
    if matches:
        for match in matches:
            number_str = match[0] if match[0] else match[2]
            unit = match[1] if match[1] else match[3]
            if not number_str or not unit:
                continue
            number = normalize_number(number_str)
            if number is None:
                continue
            unit = standardize_unit(unit)
            # Optionally convert units to standard units
            # number, unit = convert_to_standard(number, unit, entity_name)
            return f"{number} {unit}"
    return ""

# Optional function to convert units to standard units
def convert_to_standard(number, unit, entity_name):
    try:
        quantity = number * ureg(unit)
        # Define standard units per entity
        standard_units = {
            'width': ureg('centimeter'),
            'height': ureg('centimeter'),
            'depth': ureg('centimeter'),
            'item_weight': ureg('gram'),
            'maximum_weight_recommendation': ureg('gram'),
            'voltage': ureg('volt'),
            'wattage': ureg('watt'),
            'item_volume': ureg('milliliter'),
        }
        standard_unit = standard_units.get(entity_name)
        if standard_unit:
            quantity = quantity.to(standard_unit)
            return quantity.magnitude, str(standard_unit)
    except:
        pass
    return number, unit

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

    # Use cached OCR result if available
    if image_path in ocr_cache:
        ocr_text = ocr_cache[image_path]
    else:
        # Load image
        try:
            image = Image.open(image_path)
            # Preprocess image
            image = preprocess_image(image)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return ""

        # Perform OCR
        if USE_EASYOCR:
            image_np = np.array(image)
            result = reader.readtext(image_np)
            ocr_text = ' '.join([res[1] for res in result])
        else:
            ocr_config = r'--oem 3 --psm 6'
            ocr_text = pytesseract.image_to_string(image, config=ocr_config)
        
        # Normalize OCR text
        ocr_text = ocr_text.replace('|', '1').replace('l', '1').replace('I', '1').replace('O', '0')\
                           .replace('S', '5').replace('B', '8')
        ocr_cache[image_path] = ocr_text

    # Extract entity value from OCR text
    predicted_value = extract_entity_value(ocr_text, entity_name)

    # If prediction is still empty, try NER
    if not predicted_value:
        doc = nlp(ocr_text)
        for ent in doc.ents:
            if ent.label_ == 'QUANTITY':
                # Attempt to parse the quantity
                text = ent.text
                match = re.match(r'([0-9]+(?:[.,][0-9]+)?|\d+/\d+)\s*(\w+)', text)
                if match:
                    number_str, unit = match.groups()
                    number = normalize_number(number_str)
                    if number is None:
                        continue
                    unit = standardize_unit(unit)
                    if unit in entity_unit_map.get(entity_name, set()):
                        predicted_value = f"{number} {unit}"
                        break

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
            'entity_keywords': entity_keywords,
        }, f)
