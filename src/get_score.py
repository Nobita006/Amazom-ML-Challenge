import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from PIL import Image
from urllib.parse import urlparse
import string

# Load entity-unit mappings with abbreviations and plural forms
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# Function to extract the image filename from the image link
def get_image_filename(image_link):
    parsed_url = urlparse(image_link)
    return os.path.basename(parsed_url.path)

# Function to preprocess images with aspect ratio preservation and padding
def preprocess_image(image_path, image_size=250):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize while keeping aspect ratio
    img.thumbnail((image_size, image_size), Image.LANCZOS)
    
    # Create a new square image with a black background
    new_img = Image.new('L', (image_size, image_size), color=0)
    
    # Paste the resized image onto the center of the new square image
    new_img.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
    
    img = np.array(new_img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# Encode text into numerical representation for training
def encode_text(text):
    try:
        number, unit = text.split()
        unit_index = list(entity_unit_map.keys()).index(unit)
        # Return a one-hot encoded vector for the unit and number
        one_hot = np.zeros(len(entity_unit_map.keys()))
        one_hot[unit_index] = 1
        return [float(number)] + one_hot.tolist()
    except (ValueError, IndexError) as e:
        # If parsing fails, return zeros
        return [0.0] + [0] * len(entity_unit_map.keys())

# Decode numerical representation back to text
def decode_text(prediction):
    # If prediction is a 1D array, reshape it to 2D for consistent processing
    if len(prediction.shape) == 1:
        prediction = np.expand_dims(prediction, axis=0)
    
    if len(prediction[0]) == 1 + len(entity_unit_map.keys()):  # Ensure the prediction length is valid
        number = prediction[0][0]  # Get the predicted number
        unit_idx = np.argmax(prediction[0][1:])  # Get the index of the predicted unit
        unit = list(entity_unit_map.keys())[unit_idx]  # Map the index back to the unit
        return f"{number:.2f} {unit}"
    else:
        return ""  # Return empty if the prediction is invalid

# Load test data
def load_test_data(csv_path, images_dir, start_index=0, num_files=50):
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df = df.iloc[start_index:start_index + num_files]  # Select the specified range

    images = []
    labels = []

    for idx, row in df.iterrows():
        image_path = os.path.join(images_dir, get_image_filename(row['image_link']))
        if os.path.exists(image_path):
            try:
                img = preprocess_image(image_path)
                images.append(img)
                labels.append(encode_text(row['entity_value']))  # Encode labels
            except (OSError, IOError) as e:
                # Skip images that are truncated or have issues
                print(f"Skipping file {image_path}: {e}")
                continue

    return np.array(images), np.array(labels)

# Evaluate Model and Calculate F1 Score
def evaluate_model(model_path, csv_path, images_dir, start_index=0, num_files=50):
    # Load model using the SavedModel format
    model = tf.keras.models.load_model(model_path)

    # Load test data
    val_images, val_labels = load_test_data(csv_path, images_dir, start_index, num_files)

    # Predict on test data
    predictions = model.predict(val_images)
    predicted_texts = [decode_text(pred) for pred in predictions]

    # Convert true labels to their original format
    true_texts = [decode_text(true) for true in val_labels]

    # Calculate F1 Score
    f1 = f1_score(true_texts, predicted_texts, average='weighted', zero_division=1)
    precision = precision_score(true_texts, predicted_texts, average='weighted', zero_division=1)
    recall = recall_score(true_texts, predicted_texts, average='weighted', zero_division=1)

    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Example usage
if __name__ == "__main__":
    model_path = 'ocr_model.keras'  # Path to your saved model
    csv_path = 'dataset/cleaned_train2.csv'  # Path to the CSV file
    images_dir = 'images/train'  # Directory containing images
    start_index = 110000  # Starting index of the CSV data to test
    num_files = 1000  # Number of files to test

    evaluate_model(model_path, csv_path, images_dir, start_index, num_files)
