import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import logging

# Import constants from the correct location
import constants  # Ensure constants.py is in the same directory or adjust the import path accordingly

# Constants
IMAGE_SIZE = (224, 224)
MODEL_PATH = r'saved_models/entity_value_predictor.keras'  # Use raw string to avoid invalid escape sequences
ALLOWED_UNITS = constants.allowed_units

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_image(image_path):
    """
    Load and preprocess the image for prediction.
    
    Args:
    - image_path (str): Path to the image.
    
    Returns:
    - np.array: Preprocessed image ready for prediction.
    """
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        logging.error(f"Error loading or preprocessing image {image_path}: {e}")
        return None

def load_trained_model(model_path):
    """
    Load the trained model from the specified path.
    
    Args:
    - model_path (str): Path to the saved Keras model.
    
    Returns:
    - Model: Loaded Keras model.
    """
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None

def format_prediction(value, entity_name):
    """
    Format the prediction value as "x unit" where unit is based on the entity name.
    
    Args:
    - value (float): Predicted numerical value.
    - entity_name (str): The name of the entity to determine the unit.
    
    Returns:
    - str: Formatted prediction.
    """
    # Determine unit based on entity name
    unit = "centimetre"  # Default unit
    if entity_name in constants.entity_unit_map:
        possible_units = constants.entity_unit_map[entity_name]
        unit = np.random.choice(list(possible_units))  # Randomly choose a valid unit for now
    
    return f"{value:.2f} {unit}"

def predictor(model, image_link, category_id, entity_name):
    """
    Predict the entity value for a given image using the trained model.
    
    Args:
    - model (Model): Trained Keras model.
    - image_link (str): URL of the image to predict.
    - category_id (int): ID of the category.
    - entity_name (str): The entity name to predict.
    
    Returns:
    - str: Predicted value in the format "x unit".
    """
    image_path = os.path.join('images/test', os.path.basename(image_link))  # Assume images are downloaded to 'images/test' folder
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return ""  # Return empty if image is missing

    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        return ""  # Return empty if image loading or preprocessing fails

    # Make prediction using the model
    prediction = model.predict(img_array)
    predicted_value = prediction[0][0]  # Get the predicted value from the model output

    # Format prediction in the required "x unit" format
    return format_prediction(predicted_value, entity_name)

if __name__ == "__main__":
    # Load the trained model
    model = load_trained_model(MODEL_PATH)

    if model is None:
        logging.error("Failed to load model. Exiting...")
        exit(1)

    # Load test dataset
    DATASET_FOLDER = 'dataset'
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

    # Predict and generate output
    test['prediction'] = test.apply(
        lambda row: predictor(model, row['image_link'], row['group_id'], row['entity_name']), axis=1
    )

    # Save the predictions to a CSV file
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
    logging.info(f"Predictions saved to {output_filename}.")
