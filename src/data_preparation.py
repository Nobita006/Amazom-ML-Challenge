import pandas as pd
import os
import re
import constants
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from utils import common_mistake, validate_image  # Import validate_image function from utils.py

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path, sample_size=None):
    """
    Load CSV data from the given file path with an optional sampling size.
    
    Args:
    - file_path (str): Path to the CSV file.
    - sample_size (int, optional): Number of samples to load from the file. Default is None (load all data).
    
    Returns:
    - DataFrame: Loaded DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        if sample_size is not None:
            data = data.sample(n=sample_size, random_state=42)  # Sample a subset for quicker processing
            logging.info(f"Loaded a sample of {sample_size} records from {file_path}.")
        else:
            logging.info(f"Data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def explore_data(df, dataset_name="dataset"):
    """
    Explore the data to understand the distribution of entity names and values.
    
    Args:
    - df (DataFrame): The DataFrame to explore.
    - dataset_name (str): Name of the dataset for labeling in plots and logs.
    """
    logging.info(f"Exploring {dataset_name}...")

    # Summary of the data
    logging.info(f"{dataset_name} shape: {df.shape}")
    logging.info(f"{dataset_name} columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    logging.info(f"Missing values in {dataset_name}:\n{missing_values}")
    
    # Distribution of entity_name
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='entity_name', order=df['entity_name'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Distribution of Entity Names in {dataset_name}')
    plt.show()

    # Distribution of entity_value (only for training data)
    if 'entity_value' in df.columns:
        plt.figure(figsize=(12, 6))
        df['entity_value'].value_counts().nlargest(20).plot(kind='bar')
        plt.title(f'Top 20 Most Frequent Entity Values in {dataset_name}')
        plt.show()

def clean_and_standardize_entity_value(entity_value):
    """
    Clean and standardize the entity_value to a consistent format (e.g., "x unit").
    
    Args:
    - entity_value (str): The entity value to standardize.
    
    Returns:
    - str: Standardized entity value.
    """
    if pd.isna(entity_value) or entity_value.strip() == "":
        logging.warning(f"Empty or NaN entity_value encountered: {entity_value}")
        return ""

    # Handle range cases such as "[800.0, 850.0] gram" by taking the average value
    range_pattern = re.compile(r'^\[(\d+\.?\d*),\s*(\d+\.?\d*)\]\s+([a-zA-Z\s]+)$')
    match = range_pattern.match(entity_value)
    if match:
        min_val, max_val, unit = match.groups()
        try:
            number = (float(min_val) + float(max_val)) / 2  # Taking the average
            unit = common_mistake(unit.strip().lower())
            if unit in constants.allowed_units:
                return f"{number} {unit}"
            else:
                logging.warning(f"Invalid unit in entity_value after range processing: {entity_value}")
                return ""
        except ValueError as ve:
            logging.error(f"Error processing range in entity_value: {entity_value}. Error: {ve}")
            return ""

    # Handle formats like "0 kilogram to 15 kilogram" by using the maximum value
    range_to_pattern = re.compile(r'^\d+\.?\d*\s+[a-zA-Z]+\s+to\s+(\d+\.?\d*)\s+([a-zA-Z\s]+)$')
    match = range_to_pattern.match(entity_value)
    if match:
        max_val, unit = match.groups()
        try:
            unit = common_mistake(unit.strip().lower())
            if unit in constants.allowed_units:
                return f"{max_val} {unit}"
            else:
                logging.warning(f"Invalid unit in entity_value after range-to processing: {entity_value}")
                return ""
        except ValueError as ve:
            logging.error(f"Error processing range-to in entity_value: {entity_value}. Error: {ve}")
            return ""

    # Standard pattern matching "number unit"
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(entity_value):
        logging.warning(f"Invalid entity_value format: {entity_value}")
        return ""

    # Split the entity value into number and unit
    parts = entity_value.split(maxsplit=1)
    try:
        number = float(parts[0])
    except ValueError:
        logging.warning(f"Invalid number in entity_value: {entity_value}")
        return ""
    
    unit = parts[1].strip().lower()
    unit = common_mistake(unit)
    
    # Check if the unit is allowed
    if unit not in constants.allowed_units:
        logging.warning(f"Invalid unit in entity_value after standard processing: {entity_value}")
        return ""

    # Return the standardized entity value
    return f"{number} {unit}"

def clean_data(df, image_folder):
    """
    Clean and standardize the dataset, and filter out rows with missing or invalid images.
    
    Args:
    - df (DataFrame): The DataFrame to clean.
    - image_folder (str): Directory containing images.
    
    Returns:
    - DataFrame: Cleaned and standardized DataFrame.
    """
    logging.info("Cleaning data...")

    # Standardize the entity_value column
    df['entity_value'] = df['entity_value'].apply(clean_and_standardize_entity_value)

    # Drop rows where entity_value is empty after standardization
    initial_shape = df.shape
    df = df[df['entity_value'] != ""]
    logging.info(f"Removed {initial_shape[0] - df.shape[0]} rows with invalid entity values.")
    
    # Validate images and filter out invalid ones
    df['filename'] = df['image_link'].apply(lambda x: os.path.basename(x))  # Extract the filename
    df['valid_image'] = df['filename'].apply(lambda x: validate_image(os.path.join(image_folder, x)))
    df = df[df['valid_image']]
    logging.info(f"Removed {initial_shape[0] - df.shape[0]} rows with invalid or missing images.")

    # Drop helper columns
    df = df.drop(columns=['filename', 'valid_image'])
    
    return df

def save_cleaned_data(df, file_path):
    """
    Save the cleaned DataFrame to a CSV file.
    
    Args:
    - df (DataFrame): The cleaned DataFrame.
    - file_path (str): Path to save the CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Cleaned data saved to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving cleaned data to {file_path}: {e}")

if __name__ == "__main__":
    # Load the data
    train_file_path = 'dataset/train.csv'  # Original train CSV path
    cleaned_data_file_path = 'dataset/cleaned_train2.csv'  # Cleaned train CSV path
    train_image_folder = 'images/train'  # Directory containing train images

    # Load and explore train data
    train_df = load_data(train_file_path)
    if train_df is not None:
        explore_data(train_df, dataset_name="Training Data")
        cleaned_train_df = clean_data(train_df, train_image_folder)
        save_cleaned_data(cleaned_train_df, cleaned_data_file_path)
