import tensorflow as tf

# Check if GPU is available and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU instead")


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np
import os
import re

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Adjust as needed for large dataset
LEARNING_RATE = 0.001
CLEANED_TRAIN_CSV = 'dataset/cleaned_train.csv'
TRAIN_DATA_PATH = 'images/train'
MODEL_SAVE_PATH = 'saved_models/entity_value_predictor.keras'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path, image_folder, max_images=None):
    """
    Load and preprocess CSV data for training, with an option to limit the number of images used.
    
    Args:
    - file_path (str): Path to the CSV file containing data.
    - image_folder (str): Folder path where images are stored.
    - max_images (int, optional): Maximum number of images to use for training. Default is None (use all images).
    
    Returns:
    - DataFrame: Loaded DataFrame with corrected image links and numeric entity values.
    """
    df = pd.read_csv(file_path)
    
    # Extract filenames from URLs and check if they exist in the directory
    df['image_link'] = df['image_link'].apply(lambda x: os.path.basename(x))  # Extract the filename
    df = df[df['image_link'].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))]  # Check if file exists

    # Extract numerical value from 'entity_value' column
    df['numeric_value'] = df['entity_value'].apply(extract_numeric_value)
    
    # Limit the number of images if max_images is specified
    if max_images is not None:
        df = df.sample(n=max_images, random_state=42).reset_index(drop=True)
    
    logging.info(f"Loaded and filtered {len(df)} entries from {file_path}.")
    return df

def extract_numeric_value(entity_value):
    """
    Extracts the numeric part from the entity_value string.
    
    Args:
    - entity_value (str): Entity value string (e.g., "1.0 kilogram").
    
    Returns:
    - float: The numeric value extracted from the string.
    """
    match = re.match(r'^(\d+(\.\d+)?)\s+[a-zA-Z\s]+$', entity_value)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Invalid entity value format: {entity_value}")

def preprocess_data(df):
    """
    Preprocess the data for model training.
    
    Args:
    - df (DataFrame): The DataFrame containing the data.
    
    Returns:
    - train_df (DataFrame): The training data DataFrame.
    - val_df (DataFrame): The validation data DataFrame.
    - train_gen (DataFrameIterator): The training data generator.
    - val_gen (DataFrameIterator): The validation data generator.
    - steps_per_epoch (int): Number of steps per epoch.
    - validation_steps (int): Number of validation steps.
    """
    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Image data generator with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Image data generator for validation (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=TRAIN_DATA_PATH,
        x_col='image_link',
        y_col='numeric_value',  # Use the numeric value for regression
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='raw'  # 'raw' mode for regression
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        directory=TRAIN_DATA_PATH,
        x_col='image_link',
        y_col='numeric_value',  # Use the numeric value for regression
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='raw'
    )
    
    # Calculate steps per epoch dynamically
    steps_per_epoch = np.ceil(train_gen.samples / BATCH_SIZE).astype(int)
    validation_steps = np.ceil(val_gen.samples / BATCH_SIZE).astype(int)

    return train_df, val_df, train_gen, val_gen, steps_per_epoch, validation_steps

def build_model(input_shape=(224, 224, 3)):
    """
    Build and compile the Keras model.
    
    Args:
    - input_shape (tuple): The input shape of the images.
    
    Returns:
    - model (Model): The compiled Keras model.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='linear')(x)  # Output layer for regression

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the layers except the last few
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error', metrics=['mae'])
    logging.info("Model built and compiled successfully.")
    return model

def train_model(model, train_gen, val_gen, steps_per_epoch, validation_steps):
    """
    Train the model on the provided training and validation data.
    
    Args:
    - model (Model): The Keras model to train.
    - train_gen (DataFrameIterator): The training data generator.
    - val_gen (DataFrameIterator): The validation data generator.
    - steps_per_epoch (int): Number of steps per epoch.
    - validation_steps (int): Number of validation steps.
    
    Returns:
    - History: The training history.
    """
    logging.info("Starting model training...")
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    tensorboard = TensorBoard(log_dir='./logs')

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard]
    )
    
    logging.info("Model training completed.")
    return history

if __name__ == "__main__":
    # Define the maximum number of images to use
    MAX_IMAGES = 20000  # Change this to the desired number of images

    # Load and preprocess the data
    df = load_data(CLEANED_TRAIN_CSV, TRAIN_DATA_PATH, max_images=MAX_IMAGES)
    train_df, val_df, train_gen, val_gen, steps_per_epoch, validation_steps = preprocess_data(df)

    # Build the model
    model = build_model(input_shape=(*IMAGE_SIZE, 3))

    # Train the model
    history = train_model(model, train_gen, val_gen, steps_per_epoch, validation_steps)

    # Save the trained model
    save_model(model, MODEL_SAVE_PATH)
