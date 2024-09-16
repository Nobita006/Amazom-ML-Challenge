import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_predictions_and_labels(output_file, test_file):
    """
    Load predicted values and true labels from output and test CSV files.
    
    Args:
    - output_file (str): Path to the CSV file with predictions.
    - test_file (str): Path to the CSV file with true labels.
    
    Returns:
    - DataFrame: Combined DataFrame with 'index', 'predicted_value', and 'true_value' columns.
    """
    try:
        # Load predictions and test data
        output_df = pd.read_csv(output_file)
        test_df = pd.read_csv(test_file)

        # Merge predictions with true labels on the 'index' column
        merged_df = pd.merge(output_df, test_df[['index', 'entity_name', 'entity_value']], on='index')
        merged_df.rename(columns={'prediction': 'predicted_value', 'entity_value': 'true_value'}, inplace=True)

        # Convert predicted and true values to comparable formats
        merged_df['predicted_value_numeric'], merged_df['predicted_unit'] = zip(
            *merged_df['predicted_value'].apply(parse_string))
        merged_df['true_value_numeric'], merged_df['true_unit'] = zip(
            *merged_df['true_value'].apply(parse_string))

        # Keep only the rows where units match
        matched_df = merged_df[merged_df['predicted_unit'] == merged_df['true_unit']]
        
        logging.info("Predictions and true labels loaded and processed successfully.")
        return matched_df

    except Exception as e:
        logging.error(f"Error loading predictions and true labels: {e}")
        return None

def parse_string(s):
    """
    Parse a string in the format "x unit" to extract numerical value and unit.
    
    Args:
    - s (str): String to parse.
    
    Returns:
    - (float, str): Tuple containing the numerical value and unit.
    """
    try:
        parts = s.split(maxsplit=1)
        number = float(parts[0])
        unit = parts[1].strip().lower()
        return number, unit
    except Exception as e:
        logging.error(f"Error parsing string '{s}': {e}")
        return None, None

def calculate_metrics(df):
    """
    Calculate evaluation metrics: MAE, MSE, RMSE, R^2 Score.
    
    Args:
    - df (DataFrame): DataFrame containing 'predicted_value_numeric' and 'true_value_numeric' columns.
    
    Returns:
    - dict: Dictionary containing calculated metrics.
    """
    try:
        y_true = df['true_value_numeric']
        y_pred = df['predicted_value_numeric']

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'Mean Absolute Error (MAE)': mae,
            'Mean Squared Error (MSE)': mse,
            'Root Mean Squared Error (RMSE)': rmse,
            'R^2 Score': r2
        }

        logging.info(f"Calculated metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return None

def visualize_errors(df):
    """
    Visualize prediction errors using scatter plot and histogram.
    
    Args:
    - df (DataFrame): DataFrame containing 'true_value_numeric' and 'predicted_value_numeric' columns.
    """
    try:
        # Scatter plot for true vs predicted values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='true_value_numeric', y='predicted_value_numeric', data=df)
        plt.plot([df['true_value_numeric'].min(), df['true_value_numeric'].max()],
                 [df['true_value_numeric'].min(), df['true_value_numeric'].max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')
        plt.show()

        # Histogram of errors
        errors = df['true_value_numeric'] - df['predicted_value_numeric']
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=50, kde=True)
        plt.xlabel('Prediction Error')
        plt.title('Distribution of Prediction Errors')
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing errors: {e}")

if __name__ == "__main__":
    # Define paths to files
    DATASET_FOLDER = 'dataset'
    output_file = os.path.join(DATASET_FOLDER, 'sample-test_out.csv')
    test_file = os.path.join(DATASET_FOLDER, 'sample_test.csv')

    # Load predictions and true values
    df = load_predictions_and_labels(output_file, test_file)

    if df is not None and not df.empty:
        # Calculate evaluation metrics
        metrics = calculate_metrics(df)

        # Visualize errors
        visualize_errors(df)
    else:
        logging.error("Failed to load data for evaluation or no matching data available.")
