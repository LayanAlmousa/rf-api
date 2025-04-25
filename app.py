from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import pandas as pd
import numpy as np
import logging

from utils import (
    preprocess_gsr_dataset,  # Adjusted method name
    segment_gsr_data,        # Adjusted method name
    extract_features_matrix_optimized
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the ONNX model
session = ort.InferenceSession("random_forest.onnx")

def clean_gsr_data(df: pd.DataFrame, gsr_col: str) -> pd.Series:
    """
    Cleans the GSR data by removing non-numeric values from the specified column.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing GSR data.
        gsr_col (str): The name of the column containing the GSR data.
        
    Returns:
        pd.Series: Cleaned GSR data (only numeric values).
    """
    # Convert strings to float safely, using `errors='coerce'` to turn invalid values into NaNs
    gsr_series = pd.to_numeric(df[gsr_col], errors='coerce')

    # Log the state after conversion
    logger.debug(f"GSR data after conversion: {gsr_series.head()}")

    # Drop any NaN values
    gsr_series = gsr_series.dropna()
    
    # Log the cleaned GSR data after dropping NaN values
    logger.debug(f"Cleaned GSR data after dropping NaN: {gsr_series.head()}")
    
    return gsr_series

@app.route('/predict-session', methods=['POST'])
def predict_session():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            logger.debug("No file uploaded.")
            return jsonify({'error': 'No file uploaded'}), 400

        # Read uploaded CSV (tab-separated with a header skip)
        file = request.files['file']
        df = pd.read_csv(file, sep="\t", header=2)

        # Clean and rename the columns
        df.columns = df.columns.str.strip()
        df.rename(columns={'uS': 'GSR'}, inplace=True)
    
        # Log the first few rows of the dataframe
        logger.debug(f"DataFrame loaded: {df.head()}")

        # Use only numeric GSR data from the 'GSR' column
        gsr_col = 'GSR'
        if gsr_col not in df.columns:
            logger.debug(f"Expected column {gsr_col} not found.")
            return jsonify({'error': f'Expected column "{gsr_col}" not found'}), 400

        # Clean the GSR data
        gsr_series = clean_gsr_data(df, gsr_col)

        # Log the cleaned GSR data
        logger.debug(f"Cleaned GSR data: {gsr_series.head()}")

        if gsr_series.empty:
            logger.debug("No valid numeric GSR values found after cleaning.")
            return jsonify({'error': 'No valid numeric GSR values found'}), 400

        # Create a dummy DataFrame for input
        raw_df = pd.DataFrame({
            'Subject': ['Sample'],
            'GSR_Data': [gsr_series],
            'Label': ['unknown']
        })

        # Preprocess (tonic & phasic decomposition)
        logger.debug("Starting preprocessing of GSR signal.")
        clean_data = preprocess_gsr_dataset(raw_df, fs=256)
        
        # Log the preprocessed signal
        logger.debug(f"Preprocessed data: {clean_data.head()}")

        # Segmentation of the GSR signal into 10-second windows with 5-second overlap
        logger.debug("Starting segmentation of the GSR signal.")
        segmented = segment_gsr_data(clean_data, fs=256, window_sec=10.0, overlap_sec=5.0)
        
        # Log the segmented data
        logger.debug(f"Segmented data: {segmented.head()}")

        # Feature extraction
        logger.debug("Starting feature extraction.")
        features = extract_features_matrix_optimized(segmented, fs=256)
        
        # Log the extracted features before passing to the model
        logger.debug(f"Extracted features: {features.head()}")

        # Prepare the input tensor (excluding 'Stress' column)
        X = features.drop(columns=['Stress']).values.astype(np.float32)

        # Log the shape of the input features
        logger.debug(f"Input features shape: {X.shape}")

        # Run ONNX prediction
        input_name = session.get_inputs()[0].name
        logger.debug(f"Running ONNX prediction with input: {input_name}")
        outputs = session.run(None, {input_name: X})

        # Get classification result (0 or 1)
        classification = 1 if outputs[0][0][1] > 0.5 else 0

        logger.debug(f"Classification result: {classification}")

        return jsonify({'classification': classification})

    except Exception as e:
        logger.debug(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

# For Render
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
