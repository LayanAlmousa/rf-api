from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
import joblib  # Used to load the .pkl model

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

# Load the .pkl model (instead of ONNX)
model = joblib.load('random_forest.pkl')  # Replace this with the path to your .pkl file


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

        gsr_signal = df['GSR'].values
        
        # Create a dummy DataFrame for input
        raw_df = pd.DataFrame({
            'Subject': ['Sample'],
            'GSR_Data': [pd.Series(gsr_signal)],
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
        X = features.drop(columns=['Stress'])
        
        # Log the shape of the input features
        logger.debug(f"Input features shape: {X.shape}")

        # Run prediction using the .pkl model
        preds = model.predict(X)  # Use the model's `predict` method
        probs = model.predict_proba(X)[:, 1]  # Probability of stress

        # Get classification result (0 or 1)
        classification = 1 if probs[0] > 0.5 else 0  # Use the threshold for stress classification

        logger.debug(f"Classification result: {classification}")

        return jsonify({'classification': classification, 'probability': probs[0]})

    except Exception as e:
        logger.debug(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

# For Render
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
