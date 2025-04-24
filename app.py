import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import pandas as pd
import numpy as np
from utils import (
    preprocess_gsr_signal,
    segment_single_gsr_segment,
    extract_features_matrix_optimized
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the ONNX model
session = ort.InferenceSession("random_forest.onnx")

@app.route('/predict-session', methods=['POST'])
def predict_session():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400

        # Read uploaded CSV (tab-separated with a header skip)
        file = request.files['file']
        df = pd.read_csv(file, sep="\t", skiprows=1)
        logger.debug(f"DataFrame loaded: {df.head()}")

        # Use only numeric GSR data from the resistance column
        gsr_col = 'Shimmer_1875_GSR_Skin_Resistance_CAL'
        if gsr_col not in df.columns:
            logger.error(f"Column '{gsr_col}' not found in the data")
            return jsonify({'error': f'Expected column \"{gsr_col}\" not found'}), 400

        # Convert strings to float safely
        gsr_series = pd.to_numeric(df[gsr_col], errors='coerce').dropna()
        logger.debug(f"Cleaned GSR data: {gsr_series.head()}")

        if gsr_series.empty:
            logger.error("No valid numeric GSR values found")
            return jsonify({'error': 'No valid numeric GSR values found'}), 400

        # Match training format: one row with 'GSR_Data'
        session_df = pd.DataFrame([{
            'GSR_Data': gsr_series,
            'Stress': 'unknown'
        }])
        logger.debug(f"Session DataFrame: {session_df.head()}")

        # Preprocess + feature extraction
        clean = preprocess_gsr_signal(session_df, fs=256)
        segmented = segment_single_gsr_segment(clean, fs=256, window_sec=10.0, overlap_sec=5.0)
        features = extract_features_matrix_optimized(segmented, fs=256)
        logger.debug(f"Extracted features: {features.head()}")

        # Prepare [1, 27] input tensor
        X = features.drop(columns=['Stress']).values.astype(np.float32)
        if X.shape[0] > 1:
            X = X.mean(axis=0).reshape(1, -1)

        # Run ONNX prediction
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X})
        logger.debug(f"ONNX model outputs: {outputs}")

        # Get classification result (0 or 1)
        classification = 1 if outputs[0][0][1] > 0.5 else 0
        logger.info(f"Predicted classification: {classification}")

        return jsonify({
            'classification': classification
        })

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

# For Render
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
