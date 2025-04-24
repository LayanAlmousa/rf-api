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

app = Flask(__name__)
CORS(app)

# Load the ONNX model
session = ort.InferenceSession("random_forest.onnx")

@app.route('/predict-session', methods=['POST'])
def predict_session():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read uploaded CSV (tab-separated with a header skip)
        file = request.files['file']
        df = pd.read_csv(file, sep="\t", skiprows=1)

        # Log the DataFrame to see the raw data
        print("DEBUG: DataFrame loaded:")
        print(df.head())

        # Ensure we clean non-numeric columns before processing
        gsr_col = 'Shimmer_1875_GSR_Skin_Resistance_CAL'
        if gsr_col not in df.columns:
            return jsonify({'error': f'Expected column \"{gsr_col}\" not found'}), 400

        # Convert strings to float safely, logging the cleaned GSR data
        df[gsr_col] = pd.to_numeric(df[gsr_col], errors='coerce')
        print(f"DEBUG: Cleaned GSR Data: {df[gsr_col].head()}")

        gsr_series = df[gsr_col].dropna()

        if gsr_series.empty:
            return jsonify({'error': 'No valid numeric GSR values found'}), 400

        # Match training format: one row with 'GSR_Data'
        session_df = pd.DataFrame([{
            'GSR_Data': gsr_series,
            'Stress': 'unknown'
        }])

        print(f"DEBUG: Session DataFrame: {session_df.head()}")

        # Preprocess + feature extraction
        clean = preprocess_gsr_signal(session_df, fs=256)
        segmented = segment_single_gsr_segment(clean, fs=256, window_sec=10.0, overlap_sec=5.0)
        features = extract_features_matrix_optimized(segmented, fs=256)

        print(f"DEBUG: Features extracted: {features.head()}")

        # Prepare [1, 27] input tensor
        X = features.drop(columns=['Stress']).values.astype(np.float32)
        if X.shape[0] > 1:
            X = X.mean(axis=0).reshape(1, -1)

        # Run ONNX prediction
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: X})

        # Get classification result (0 or 1)
        classification = 1 if outputs[0][0][1] > 0.5 else 0

        return jsonify({
            'classification': classification
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Render
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
