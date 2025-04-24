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

        # Debugging: log the first few rows of the dataframe
        print("DEBUG: DataFrame loaded:", df.head())

        # Use only numeric GSR data from the resistance column
        gsr_col = 'Shimmer_1875_GSR_Skin_Resistance_CAL'
        if gsr_col not in df.columns:
            return jsonify({'error': f'Expected column \"{gsr_col}\" not found'}), 400

        # Convert strings to float safely
        gsr_series = pd.to_numeric(df[gsr_col], errors='coerce').dropna()

        # Debugging: log the cleaned GSR data
        print("DEBUG: Cleaned GSR data:", gsr_series.head())

        if gsr_series.empty:
            return jsonify({'error': 'No valid numeric GSR values found'}), 400

        # Match training format: one row with 'GSR_Data'
        session_df = pd.DataFrame([{
            'GSR_Data': gsr_series
        }])

        # Debugging: log the session DataFrame (no Stress column)
        print("DEBUG: Session DataFrame (no Stress column):", session_df.head())

        # Preprocess + feature extraction
        clean = preprocess_gsr_signal(session_df, fs=256)
        
        # Debugging: log the preprocessed signal
        print("DEBUG: Preprocessed data:", clean)

        segmented = segment_single_gsr_segment(clean, fs=256, window_sec=10.0, overlap_sec=5.0)
        
        # Debugging: log the segmented data
        print("DEBUG: Segmented data:", segmented.head())

        features = extract_features_matrix_optimized(segmented, fs=256)
        
        # Debugging: log the features before passing to the model
        print("DEBUG: Extracted features:", features.head())

        # Prepare [1, 27] input tensor (excluding 'Stress' column)
        X = features.drop(columns=['Stress'], errors='ignore').values.astype(np.float32)

        # Debugging: log the shape of the input features
        print("DEBUG: Input features shape:", X.shape)

        if X.shape[0] > 1:
            X = X.mean(axis=0).reshape(1, -1)

        # Debugging: log the final input features shape
        print("DEBUG: Final input features shape:", X.shape)

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
