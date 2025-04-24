from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import pandas as pd
import numpy as np

# Import your preprocessing pipeline
from utils import (
    preprocess_gsr_dataset,
    segment_gsr_data,
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

        # Read uploaded CSV
        file = request.files['file']
        df = pd.read_csv(file)

        # Use only numeric GSR data from the resistance column
        gsr_col = 'Shimmer_1875_GSR_Skin_Resistance_CAL'
        if gsr_col not in df.columns:
            return jsonify({'error': f'Expected column "{gsr_col}" not found'}), 400

        gsr_raw = df[gsr_col]
        gsr_filtered = gsr_raw[gsr_raw.apply(lambda x: isinstance(x, (int, float)))]
        gsr_series = pd.Series(gsr_filtered.dropna().values)

        if gsr_series.empty:
            return jsonify({'error': 'No valid numeric GSR values found'}), 400

        # Match training format: one row with 'GSR_Data'
        session_df = pd.DataFrame([{
            'GSR_Data': gsr_series,
            'Stress': 'unknown'
        }])

        # Preprocess + feature extraction
        clean = preprocess_gsr_dataset(session_df, fs=256)
        segmented = segment_gsr_data(clean, fs=256, window_sec=10.0, overlap_sec=5.0)
        features = extract_features_matrix_optimized(segmented, fs=256)

        # Run ONNX prediction
        input_name = session.get_inputs()[0].name
        X = features.values.astype(np.float32)
        outputs = session.run(None, {input_name: X})
        probs = [float(p[1]) for p in outputs[0]]  # Class 1 = stress

        return jsonify({
            'mean_stress_probability': sum(probs) / len(probs),
            'all_segment_probabilities': probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Render
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
