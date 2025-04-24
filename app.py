from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import pandas as pd
import numpy as np

# Import your preprocessing pipeline (same as used in main.py)
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
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read and parse the uploaded CSV file
        file = request.files['file']
        df = pd.read_csv(file, sep="\t", skiprows=2)

        # Extract GSR signal (uS column)
        gsr_signal = df['uS'].astype(float).dropna().tolist()

        # Wrap into a structure matching training data format
        session_df = pd.DataFrame([{'Raw': gsr_signal, 'Stress': 'unknown'}])

        # Preprocess and extract features
        clean = preprocess_gsr_dataset(session_df, fs=256)
        segmented = segment_gsr_data(clean, fs=256, window_sec=10.0, overlap_sec=5.0)
        features = extract_features_matrix_optimized(segmented, fs=256)

        # Run ONNX prediction
        input_name = session.get_inputs()[0].name
        X = features.values.astype(np.float32)
        outputs = session.run(None, {input_name: X})
        probs = [float(p[1]) for p in outputs[0]]  # Stress class probability per segment

        # Return average and full list of probabilities
        return jsonify({
            'mean_stress_probability': sum(probs) / len(probs),
            'all_segment_probabilities': probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
