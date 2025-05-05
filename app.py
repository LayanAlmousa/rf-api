from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
import joblib
import os
import tempfile

import firebase_admin
from firebase_admin import credentials, firestore

from utils import (
    preprocess_gsr_dataset,
    segment_gsr_data,
    extract_features_matrix_optimized
)

# === Logging Setup ===
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# === Flask Setup ===
app = Flask(__name__)
CORS(app)

# === Firebase Admin Init ===
firebase_key_json = os.getenv('FIREBASE_KEY_JSON')
if not firebase_key_json:
    raise RuntimeError("❌ FIREBASE_KEY_JSON not set in environment variables")

with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(firebase_key_json.encode('utf-8'))
    temp_file_path = temp_file.name
    logger.info(f"Firebase credentials written to temporary file: {temp_file_path}")

cred = credentials.Certificate(temp_file_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# === Load ML Model ===
model = joblib.load('random_forest.pkl')


@app.route('/predict-session', methods=['POST'])
def predict_session():
    try:
        # Step 1: File check
        if 'file' not in request.files:
            logger.debug("No file uploaded.")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        df = pd.read_csv(file, sep="\t", header=2)

        # Step 2: Clean and parse GSR column
        df.columns = df.columns.str.strip()
        df.rename(columns={'uS': 'GSR'}, inplace=True)

        if 'GSR' not in df.columns:
            logger.debug("GSR column missing.")
            return jsonify({'error': 'Expected column "GSR" not found'}), 400

        gsr_signal = df['GSR'].values

        raw_df = pd.DataFrame({
            'Subject': ['Sample'],
            'GSR_Data': [pd.Series(gsr_signal)],
            'Label': ['unknown']
        })

        # Step 3: Preprocessing
        logger.debug("Starting preprocessing.")
        clean_data = preprocess_gsr_dataset(raw_df, fs=256)

        # Step 4: Segmentation
        logger.debug("Starting segmentation.")
        segmented = segment_gsr_data(clean_data, fs=256, window_sec=10.0, overlap_sec=5.0)

        # Step 5: Feature extraction
        logger.debug("Extracting features.")
        features = extract_features_matrix_optimized(segmented, fs=256)
        X = features.drop(columns=['Stress'])

        logger.debug(f"Feature shape: {X.shape}")

        # Step 6: Prediction
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        classification = 1 if probs[0] > 0.5 else 0
        probability = float(probs[0])

        logger.debug(f"Prediction: {classification}, Probability: {probability}")

        # Step 7: Store in Firebase
        user_id = request.headers.get("uid")
        if not user_id:
            logger.error("User ID missing.")
            return jsonify({'error': 'Missing user ID in headers'}), 400

        user_ref = db.collection("UserInfo").document(user_id)
        sessions_ref = user_ref.collection("GSR_Sessions")

        # Check if file with same name already exists
        existing = sessions_ref.where("file_name", "==", file.filename).limit(1).get()
        if existing:
            logger.warning(f"Duplicate file detected: {file.filename}")
            return jsonify({
                'error': f'File "{file.filename}" has already been uploaded.',
                'code': 'DUPLICATE_FILE'
            }), 409

        # Save new session
        sessions_ref.add({
            "date_uploaded": firestore.SERVER_TIMESTAMP,
            "file_name": file.filename,
            "isAnxious": bool(classification),
            "stress_probability": probability
        })

        logger.info(f"Session stored for user: {user_id}, file: {file.filename}")

        return jsonify({
            "message": "Prediction and upload successful.",
            "classification": classification,
            "probability": probability,
            "isAnxious": bool(classification)
        })

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
