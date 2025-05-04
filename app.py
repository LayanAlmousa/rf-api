import os  # For environment variables
import json  # For handling JSON data
import tempfile  # For creating temporary files
import logging  # For logging
import joblib  # For loading the .pkl model (used to load the random forest model)
import firebase_admin  # Firebase Admin SDK
from firebase_admin import credentials, firestore  # For Firestore integration
from flask import Flask, request, jsonify  # Flask imports
from flask_cors import CORS  # For Cross-Origin Resource Sharing (CORS)
import pandas as pd  # For handling data (e.g., reading CSV files)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Firebase Admin SDK initialization
firebase_key_json = os.getenv('FIREBASE_KEY_JSON')  # Retrieve the Firebase Admin SDK key from environment variables
if not firebase_key_json:
    raise RuntimeError("❌ FIREBASE_KEY_JSON not set in environment variables")

# Save the JSON string to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(firebase_key_json.encode('utf-8'))  # Write the JSON string as bytes
    temp_file_path = temp_file.name  # Get the file path of the temp file

    logger.info(f"Firebase credentials written to temporary file: {temp_file_path}")

# Initialize Firebase Admin SDK using the temp file
cred = credentials.Certificate(temp_file_path)
firebase_admin.initialize_app(cred)

db = firestore.client()  # Firestore client to interact with the database
app = Flask(__name__)
CORS(app)

# Load the .pkl model (instead of ONNX)
model = joblib.load('random_forest.pkl')  # Replace this with the path to your .pkl file

@app.route('/predict-session', methods=['POST'])
def predict_session():
    try:
        # Ensure file is uploaded
        if 'file' not in request.files:
            logger.debug("No file uploaded.")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file:
            return jsonify({'error': 'Invalid file'}), 400

        # Read the uploaded CSV (tab-separated with a header skip)
        df = pd.read_csv(file, sep="\t", header=2)

        # Clean and rename the columns
        df.columns = df.columns.str.strip()
        df.rename(columns={'uS': 'GSR'}, inplace=True)

        # Process the GSR data
        gsr_signal = df['GSR'].values
        logger.debug(f"Processed GSR signal: {gsr_signal[:5]}")

        # Send data to the model for prediction
        model_response = requests.post(FLASK_MODEL_URL, files={'file': file})
        if model_response.status_code != 200:
            return jsonify({"error": "Failed to get prediction from model"}), 500

        model_data = model_response.json()
        is_anxious = model_data.get("isAnxious", False)
        stress_probability = model_data.get("stress_probability", 0.0)

        # Extract user_id from the request headers (user authentication)
        user_id = request.headers.get("user_id")

        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400

        # Save the prediction results to Firestore under the user's GSR_Sessions subcollection
        user_ref = db.collection("UserInfo").document(user_id)
        gsr_session_ref = user_ref.collection("GSR_Sessions").add({
            "date_uploaded": firestore.SERVER_TIMESTAMP,  # Automatically generate timestamp
            "file_name": file.filename,
            "isAnxious": is_anxious,
            "stress_probability": stress_probability
        })

        logger.info(f"Stored GSR session for user {user_id} with file {file.filename}")
        
        # Return the prediction result
        return jsonify({
            "message": "GSR session processed and results saved",
            "isAnxious": is_anxious,
            "stress_probability": stress_probability
        })

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500


# For Render
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
