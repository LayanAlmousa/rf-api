from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

# Load the ONNX model once
session = ort.InferenceSession("random_forest.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read raw bytes directly from the uploaded CSV file
        raw_bytes = file.read()

        # Your ONNX model must be built to parse this raw input
        input_name = session.get_inputs()[0].name
        inputs = {input_name: raw_bytes}
        outputs = session.run(None, inputs)

        return jsonify({'prediction': outputs[0].tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
