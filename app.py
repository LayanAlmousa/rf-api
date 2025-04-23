from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np

app = Flask(__name__)
CORS(app)

# Load ONNX model
session = ort.InferenceSession("random_forest.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features'], dtype=np.float32).reshape(1, -1)

    inputs = {session.get_inputs()[0].name: features}
    outputs = session.run(None, inputs)

    # Return probability of stress (e.g., outputs[0] = [[0.2, 0.8]])
    return jsonify({'stress_probability': float(outputs[0][0][1])})

if __name__ == '__main__':
    app.run(debug=True)
