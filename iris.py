from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('iris_model.pkl')

# class names
class_map = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Serve HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'templates/index.html')

# Handle prediction
@app.route('/model_prediction', methods=['POST'])
def model_prediction():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    label = class_map[int(prediction)]
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
