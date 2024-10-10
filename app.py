import os
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model using Pickle
try:
    with open('digit_recognition_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    raise RuntimeError("Failed to load model: " + str(e))

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict the digit from an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    
    # Validate the file type
    if not file or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image.'}), 400

    try:
        # Process the image
        image = Image.open(file.stream).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to model's expected input
        image = np.array(image) / 255.0  # Normalize pixel values
        image = image.reshape(1, 28, 28, 1)  # Reshape for prediction

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)

        return jsonify({'digit': int(predicted_class[0])})

    except Exception as e:
        return jsonify({'error': 'Error processing image: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
