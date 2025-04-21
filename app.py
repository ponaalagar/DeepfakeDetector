import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import io # To handle image stream

# --- Configuration ---
MODEL_PATH = "models\cnn_model_tf.h5" # Adjust if using .keras
UPLOAD_FOLDER = 'uploads' # Optional: If you need to save uploads temporarily
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # Needed for flashing messages

# Ensure upload folder exists (optional)
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Model ---
# Load the model once when the application starts
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=True) # compile=True might be needed depending on saving method
    model.summary() # Print model summary to console on startup
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_stream):
    """Loads, resizes, and preprocesses the image for the model."""
    try:
        img = Image.open(image_stream).convert('RGB') # Ensure RGB
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.array(img)

        # Normalize to [-1, 1] (match training preprocessing)
        img_array = (img_array / 255.0 - 0.5) * 2.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, preprocessing, prediction, and renders the result."""
    if model is None:
        flash("Model not loaded. Please check server logs.", "error")
        return render_template('index.html', error="Model is unavailable.")

    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url) # Redirect back to upload

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index')) # Redirect to index route

    if file and allowed_file(file.filename):
        # Preprocess the image directly from the stream
        preprocessed_img = preprocess_image(file.stream)

        if preprocessed_img is None:
             flash('Could not process image file.', 'error')
             return redirect(url_for('index'))

        # Make prediction
        try:
            prediction_prob = model.predict(preprocessed_img)[0][0] # Get the single probability output
            prediction_text = "Real" if prediction_prob > 0.5 else "Fake"
            probability = float(prediction_prob) # Pass probability to template

            print(f"Prediction Probability (Real): {probability:.4f}, Class: {prediction_text}") # Log prediction

            return render_template('index.html',
                                   prediction_text=prediction_text,
                                   probability=probability)
        except Exception as e:
             print(f"Error during prediction: {e}")
             flash("An error occurred during prediction.", "error")
             return redirect(url_for('index'))

    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg', 'error')
        return redirect(url_for('index'))

# --- Run the App ---
if __name__ == '__main__':
    # Use debug=True only for development, False for production
    app.run(debug=True, host='0.0.0.0', port=5000) # Host 0.0.0.0 makes it accessible on your network