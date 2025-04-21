import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import io
from werkzeug.utils import secure_filename

# --- Configuration ---
MODEL_PATH = "models/cnn_model_tf.h5"
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Model ---
try:
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=True)
    model.summary()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_stream):
    try:
        img = Image.open(image_stream).convert('RGB')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.array(img)
        img_array = (img_array / 255.0 - 0.5) * 2.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash("Model not loaded. Please check server logs.", "error")
        return render_template('index.html', error="Model is unavailable.")

    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        with open(filepath, 'rb') as f:
            preprocessed_img = preprocess_image(f)

        if preprocessed_img is None:
            flash('Could not process image file.', 'error')
            return redirect(url_for('index'))

        try:
            prediction_prob = model.predict(preprocessed_img)[0][0]
            prediction_text = "Real" if prediction_prob > 0.5 else "Fake"
            probability = float(prediction_prob)

            return render_template('index.html',
                                   prediction_text=prediction_text,
                                   probability=probability,
                                   image_path=filepath)
        except Exception as e:
            print(f"Error during prediction: {e}")
            flash("An error occurred during prediction.", "error")
            return redirect(url_for('index'))

    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
