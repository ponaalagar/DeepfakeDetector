import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import io
from werkzeug.utils import secure_filename
import uuid

# --- Configuration ---
MODEL_PATH = "models/cnn_model_tf.h5"
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# Update the directories
DATA_REAL_FOLDER = 'data/real'
DATA_FAKE_FOLDER = 'data/fake'
STATIC_REAL_FOLDER = 'static/real'
STATIC_FAKE_FOLDER = 'static/fake'

# Ensure all directories exist
os.makedirs(DATA_REAL_FOLDER, exist_ok=True)
os.makedirs(DATA_FAKE_FOLDER, exist_ok=True)
os.makedirs(STATIC_REAL_FOLDER, exist_ok=True)
os.makedirs(STATIC_FAKE_FOLDER, exist_ok=True)

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

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
    return render_template('index.html', prediction_text=None, probability=None, image_path=None, filename=None)

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
        # Generate unique filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        
        # Save to uploads temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(temp_path)

        with open(temp_path, 'rb') as f:
            preprocessed_img = preprocess_image(f)

        if preprocessed_img is None:
            flash('Could not process image file.', 'error')
            os.remove(temp_path)  # Clean up
            return redirect(url_for('index'))

        try:
            prediction_prob = model.predict(preprocessed_img)[0][0]
            prediction_text = "Real" if prediction_prob > 0.5 else "Fake"
            probability = float(prediction_prob)

            # Determine target directories
            data_target_folder = DATA_REAL_FOLDER if prediction_text == "Real" else DATA_FAKE_FOLDER
            static_target_folder = STATIC_REAL_FOLDER if prediction_text == "Real" else STATIC_FAKE_FOLDER

            # Copy to both data and static directories
            import shutil
            data_final_path = os.path.join(data_target_folder, unique_filename)
            static_final_path = os.path.join(static_target_folder, unique_filename)
            
            shutil.copy2(temp_path, data_final_path)  # Copy to data directory
            os.rename(temp_path, static_final_path)    # Move to static directory

            # Create URL for template
            image_url = f"/static/{'real' if prediction_text == 'Real' else 'fake'}/{unique_filename}"
            
            return render_template('index.html',
                               prediction_text=prediction_text,
                               probability=probability,
                               image_path=image_url,
                               filename=unique_filename)
        except Exception as e:
            print(f"Error during prediction: {e}")
            flash("An error occurred during prediction.", "error")
            if os.path.exists(temp_path):
                os.remove(temp_path)  # Clean up
            return redirect(url_for('index'))

    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg', 'error')
        return redirect(url_for('index'))

@app.route('/feedback', methods=['POST'])
def feedback():
    filename = request.form.get('filename')
    correct = request.form.get('correct')

    if not filename or not correct:
        flash('Invalid feedback submission.', 'error')
        return redirect(url_for('index'))

    # Check current location of file
    current_data_folder = DATA_REAL_FOLDER if os.path.exists(os.path.join(DATA_REAL_FOLDER, filename)) else DATA_FAKE_FOLDER
    current_static_folder = STATIC_REAL_FOLDER if os.path.exists(os.path.join(STATIC_REAL_FOLDER, filename)) else STATIC_FAKE_FOLDER

    # Determine target folders
    target_data_folder = DATA_FAKE_FOLDER if current_data_folder == DATA_REAL_FOLDER else DATA_REAL_FOLDER
    target_static_folder = STATIC_FAKE_FOLDER if current_static_folder == STATIC_REAL_FOLDER else STATIC_REAL_FOLDER

    try:
        # Move files in both directories
        os.rename(os.path.join(current_data_folder, filename), 
                 os.path.join(target_data_folder, filename))
        os.rename(os.path.join(current_static_folder, filename), 
                 os.path.join(target_static_folder, filename))
        
        flash('Feedback recorded. Image moved to the correct folder.', 'success')
    except Exception as e:
        print(f"Error handling feedback: {e}")
        flash('An error occurred while processing feedback.', 'error')

    return redirect(url_for('index'))

# --- Production Configuration ---
# Disable debug mode for production
DEBUG_MODE = False

# Configure logging for production
import logging
from logging.handlers import RotatingFileHandler

if not DEBUG_MODE:
    # Set up logging to a file
    log_handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=3)
    log_handler.setLevel(logging.INFO)
    app.logger.addHandler(log_handler)

# --- WSGI Entry Point ---
# This ensures the app can be served by a WSGI server like Gunicorn or uWSGI.
application = app

if __name__ == '__main__':
    # For local development only; use Gunicorn in production
    app.run(debug=DEBUG_MODE, host='0.0.0.0', port=5000)
