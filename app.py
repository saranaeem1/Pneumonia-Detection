from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

model = load_model('model/pneumonia_model.h5')
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part")
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    
    if file.filename == '':
        print("No selected file")
        return jsonify({'message': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file.save(filename)
            print(f"File saved to: {filename}")
            
            # Process the image and make prediction
            img = Image.open(filename).convert("RGB")  
            img = img.resize((150, 150))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            print(f"Image shape: {img_array.shape}")

            # Predict using the pre-trained model
            prediction = model.predict(img_array)
            print(f"Prediction: {prediction}")

            result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
            return jsonify({'result': result, 'confidence': float(prediction[0][0])}), 200

        except Exception as e:
            print(f"Error processing the image: {e}")
            return jsonify({'message': 'Error processing the image.'}), 500
    else:
        print("Invalid file format")
        return jsonify({'message': 'Invalid file format. Only JPG, PNG, JPEG allowed'}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)
