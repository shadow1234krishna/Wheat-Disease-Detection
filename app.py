from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/testCDD'
app.config['MODEL_PATH'] = 'wheatDiseaseModel.h5'

# Load the trained model
model_path = app.config['MODEL_PATH']
model = load_model(model_path)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_disease(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file provided")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Make a prediction using the trained model
        prediction = predict_disease(file_path)
        result = f"Disease probability: {prediction[0][0]:.2%}"
        print(result)

        # Display the test image
        img = image.load_img(file_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(result)
        plt.show()

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
