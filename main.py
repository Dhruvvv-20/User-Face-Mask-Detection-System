from flask import Flask, render_template, request
import os
import cv2  # Assuming you have OpenCV installed
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Replace with your actual model loading logic
model = load_model('trained_model.h5')  # Placeholder for your model

def predict_from_file(file_path):
    try:
        # Read the image
        input_image = cv2.imread(file_path)
        print("Image read successfully:", input_image.shape)

        # Resize and scale the image
        input_image_resized = cv2.resize(input_image, (128, 128))
        print("Image resized to:", input_image_resized.shape)
        input_image_scaled = input_image_resized / 255
        print("Image scaled to range [0, 1]")

        # Reshape for model input
        input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])
        print("Image reshaped for model input:", input_image_reshaped.shape)

        # Make prediction using your model
        input_prediction = model.predict(input_image_reshaped)
        print("Prediction output:", input_prediction)

        input_pred_label = np.argmax(input_prediction)
        print("Predicted label:", input_pred_label)

        # Return prediction label (0: no mask, 1: mask)
        return input_pred_label
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None  # Indicate prediction error

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['a']

        if f:
            file_path = os.path.join('uploads', f.filename)
            f.save(file_path)

            prediction_label = predict_from_file(file_path)

            if prediction_label is not None:
                if prediction_label == 1:
                    prediction_text = 'The person in the image is wearing a mask'
                else:
                    prediction_text = 'The person in the image is not wearing a mask'
                return render_template('after.html', prediction=prediction_text)
            else:
                return "Error occurred during prediction."
        else:
            return "No file uploaded!"

    return render_template('home.html')

if (__name__ == "__main__"):
    app.run(debug=True)