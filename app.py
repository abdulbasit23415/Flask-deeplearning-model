from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

app = Flask(__name__)

# Model path - ensure the model file is in the same directory as app.py
model_path = os.path.join(os.getcwd(), "titanic_mlp_model.h5")

# Load the model
print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form values
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        sex = int(request.form['sex'])  # 0 = female, 1 = male
        pclass = int(request.form['pclass'])

        # Prepare input data for prediction
        input_data = np.array([[pclass, sex, age, fare]])
        prediction = model.predict(input_data)
        survived = int(prediction[0][0] > 0.5)

        return render_template('result.html', prediction=survived)

if __name__ == '__main__':
    # Use dynamic port for Railway deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
