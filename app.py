import keras
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

# Load the saved model
model = keras.models.load_model('titanic_mlp_model.h5')
# Initialize a scaler (ensure it matches the one used during training)
scaler = StandardScaler()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import numpy as np
    scaler.fit(np.zeros((1,9)))
    try:
        # Map form inputs to numerical values
        sex = 1.0 if request.form['sex'].lower() == 'male' else 0.0
        adult_male = 1.0 if request.form['adult_male'].lower() == 'true' else 0.0
        alone = 1.0 if request.form['alone'].lower() == 'true' else 0.0

        # Example for other categorical features (customize based on your model's training):
        embarked_map = {'C': 0.0, 'Q': 1.0, 'S': 2.0}
        embarked = embarked_map.get(request.form['embarked'].upper(), 0.0)

        # Numerical features
        pclass = float(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = float(request.form['sibsp'])
        parch = float(request.form['parch'])
        fare = float(request.form['fare'])

        # Combine all features in the order your model expects
        features = [
            pclass, sex, age, sibsp, parch, fare, embarked, adult_male, alone
        ]

        # Reshape and scale
        features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features)  # Use pre-fitted scaler!

        # Predict and return
        prediction = model.predict(scaled_features)
        result = "Survived" if prediction[0][0] > 0.5 else "Did not survive"
        print(f"result: {result}")
        return result
    
    except Exception as e:
        return f"error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)