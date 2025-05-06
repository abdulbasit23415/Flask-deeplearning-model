import keras
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from pathlib import Path

app = Flask(__name__)

# Load the saved model
model_path = Path(r"C:\Users\Aims Tech\Downloads\Sem 6\ML\Flask Assignment 3\src\titanic_mlp_model.h5")
print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)
# Initialize a scaler (ensure it matches the one used during training)
scaler = StandardScaler()



@app.route('/test')
def test():
    return {"message": "Welcome to the Titanic Survival Prediction API"}
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import numpy as np
    print("Predicting...")
    scaler.fit(np.zeros((1,9)))
    print("Scaler fitted")
    try:
        # Map form inputs to numerical values
        sex = 1.0 if request.form['sex'].lower() == 'male' else 0.0
        print("Sex mapped")
        adult_male = 1.0 if request.form['adult_male'].lower() == 'true' else 0.0
        print("Adult male mapped")
        alone = 1.0 if request.form['alone'].lower() == 'true' else 0.0
        print("Alone mapped")
        # Example for other categorical features (customize based on your model's training):
        embarked_map = {'C': 0.0, 'Q': 1.0, 'S': 2.0}
        print("Embarked mapped")
        embarked = embarked_map.get(request.form['embarked'].upper(), 0.0)
        print("Embarked mapped")

        # Numerical features
        pclass = float(request.form['pclass'])
        print("Pclass mapped")
        age = float(request.form['age'])
        print("Age mapped")
        sibsp = float(request.form['sibsp'])
        print("Sibsp mapped")
        parch = float(request.form['parch'])
        print("Parch mapped")
        fare = float(request.form['fare'])
        print("Fare mapped")
        # Combine all features in the order your model expects
        features = [
            pclass, sex, age, sibsp, parch, fare, embarked, adult_male, alone
        ]
        print("Features mapped")

        # Reshape and scale
        features = np.array(features).reshape(1, -1)
        print("Features reshaped")
        scaled_features = scaler.transform(features)  # Use pre-fitted scaler!
        print("Scaled features")

        # Predict and return
        prediction = model.predict(scaled_features)
        print("Prediction made")
        result = "Survived" if prediction[0][0] > 0.5 else "Did not survive"
        
        print(f"result: {result}")
        return result
    
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
