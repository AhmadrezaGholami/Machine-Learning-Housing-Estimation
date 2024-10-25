from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from Regressor_Scratch import SGDRegressor_Scratch
from Regressor_Scratch import NormalEquationRegressor_Scratch

app = Flask(__name__)

# Load both models
with open('sgd_model.pkl', 'rb') as f:
    sgd_model = pickle.load(f)

with open('ne_model.pkl', 'rb') as f:
    ne_model = pickle.load(f)

# Load the fitted scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [float(x) for x in request.form.getlist('features[]')]
        model_choice = request.form['model_choice']

        # Select the model and make prediction
        if model_choice == 'sgd_model':
            # Scale the input features using the fitted scaler
            scaled_features = scaler.transform([features])
            prediction = sgd_model.predict(scaled_features)
        elif model_choice == 'ne_model':
            # Use raw features directly for Normal Equation Model
            prediction = ne_model.predict(np.array(features).reshape(1, -1))
        else:
            return jsonify({'error': 'Invalid model choice'})

        # Return the result
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
