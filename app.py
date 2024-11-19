from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and preprocessing objects
final_model = joblib.load('final.pkl')
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_features.pkl')
selector = joblib.load('feature_selector.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = request.json
        anglez = float(data['anglez'])
        enmo = float(data['enmo'])

        # Feature engineering
        angle_diff = anglez - enmo
        anglez_squared = anglez ** 2
        log_enmo = np.log1p(enmo)

        # Prepare test data
        X_test = np.array([[anglez, enmo, angle_diff, anglez_squared, log_enmo]])
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)
        X_test_poly = poly.transform(X_test_scaled)
        X_test_selected = selector.transform(X_test_poly)

        # Make prediction
        prediction = final_model.predict(X_test_selected)[0]
        probabilities = final_model.predict_proba(X_test_selected)[0].tolist()

        # Respond with the result
        response = {
            'prediction': int(prediction),
            'probabilities': probabilities
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
