from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the models
rf_model = joblib.load('best_random_forest_model.pkl')
gb_model = joblib.load('best_gradient_boosting_model.pkl')

# Get feature names expected by models (assuming same for both)
feature_names = list(rf_model.feature_names_in_)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Choose model
        model_type = data.get("model", "random_forest").lower()
        if model_type not in ["random_forest", "gradient_boosting"]:
            return jsonify({"error": "Invalid model type. Use 'random_forest' or 'gradient_boosting'."}), 400

        model = rf_model if model_type == "random_forest" else gb_model

        # Build input DataFrame
        input_data = {key: data.get(key, 0) for key in feature_names}
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "model": model_type,
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
