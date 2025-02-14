import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Function to load training data
def load_data():
    data = {
        "year": [2020, 2021, 2022, 2023, 2024],
        "energy_consumption": [500, 700, 1000, 1200, 1500],
        "total_emission": [1200, 1400, 1800, 2100, 2500]  # Historical CO₂ emissions
    }
    df = pd.DataFrame(data)
    return df

# Function to train a predictive model
def train_model():
    df = load_data()

    X = df[["year", "energy_consumption"]]
    y = df["total_emission"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "emission_predictor.pkl")
    print("✅ Model trained and saved successfully!")

# Train the model once
train_model()

@app.route('/predict_future', methods=['POST'])
def predict():
    try:
        model = joblib.load("emission_predictor.pkl")

        data = request.json
        year = data.get("year")
        energy_consumption = data.get("energy_consumption")

        if not year or not energy_consumption:
            return jsonify({"error": "Missing 'year' or 'energy_consumption'"}), 400

        input_df = pd.DataFrame([[year, energy_consumption]], columns=["year", "energy_consumption"])
        predicted_total_emission = model.predict(input_df)[0]

        # Distribute the emission prediction across 12 months using random variation
        np.random.seed(42)  # Ensure reproducibility
        monthly_distribution = np.random.normal(loc=1, scale=0.05, size=12)
        monthly_distribution = monthly_distribution / monthly_distribution.sum()  # Normalize

        monthly_emissions = {month: round(predicted_total_emission * weight, 2)
                             for month, weight in zip(
                                 ["January", "February", "March", "April", "May", "June",
                                  "July", "August", "September", "October", "November", "December"],
                                 monthly_distribution)}

        response = {
            "year": year,
            "predicted_monthly_emission": monthly_emissions,
            "total_predicted_emission": round(predicted_total_emission, 2)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "CO₂ Emission Prediction API is running!"

if __name__ == '__main__':
    app.run(debug=True)
