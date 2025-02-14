from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from collections import OrderedDict

app = Flask(__name__)
CORS(app)

classification_model = joblib.load("carbon_model.pkl")
encoders = joblib.load("encoders.pkl")

def load_data():
    data = {
        "year": [2020, 2021, 2022, 2023, 2024],
        "energy_consumption": [500, 700, 1000, 1200, 1500],
        "total_emission": [1200, 1400, 1800, 2100, 2500]  
    }
    return pd.DataFrame(data)

def train_prediction_model():
    df = load_data()
    X = df[["year", "energy_consumption"]]
    y = df["total_emission"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "emission_predictor.pkl")
    print("✅ Prediction model trained and saved!")

train_prediction_model()

@app.route("/predict", methods=["POST"])
def classify_scope():
    try:
        data = request.get_json()
        if isinstance(data, dict):  
            data = [data]  
        
        df = pd.DataFrame(data)

        required_columns = ["source_type", "ownership", "activity_type", "energy_consumption", "fuel_type", "transport_distance", "emission_factor"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {', '.join(missing_cols)}"}), 400

        for col in encoders:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if x in encoders[col].classes_ else "Unknown")
                df[col] = encoders[col].transform(df[col])

        df.fillna(0, inplace=True)
        df[["energy_consumption", "transport_distance", "emission_factor"]] = df[["energy_consumption", "transport_distance", "emission_factor"]].astype(float)

        predictions = classification_model.predict(df)

        scope_labels = {0: "Scope 1", 1: "Scope 2", 2: "Scope 3"}
        results = []
        for i, row in df.iterrows():
            results.append({
                "input": data[i],
                "predicted_scope": scope_labels.get(int(predictions[i]), "Unknown")
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🔹 Future Emission Prediction Route
@app.route('/predict_future', methods=['POST'])
def predict_future():
    try:
        model = joblib.load("emission_predictor.pkl")

        data = request.json
        year = data.get("year")
        energy_consumption = data.get("energy_consumption")

        if year is None or energy_consumption is None:
            return jsonify({"error": "Missing 'year' or 'energy_consumption'"}), 400

        input_df = pd.DataFrame([[year, energy_consumption]], columns=["year", "energy_consumption"])
        predicted_total_emission = model.predict(input_df)[0]

        # 🔹 Distribute the emission prediction across 12 months with variation
        np.random.seed(42)
        monthly_distribution = np.random.normal(loc=1, scale=0.05, size=12)
        monthly_distribution = monthly_distribution / monthly_distribution.sum()

        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        
        # 🔹 Use OrderedDict to preserve month order
        monthly_emissions = OrderedDict(
            (month, round(predicted_total_emission * weight, 2))
            for month, weight in zip(months, monthly_distribution)
        )

        response = {
            "predicted_monthly_emission": monthly_emissions,
            "total_predicted_emission": round(predicted_total_emission, 2),
            "year": year
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Carbon Emission Classification & Prediction API is running!"

if __name__ == '__main__':
    app.run(debug=True)
