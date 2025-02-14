from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("carbon_model.pkl")
encoders = joblib.load("encoders.pkl")

@app.route("/predict", methods=["POST"])
def predict():
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

        numeric_cols = ["energy_consumption", "transport_distance", "emission_factor"]
        df[numeric_cols] = df[numeric_cols].astype(float)

        predictions = model.predict(df)

        # Assign proper scope classification
        scope_labels = {0: "Scope 1", 1: "Scope 2", 2: "Scope 3"}
        results = []
        for i, row in df.iterrows():
            results.append({
                "input": data[i],  # Original data
                "predicted_scope": scope_labels.get(int(predictions[i]), "Unknown")
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Carbon Emission Classification API is running."

if __name__ == "__main__":
    app.run(debug=True)
