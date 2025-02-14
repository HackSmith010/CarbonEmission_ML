import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data():
    
    data = {
        "source_type": ["Vehicle", "Electricity", "Supplier", "Vehicle", "Electricity"],
        "ownership": ["Owned", "Purchased", "Third-Party", "Owned", "Purchased"],
        "activity_type": ["Fuel Combustion", "Power Usage", "Purchased Goods", "Fuel Combustion", "Power Usage"],
        "energy_consumption": [500, 2000, 1000, 600, 1800],
        "fuel_type": ["Diesel", "Grid", "N/A", "Petrol", "Grid"],
        "transport_distance": [200, np.nan, 500, 150, np.nan],
        "emission_factor": [2.68, 0.42, 1.89, 3.2, 0.38],
        "emission_scope": ["Scope 1", "Scope 2", "Scope 3", "Scope 1", "Scope 2"]  
    }
    
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    label_encoders = {}

    # Fill missing transport_distance values
    df["transport_distance"].fillna(0, inplace=True)

    for col in ["source_type", "ownership", "activity_type", "fuel_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoder for future use
    
    return df, label_encoders

def train_model():
    df = load_data()
    df, encoders = preprocess_data(df)

    X = df.drop(columns=["emission_scope"])
    y = df["emission_scope"].map({"Scope 1": 0, "Scope 2": 1, "Scope 3": 2})  # Convert labels to numbers

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "carbon_model.pkl")
    joblib.dump(encoders, "encoders.pkl")
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
