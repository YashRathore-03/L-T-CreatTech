import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# Define paths for datasets
BASE_DIR = "C:\\Users\\91620\\tendon_profiling_project\\data"
datasets = {
    "Box Beam": pd.read_csv(os.path.join(BASE_DIR, "cleaned_Box_Beam.csv")),
    "Rectangular Beam": pd.read_csv(os.path.join(BASE_DIR, "cleaned_Rectangular_Beam.csv")),
    "FRP Beam": pd.read_csv(os.path.join(BASE_DIR, "cleaned_FRP_Beam.csv")),
    "TBeam": pd.read_csv(os.path.join(BASE_DIR, "cleaned_TBeam.csv")),
}

# Define target variables (ensure correct column names)
target_columns = ['Pcr (Cracking Load)', 'Pu (Ultimate Load)', 'Mu (Ultimate Moment)']

# Directory to save models
MODEL_DIR = "C:\\Users\\91620\\tendon_profiling_project\\models"
os.makedirs(MODEL_DIR, exist_ok=True)

models = {}
scalers = {}

# Train models for each dataset
for beam_type, df in datasets.items():
    print(f"\nProcessing {beam_type} dataset...")

    # Ensure column names are stripped of spaces
    df.columns = df.columns.str.strip()

    # Verify target columns exist
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        print(f"⚠️ Warning: Missing target columns in {beam_type}: {missing_targets}")
        continue  # Skip this dataset

    # Drop categorical and non-numeric columns
    categorical_cols = ['bond condition', 'tension method', 'concrete grade']
    df = df.drop(columns=[col for col in categorical_cols if col in df.columns], errors='ignore')

    # Convert all columns to numeric (handle any non-numeric values)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    # Separate features and target variables
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model Performance for {beam_type}:")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - R² Score: {r2:.2f}\n")

    # Save the model and scaler
    joblib.dump(model, os.path.join(MODEL_DIR, f"model_{beam_type.replace(' ', '_')}.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{beam_type.replace(' ', '_')}.pkl"))

print("🎯 Model training complete!")
