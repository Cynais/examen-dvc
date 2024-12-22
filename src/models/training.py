import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

INPUT_DIR = "data/processed"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_model():
    # Charger les ensembles normalisés
    X_train = pd.read_csv(f"{INPUT_DIR}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{INPUT_DIR}/y_train.csv").squeeze()
    
    # Charger les meilleurs paramètres
    best_params = joblib.load(f"{OUTPUT_DIR}/best_params.pkl")
    
    # Entraîner le modèle
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    
    # Sauvegarder le modèle
    joblib.dump(model, f"{OUTPUT_DIR}/trained_model.pkl")

if __name__ == "__main__":
    train_model()