import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

INPUT_DIR = "data/processed"
MODEL_DIR = "models"
METRICS_DIR = "metrics"
os.makedirs(METRICS_DIR, exist_ok=True)

def evaluate_model():
    # Charger les ensembles normalisés
    X_test = pd.read_csv(f"{INPUT_DIR}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{INPUT_DIR}/y_test.csv").squeeze()
    
    # Charger le modèle entraîné
    model = joblib.load(f"{MODEL_DIR}/trained_model.pkl")
    
    # Faire des prédictions
    predictions = model.predict(X_test)
    
    # Calculer les métriques
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Sauvegarder les métriques
    metrics = {"mse": mse, "r2": r2}
    with open(f"{METRICS_DIR}/scores.json", "w") as f:
        json.dump(metrics, f)
    
    # Sauvegarder les prédictions
    pd.DataFrame({"Actual": y_test, "Predicted": predictions}).to_csv(f"{INPUT_DIR}/prediction.csv", index=False)

if __name__ == "__main__":
    evaluate_model()