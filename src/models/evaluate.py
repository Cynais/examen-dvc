import click
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os
import yaml

def load_config(config_path):
    """Charge le fichier de configuration YAML."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def evaluate_model(config_path):
    """
    Évalue le modèle avec les données de test et sauvegarde les métriques.
    Le chemin vers le fichier YAML est passé en paramètre.
    """
    # Charger la configuration
    config = load_config(config_path)
    
    input_dir = config["evaluate_model"]["input_dir"]
    model_dir = config["evaluate_model"]["model_dir"]
    metrics_dir = config["evaluate_model"]["metrics_dir"]
    prediction_dir = config["evaluate_model"]["prediction_dir"]
    
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Vérifier l'existence des fichiers de test
    if not os.path.exists(f"{input_dir}/X_test_scaled.csv") or not os.path.exists(f"{input_dir}/y_test.csv"):
        raise FileNotFoundError(f"Les fichiers de test sont manquants dans {input_dir}.")
    
    # Charger les ensembles normalisés
    X_test = pd.read_csv(f"{input_dir}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{input_dir}/y_test.csv").squeeze()
    
    # Vérifier l'existence du modèle
    if not os.path.exists(f"{model_dir}/gbr_model.pkl"):
        raise FileNotFoundError(f"Le modèle est manquant dans {model_dir}.")
    
    # Charger le modèle entraîné
    model = joblib.load(f"{model_dir}/gbr_model.pkl")
    
    # Faire des prédictions
    predictions = model.predict(X_test)
    
    # Calculer les métriques
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Sauvegarder les métriques dans un fichier JSON
    metrics = {"mse": mse, "r2": r2}
    with open(f"{metrics_dir}/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Vérifie que le dossier existe si non le cré
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
        click.echo(f"Le répertoire {metrics_dir} a été créé.")

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
        click.echo(f"Le répertoire {prediction_dir} a été créé.")
    
    # Sauvegarder les prédictions dans un fichier CSV
    predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    predictions_df.to_csv(f"{prediction_dir}/prediction.csv", index=False)
    
        
    # Vérification de la sauvegarde du fichier prediction.csv
    if not os.path.exists(f"{prediction_dir}/prediction.csv"):
        raise FileNotFoundError(f"Le fichier de prédictions n'a pas été créé dans {prediction_dir}.")
    
    print(f"Évaluation terminée. Les résultats sont sauvegardés dans {metrics_dir} et les prédictions dans {prediction_dir}/prediction.csv.")

if __name__ == "__main__":
    # Lancer la commande avec le chemin vers le fichier YAML
    evaluate_model()