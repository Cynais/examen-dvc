import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import yaml
import click

def load_config(config_path):
    """Charge le fichier de configuration YAML."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train_model(config_path):

    # Charger la configuration
    config = load_config(config_path)
    
    input_dir = config["training"]["input_dir"]
    output_dir = config["training"]["output_dir"]
    
    # Vérifie que le dossier existe si non le cré
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Le répertoire {output_dir} a été créé.")
        
    # Charger les ensembles normalisés
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv").squeeze()
    
    # Charger les meilleurs paramètres
    best_params = joblib.load(f"{output_dir}/best_params.pkl")
    
    # Entraîner le modèle
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    
    # Sauvegarder le modèle
    joblib.dump(model, f"{output_dir}/gbr_model.pkl")

if __name__ == "__main__":
    train_model()