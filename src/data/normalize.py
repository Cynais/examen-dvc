import os
import click
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def normalize_data(config_path):
    """
    Normalise les données d'entraînement et de test.
    """
    # Charger le fichier de configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    input_dir = config["normalize_data"]["input_dir"]
    output_dir = config["normalize_data"]["output_dir"]

    # Vérifier si le répertoire d'entrée existe
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Le répertoire d'entrée {input_dir} n'existe pas.")
    
    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Le répertoire de sortie {output_dir} a été créé.")

    # Charger les données
    try:
        X_train = pd.read_csv(f"{input_dir}/X_train.csv")
        X_test = pd.read_csv(f"{input_dir}/X_test.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Erreur lors du chargement des fichiers CSV : {e}")

    # Vérification des types de données : s'assurer que toutes les colonnes sont numériques
    # Convertir les dates en timestamps si nécessaire
    for col in X_train.columns:
        if X_train[col].dtype == 'O':  # Si la colonne est de type objet (chaîne de caractères)
            try:
                # Essayer de convertir la colonne en datetime
                X_train[col] = pd.to_datetime(X_train[col], errors='coerce')
                X_test[col] = pd.to_datetime(X_test[col], errors='coerce')
                # Convertir les dates en timestamps (seconds depuis epoch)
                X_train[col] = X_train[col].astype(int) / 10**9  # Convertir en secondes
                X_test[col] = X_test[col].astype(int) / 10**9  # Convertir en secondes
            except Exception as e:
                click.echo(f"Erreur lors de la conversion de la colonne {col} : {e}")

    # Vérifier à nouveau que toutes les colonnes sont numériques après conversion
    if not all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns):
        raise ValueError("Le fichier X_train.csv contient des colonnes non numériques après conversion.")
    if not all(pd.api.types.is_numeric_dtype(X_test[col]) for col in X_test.columns):
        raise ValueError("Le fichier X_test.csv contient des colonnes non numériques après conversion.")

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarder les données normalisées
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        f"{output_dir}/X_train_scaled.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
        f"{output_dir}/X_test_scaled.csv", index=False
    )

    click.echo(f"Les données ont été normalisées et sauvegardées dans {output_dir}.")

if __name__ == "__main__":
    normalize_data()