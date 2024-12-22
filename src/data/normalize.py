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

    # Charger les données
    X_train = pd.read_csv(f"{input_dir}/X_train.csv")
    X_test = pd.read_csv(f"{input_dir}/X_test.csv")

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