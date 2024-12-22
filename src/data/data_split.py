import yaml
import click
import pandas as pd
import os
from sklearn.model_selection import train_test_split

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def split_data(config_path):
    """
    Split les données en ensembles d'entraînement et de test.
    """
    # Charger le fichier de configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Charger les données source
    data_url = config["split_data"]["data_url"]
    target_column = config["split_data"]["target_column"]
    test_size = config["split_data"]["test_size"]
    random_state = config["split_data"]["random_state"]
    output_dir = config["split_data"]["output_dir"]

    # Vérifier si le répertoire de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Le répertoire {output_dir} a été créé.")

    df = pd.read_csv(data_url)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Sauvegarder les datasets
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    click.echo(f"Les données ont été divisées et sauvegardées dans {output_dir}.")

if __name__ == "__main__":
    split_data()