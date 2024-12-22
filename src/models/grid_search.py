import click
import yaml
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def grid_search(config_path):
    """
    Effectue une recherche de grille pour trouver les meilleurs hyperparamètres.
    """
    # Charger le fichier de configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    input_dir = config["grid_search"]["input_dir"]
    output_dir = config["grid_search"]["output_dir"]
    param_grid = config["grid_search"]["param_grid"]

    # Charger les données
    X_train = pd.read_csv(f"{input_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{input_dir}/y_train.csv").squeeze()

    # Initialiser le modèle
    model = RandomForestRegressor(random_state=42)

    # GridSearch
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2")
    grid_search.fit(X_train, y_train)

    # Sauvegarder les meilleurs paramètres
    best_params = grid_search.best_params_
    joblib.dump(best_params, f"{output_dir}/best_params.pkl")

    click.echo(f"Recherche de grille terminée. Meilleurs paramètres sauvegardés dans {output_dir}.")

if __name__ == "__main__":
    grid_search()