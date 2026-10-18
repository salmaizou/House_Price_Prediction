from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


project_folder = Path(__file__).resolve().parent
dataset_path = project_folder / "house_price_data.csv"
output_path = project_folder / "analyse_exploratoire.png"
data = pd.read_csv(dataset_path)

required_columns = {
    "Area",
    "Location",
    "Condition",
    "Garage",
    "Price",
}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le dataset : {sorted(missing_columns)}")

numeric_data = data.select_dtypes(include="number").drop(columns=["Id"], errors="ignore")
price_correlations = numeric_data.corr()["Price"].drop("Price").sort_values()

print("=== Corrélation des variables numériques avec Price ===")
print(price_correlations.round(3).to_string())

figure, axes = plt.subplots(2, 3, figsize=(16, 9))
figure.suptitle("Exploration du dataset House Price Prediction", fontsize=16)

axes[0, 0].hist(data["Price"], bins=30, edgecolor="black")
axes[0, 0].set_title("Distribution des prix")
axes[0, 0].set_xlabel("Prix")
axes[0, 0].set_ylabel("Nombre de maisons")

axes[0, 1].scatter(data["Area"], data["Price"], alpha=0.5)
axes[0, 1].set_title("Prix selon la surface")
axes[0, 1].set_xlabel("Surface")
axes[0, 1].set_ylabel("Prix")

price_correlations.plot(kind="bar", ax=axes[0, 2], color="steelblue")
axes[0, 2].set_title("Corrélation avec le prix")
axes[0, 2].set_xlabel("Variable")
axes[0, 2].set_ylabel("Corrélation")
axes[0, 2].tick_params(axis="x", rotation=45)

for axis, column, title in [
    (axes[1, 0], "Location", "Prix moyen selon la localisation"),
    (axes[1, 1], "Condition", "Prix moyen selon l'état"),
    (axes[1, 2], "Garage", "Prix moyen selon le garage"),
]:
    average_prices = data.groupby(column, observed=True)["Price"].mean().sort_values()
    average_prices.plot(kind="bar", ax=axis, color="darkseagreen")
    axis.set_title(title)
    axis.set_xlabel(column)
    axis.set_ylabel("Prix moyen")
    axis.tick_params(axis="x", rotation=30)

figure.tight_layout()
figure.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nGraphiques enregistrés dans : {output_path}")
