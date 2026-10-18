from pathlib import Path

import pandas as pd


dataset_path = Path(__file__).resolve().parent / "house_price_data.csv"
data = pd.read_csv(dataset_path)

print("=== Dimensions du dataset ===")
print(f"{data.shape[0]} lignes et {data.shape[1]} colonnes")

print("\n=== Aperçu des 5 premières lignes ===")
print(data.head().to_string(index=False))

print("\n=== Types des colonnes ===")
print(data.dtypes.to_string())

print("\n=== Valeurs manquantes ===")
missing_values = data.isna().sum()
print(missing_values.to_string())

print("\n=== Statistiques des colonnes numériques ===")
print(data.describe().round(2).to_string())

categorical_columns = data.select_dtypes(exclude="number").columns
if len(categorical_columns) > 0:
    print("\n=== Répartition des colonnes catégorielles ===")
    for column in categorical_columns:
        print(f"\n{column} :")
        print(data[column].value_counts(dropna=False).to_string())
