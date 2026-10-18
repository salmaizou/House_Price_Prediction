from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


dataset_path = Path(__file__).resolve().parent / "house_price_data.csv"
data = pd.read_csv(dataset_path)

target_column = "Price"
if target_column not in data.columns:
    raise ValueError(f"Le dataset doit contenir une colonne {target_column!r}.")

features = data.drop(columns=[target_column, "Id"], errors="ignore")
target = data[target_column]

numeric_columns = features.select_dtypes(include="number").columns
categorical_columns = features.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numbers", "passthrough", numeric_columns),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regression", RandomForestRegressor(n_estimators=200, random_state=42)),
    ]
)

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
model.fit(features_train, target_train)
predictions = model.predict(features_test)

baseline_predictions = [target_train.mean()] * len(target_test)
mae = mean_absolute_error(target_test, predictions)
rmse = mean_squared_error(target_test, predictions) ** 0.5
r2 = r2_score(target_test, predictions)
baseline_mae = mean_absolute_error(target_test, baseline_predictions)

print(f"Maisons utilisées pour l'entraînement : {len(features_train)}")
print(f"Maisons utilisées pour le test : {len(features_test)}")
print(f"Erreur absolue moyenne du modèle (MAE) : {mae:,.0f}")
print(f"Racine de l'erreur quadratique moyenne (RMSE) : {rmse:,.0f}")
print(f"Score R² : {r2:.3f}")
print(f"MAE d'une prédiction toujours égale au prix moyen : {baseline_mae:,.0f}")

comparison = pd.DataFrame(
    {
        "Prix réel": target_test.head(10).to_numpy(),
        "Prix prédit": predictions[:10].round(0).astype(int),
    }
)
print("\nExemples de prédictions (10 maisons du jeu de test) :")
print(comparison.to_string(index=False))
