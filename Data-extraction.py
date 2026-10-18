from pathlib import Path
import shutil

import kagglehub


dataset_handle = "zafarali27/house-price-prediction-dataset"
dataset_file = "House Price Prediction Dataset.csv"
project_folder = Path(__file__).resolve().parent
local_dataset = project_folder / "house_price_data.csv"

downloaded_file = Path(
    kagglehub.dataset_download(dataset_handle, path=dataset_file)
)
if downloaded_file.resolve() != local_dataset.resolve():
    shutil.copy2(downloaded_file, local_dataset)

print(f"Dataset saved to: {local_dataset}")
