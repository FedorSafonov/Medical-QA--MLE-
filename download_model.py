import requests
import zipfile
import os

# URL модели
MODEL_URL = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/all-MiniLM-L6-v2.zip"

# Имя директории модели
MODEL_DIR = "model"

# Загрузка модели
response = requests.get(MODEL_URL)
with open(f"{MODEL_DIR}.zip", "wb") as f:
    f.write(response.content)

# Распаковка модели
with zipfile.ZipFile(f"{MODEL_DIR}.zip", "r") as zip_ref:
    zip_ref.extractall(MODEL_DIR)

# Удаление zip-файла
os.remove(f"{MODEL_DIR}.zip")

print(f"Model downloaded and extracted to {MODEL_DIR}")