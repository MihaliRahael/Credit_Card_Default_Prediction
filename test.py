import os
import sys

ROOT_DIR = os.getcwd()
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

folder_name = list(map(int, os.listdir(MODEL_DIR)))
latest_model_dir = os.path.join(MODEL_DIR, f"{max(folder_name)}")
file_name_0 = os.listdir(latest_model_dir)[0]
print(file_name_0)
file_name_0 = os.listdir(latest_model_dir)[1]
print(file_name_0)

latest_model_path = os.path.join(latest_model_dir, file_name_0)
print(latest_model_path)

