import os
from credit_default.constant.s3_bucket import TRAINING_BUCKET_NAME

SAVED_MODEL_DIR =os.path.join("saved_models")
# defining common constant variable for training pipeline
TARGET_COLUMN = "Default_Pay"
PIPELINE_NAME: str = "credit_default"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "credit_default.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
PREPROCESSED_SCHEMA_FILE_PATH = os.path.join("config", "schema_ohe.yaml")
SCHEMA_DROP_COLS = "drop_columns"   # columns which are gonna drop, this is decided by EDA

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "creditdefault"  # from which DB collection i will import the data
DATA_INGESTION_DIR_NAME: str = "data_ingestion" # To which directory will i  export my data
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"  # i will export the data into feature_store inside DB
DATA_INGESTION_INGESTED_DIR: str = "ingested"  # To keep the training and testing file
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_PREPROCESSOR_OBJECT_DIR: str = "preprocessor_object"

"""
Model Trainer related constant start with MODEL_TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
BASE_ACCURACY: float = 0.5
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.4


"""
Model Evaluation related constant start with MODEL_EVALUATION VAR NAME
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME= "report.yaml"
BEST_MODEL_KEY = "best_model"
MODEL_PATH_KEY = "model_path"
HISTORY_KEY = "history"

"""
Model Pusher related constant start with MODEL_PUSHER VAR NAME
"""
MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR