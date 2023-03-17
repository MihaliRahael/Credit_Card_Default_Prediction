import sys,os
import pandas as pd
from io import BytesIO
from credit_default.entity.config_entity import PredictionPipelineConfig, ModelEvaluationConfig
from credit_default.exception import CreditDefaultException
from credit_default.cloud_storage.s3_syncer import S3Sync
from credit_default.constant.training_pipeline import *
from credit_default.logger import logging
from credit_default.entity.artifact_entity import *
from credit_default.utils.main_utils import *
from credit_default.ml.model.estimator import TargetValueMapping
from credit_default.ml.model.estimator import Transform_Predict
from credit_default.constant.s3_bucket import PREDICTION_BUCKET_NAME
from credit_default.constant import training_pipeline, prediction_pipeline
from credit_default.constant.training_pipeline import SCHEMA_FILE_PATH

class PredictPipeline:
    def __init__(self,input_data,time, model_dir: str):
        self.prediction_pipeline_config = PredictionPipelineConfig(timestamp=time)
        self.input_data = input_data
        self.model_dir = model_dir
        self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
    def read_data_from_api(self)-> pd.DataFrame:
        try:
            logging.info("Reading data...")
            df= pd.read_csv(BytesIO(self.input_data))
            #print(df.shape)
            _id = str(self.schema_config["drop_columns"])   
            if _id in df.columns:
                df = df.drop(self.schema_config["drop_columns"],axis=1)
            return df
        except  Exception as e:
            raise  CreditDefaultException(e,sys)

    def save_to_csv(self,data:pd.DataFrame, path):
        try:
            logging.info(f"saving data to csv in {path}")
            data.to_csv(path)
            
        except  Exception as e:
            raise  CreditDefaultException(e,sys)
    
    def get_latest_model(self):
        try:
            logging.info(f"Fetching the latest best model...")
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            model_object = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, model_object)
            
            #preprocessor_object = os.listdir(latest_model_dir)[1]
            # latest_preprocessor_path = os.path.join(latest_model_dir, preprocessor_object)
            
            return latest_model_path
        
        except Exception as e:
            raise CreditDefaultException(e, sys) from e
    
    def predict(self,data:pd.DataFrame, latest_model_path, latest_preprocessor_path) -> pd.DataFrame:
        
        preprocessor = load_object(file_path=latest_preprocessor_path)
        best_model= load_object(file_path=latest_model_path)
        
        preprocessed_model = Transform_Predict(preprocessor=preprocessor,model=best_model)
        
        y_pred = preprocessed_model.predict(data)
        pred_df = pd.DataFrame(y_pred,columns=["predicted_column"])
        pred_df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
        #print(pred_df.head())

        return pred_df
    
    '''
    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            y_pred = model.predict(X)
            pred_df = pd.DataFrame(y_pred,columns=["predicted_column"])
            pred_df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
            print(pred_df.head())
            return pred_df
        
        except Exception as e:
            raise CreditDefaultException(e, sys) from e

    '''
    
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{PREDICTION_BUCKET_NAME}/artifact/{self.prediction_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.prediction_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise CreditDefaultException(e,sys)

    
    def run_pipeline(self):
        try:
            logging.info(f"{'**' * 10} Entering prediction pipeline {'**' * 10} ")
            input_df =self.read_data_from_api()
            artifact_dir = self.prediction_pipeline_config.artifact_dir
            os.makedirs(artifact_dir,exist_ok=True)
            self.save_to_csv(input_df,self.prediction_pipeline_config.input_file_path)

            latest_model_path = self.get_latest_model()
            best_model= load_object(file_path=latest_model_path)
            
            logging.info("Prediction starts...")
            y_pred = best_model.predict(input_df)
            pred_df = pd.DataFrame(y_pred,columns=["predicted_column"])
            pred_df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
            #print(pred_df.head())
            logging.info("Credit Crad default prediction is completed... Now lets create the prediction artifact")
            
            self.save_to_csv(pred_df,self.prediction_pipeline_config.pred_file_path)

            pred_artifact = PredictionArtifact(
                input_file_path= self.prediction_pipeline_config.input_file_path,
                prediction_file_path= self.prediction_pipeline_config.pred_file_path

            )
            #logging.info("Saving prediction artifact to database")
            #self.prediction_artifact_data.save_prediction_artifact(prediction_artifact=pred_artifact)
            
            logging.info("Saving prediction artifact to S3 bucket")
            #self.sync_artifact_dir_to_s3()
            logging.info(f"Prediction completed and artifact is: {pred_artifact}")

            return pred_artifact

        except  Exception as e:
            #self.sync_artifact_dir_to_s3()
            raise  CreditDefaultException(e,sys)