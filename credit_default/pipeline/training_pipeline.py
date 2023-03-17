import sys
from credit_default.logger import logging
from credit_default.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from credit_default.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact,DataTransformationArtifact
from credit_default.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact,ModelTrainerArtifact
from credit_default.entity.config_entity import ModelPusherConfig,ModelEvaluationConfig,ModelTrainerConfig
from credit_default.exception import CreditDefaultException
from credit_default.components.data_ingestion import DataIngestion
from credit_default.components.data_validation import DataValidation
from credit_default.components.data_transformation import DataTransformation
from credit_default.components.model_trainer import ModelTrainer
from credit_default.components.model_evaluation import ModelEvaluation
from credit_default.components.model_pusher import ModelPusher
from credit_default.cloud_storage.s3_syncer import S3Sync
from credit_default.constant.s3_bucket import TRAINING_BUCKET_NAME
from credit_default.constant.training_pipeline import SAVED_MODEL_DIR

class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()
        
# While defining method variables, we dont initialise it using self for several variables, cos those variables is first imported here, so we can initialise directly.
# data ingestion pipeline/component with o/p artifact being DataIngestionArtifact
# ie this artifact will return train and test file path 
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            logging.info(f"{'**' * 10} Starting Data ingestion ... {'**' * 10} ")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed succesfully. Artifact created...{data_ingestion_artifact}")
            logging.info('\n')
            return data_ingestion_artifact
        except  Exception as e:
            raise  CreditDefaultException(e,sys)

    def start_data_validaton(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                                data_validation_config = data_validation_config)
            logging.info(f"{'**' * 10} Starting Data validation ... {'**' * 10} ")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed succesfully. Artifact created...{data_validation_artifact}")
            logging.info('\n')
            return data_validation_artifact
        except  Exception as e:
            raise  CreditDefaultException(e,sys)

    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                    data_transformation_config=data_transformation_config)
            logging.info(f"{'**' * 10} Starting Data transformation ... {'**' * 10} ")
            data_transformation_artifact =  data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed succesfully. Artifact created...{data_transformation_artifact}")
            logging.info('\n')
            return data_transformation_artifact
        except  Exception as e:
            raise  CreditDefaultException(e,sys)
    
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            logging.info(f"{'**' * 10} Starting model training ... {'**' * 10} ")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed succesfully. Artifact created...{model_trainer_artifact}")
            logging.info('\n')
            return model_trainer_artifact
        except  Exception as e:
            raise  CreditDefaultException(e,sys)
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               data_validation_artifact:DataValidationArtifact,
                               data_transformation_artifact:DataTransformationArtifact,
                               model_trainer_artifact:ModelTrainerArtifact):
        try:
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config, data_ingestion_artifact, data_validation_artifact, data_transformation_artifact, model_trainer_artifact)
            logging.info(f"{'**' * 10} Starting model evaluation ... {'**' * 10} ")
            model_eval_artifact = model_eval.initiate_model_evaluation()
            logging.info(f"Model evaluation completed succesfully. Artifact created...{model_eval_artifact}")
            logging.info('\n')
            return model_eval_artifact
        except  Exception as e:
            raise  CreditDefaultException(e,sys)
        
    
    def start_model_pusher(self,data_transformation_artifact: DataTransformationArtifact, model_eval_artifact:ModelEvaluationArtifact,):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, data_transformation_artifact, model_eval_artifact)
            logging.info(f"{'**' * 10} Starting model pushing ... {'**' * 10} ")
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info(f"Model pushing completed succesfully. Artifact created...{model_pusher_artifact}")
            logging.info('\n')
            return model_pusher_artifact
        except  Exception as e:
            raise  CreditDefaultException(e,sys)

    def sync_artifact_dir_to_s3(self):
        try:
            aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            print(aws_buket_url)
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir,aws_buket_url=aws_buket_url)
        except Exception as e:
            raise CreditDefaultException(e,sys)
            
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            print(aws_buket_url)
            self.s3_sync.sync_folder_to_s3(folder = SAVED_MODEL_DIR,aws_buket_url=aws_buket_url)
        except Exception as e:
            raise CreditDefaultException(e,sys)

    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running=True
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_ingestion_artifact, data_validation_artifact, data_transformation_artifact, model_trainer_artifact)
            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            model_pusher_artifact = self.start_model_pusher(data_transformation_artifact, model_eval_artifact)

            TrainPipeline.is_pipeline_running=False
            logging.info("Training pipeline creation is succesfull. Starting S3 syncing")
            #self.sync_artifact_dir_to_s3()
            #self.sync_saved_model_dir_to_s3()
            logging.info("S3 syncing done")

        except  Exception as e:
            #self.sync_artifact_dir_to_s3() # even if any pipeline fails, sync until then
            #TrainPipeline.is_pipeline_running=False
            raise  CreditDefaultException(e,sys)