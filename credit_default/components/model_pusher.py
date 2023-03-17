
from credit_default.exception import CreditDefaultException
from credit_default.logger import logging
from credit_default.entity.artifact_entity import *
from credit_default.entity.config_entity import ModelEvaluationConfig,ModelPusherConfig
import os,sys
from credit_default.ml.metric.classification_metric import get_classification_score
from credit_default.utils.main_utils import save_object,load_object,write_yaml_file

import shutil

class ModelPusher:

    def __init__(self,
                model_pusher_config:ModelPusherConfig,
                data_transform_artifact: DataTransformationArtifact,
                model_eval_artifact:ModelEvaluationArtifact):
        
        try:
            self.model_pusher_config = model_pusher_config
            self.data_transform_artifact = data_transform_artifact
            self.model_eval_artifact = model_eval_artifact
        except  Exception as e:
            raise CreditDefaultException(e, sys)
    
    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            
            trained_model_path = self.model_eval_artifact.evaluated_model_path
            preprocessor_object= self.data_transform_artifact.preprocessor_object_file_path
            
            #Creating model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            #print(model_file_path)
            
            #saved model dir
            saved_model_path = self.model_pusher_config.saved_model
            #print(saved_model_path)
            
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)
            shutil.copy(src=preprocessor_object, dst=model_file_path)
            
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)
            shutil.copy(src=preprocessor_object, dst=saved_model_path)

            #prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, model_file_path=model_file_path)
            
            return model_pusher_artifact
        
        except  Exception as e:
            raise CreditDefaultException(e, sys)
    
    