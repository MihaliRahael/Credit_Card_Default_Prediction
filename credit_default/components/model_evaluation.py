import os
import sys
import numpy as np
from credit_default.constant import *
from credit_default.logger import logging
from credit_default.exception import CreditDefaultException
from credit_default.entity.config_entity import ModelEvaluationConfig
from credit_default.entity.artifact_entity import *
from credit_default.constant.training_pipeline import SCHEMA_FILE_PATH, TARGET_COLUMN
from credit_default.constant.training_pipeline import BEST_MODEL_KEY, MODEL_PATH_KEY, HISTORY_KEY
from credit_default.utils.main_utils import *
from sklearn.metrics import f1_score , roc_auc_score
from collections import namedtuple

MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])

class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self._schema_filepath = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditDefaultException(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_dir

            if not os.path.exists(model_evaluation_file_path):
                create_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise CreditDefaultException(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_dir
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval report:{model_eval_content}")
            create_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise CreditDefaultException(e, sys) from e
        
        
    def evaluate_classification_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6) -> MetricInfoArtifact:

        try:
            index_number = 0
            metric_info_artifact = None
            for model in model_list:
                model_name = str(model)  #getting model name based on model object
                logging.info(f"Started evaluating model: [{type(model).__name__}] ")
                
                #Getting prediction for training and testing dataset
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                #Calculating f1 score on training and testing dataset
                train_acc = f1_score (y_train, y_train_pred)
                test_acc = f1_score(y_test, y_test_pred)
                
                #Calculating mean squared error on training and testing dataset
                train_rocauc = np.sqrt(roc_auc_score(y_train, y_train_pred))
                test_rocauc = np.sqrt(roc_auc_score(y_test, y_test_pred))

                # Calculating harmonic mean of train_accuracy and test_accuracy
                model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
                diff_test_train_acc = abs(test_acc - train_acc)
                
                #logging all important metric
                logging.info(f"{'>>'*10} Score {'<<'*10}")
                logging.info(f"Train Score\t\t Test Score\t\t Average Score")
                logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

                logging.info(f"{'>>'*10} Loss {'<<'*10}")
                logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
                logging.info(f"Train root mean squared error: [{train_rocauc}].")
                logging.info(f"Test root mean squared error: [{test_rocauc}].")


                #if model accuracy is greater than base accuracy and train and test score is within certain thershold
                #we will accept that model as accepted model
                if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                    base_accuracy = model_accuracy
                    metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                            model_object=model,
                                                            train_rmse=train_rocauc,
                                                            test_rmse=test_rocauc,
                                                            train_accuracy=train_acc,
                                                            test_accuracy=test_acc,
                                                            model_accuracy=model_accuracy,
                                                            index_number=index_number)

                    logging.info(f"Acceptable model found {metric_info_artifact}. ")
                index_number += 1
            if metric_info_artifact is None:
                logging.info(f"No model found with higher accuracy than base accuracy")
            return metric_info_artifact
        except Exception as e:
            raise CreditDefaultException(e, sys) from e

        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(f"Loading the dataset for evaluation")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # split into input feature and target feature
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)
			
            logging.info("Fetching the best model...")
            model = self.get_best_model()
            
            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]

            metric_info_artifact = self.evaluate_classification_model(model_list=model_list,
                                                               X_train=x_train,
                                                               y_train=y_train,
                                                               X_test=x_test,
                                                               y_test=y_test,
															   base_accuracy = 70
                                                               #base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                               )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
			
        except Exception as e:
            raise CreditDefaultException(e, sys) from e

