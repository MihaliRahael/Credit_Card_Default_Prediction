import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from credit_default.constant.training_pipeline import SCHEMA_FILE_PATH
from credit_default.utils.main_utils import read_yaml_file,write_yaml_file

from credit_default.constant.training_pipeline import TARGET_COLUMN
from credit_default.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from credit_default.entity.config_entity import DataTransformationConfig
from credit_default.exception import CreditDefaultException
from credit_default.logger import logging
from credit_default.ml.model.estimator import TargetValueMapping
from credit_default.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CreditDefaultException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditDefaultException(e, sys)

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            
            numerical_columns = self._schema_config["numerical_columns"]
            categorical_columns = self._schema_config["categorical_columns"]
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            num_pipeline = Pipeline(steps=[('standard scaler', StandardScaler())])
            cat_pipeline = Pipeline(steps=[('one_hot_encoder', OneHotEncoder(handle_unknown = 'ignore', sparse = False, drop='first'))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_preprocessor", num_pipeline, numerical_columns),
                    ("categorical_preprocessor", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CreditDefaultException(e,sys) from e   

    def smote_nc(self, *params):
        try:
            smote_nc = SMOTENC(categorical_features=params[2], random_state=99, sampling_strategy=.85)
            X_resampled, y_resampled = smote_nc.fit_resample(params[0], params[1])
            
            return X_resampled, y_resampled
        
        except Exception as e:
            raise CreditDefaultException(e, sys) from e
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessor_obj = self.get_data_transformer_object()

            logging.info(f"Obtaining train and test file path.")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            #training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            logging.info(f"X_train shape: [{input_feature_train_df.shape}]")
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info(f"Shape of preprocessed data: [{input_feature_train_arr.shape}]")


            numerical_columns = self._schema_config["numerical_columns"]
            cat_col_index = list(range(len(numerical_columns),78)) # for smote-nc oversampling
            
            # Oversampling the train dataset using SMOTE-NC.
            # Reason: SMOTE-NC is a combination of SMOTE and SMOTE-N (which works when all features are categorical) for data which has both numerical and categorical features.
            # Why because SMOTE itself wouldn't work well for a categorical only feature-set
            transformed_encoded_resampled_train_data, target_feature_train_df_resampled = self.smote_nc(input_feature_train_arr, target_feature_train_df, cat_col_index)
            
            logging.info(f"Oversampling is completed successfully")
            logging.info(f"Shape of oversampled train dataframe is : [{transformed_encoded_resampled_train_data.shape}]")
            
            # concatenate input and target features 
            # ie resuls are feature scaled + oversampled train dataset & feature scaled test dataset
            
            train_arr = np.c_[transformed_encoded_resampled_train_data, np.array(target_feature_train_df_resampled)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            
            logging.info(f"Saving preprocessing object.")
            save_object( self.data_transformation_config.preprocessor_object_file_path,obj=preprocessor_obj)
            
            logging.info(f"Lets move on to create the artifact..")
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_object_file_path,                
            )
            
            return data_transformation_artifact
        except Exception as e:
            raise CreditDefaultException(e, sys) from e
        