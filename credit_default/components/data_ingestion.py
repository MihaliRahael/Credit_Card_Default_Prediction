from credit_default.exception import CreditDefaultException
from credit_default.logger import logging
from credit_default.entity.config_entity import DataIngestionConfig
from credit_default.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
import os,sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from credit_default.data_access.credit_default_data import CreditData
from credit_default.utils.main_utils import read_yaml_file
from credit_default.constant.training_pipeline import SCHEMA_FILE_PATH
class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditDefaultException(e,sys)

    def fetch_data_from_DB(self) -> DataFrame:
        try:
            logging.info("Importing data from mongodb...")
            credit_default_data = CreditData()
            dataframe = credit_default_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            return dataframe
        except  Exception as e:
            raise  CreditDefaultException(e,sys)
    
    def initial_preprocess(self, dataframe: DataFrame) -> DataFrame:
        
        try:
            logging.info("Doing initial preprocessing on imported data...")
            dataframe.drop(self._schema_config["drop_columns"],axis=1, inplace = True)
            dataframe['AGE'] = dataframe['AGE'].astype(np.float64)
            
            ## Renaming poorly labelled features
            dataframe.rename(columns={'PAY_0':'PAY_SEPT','PAY_2':'PAY_AUG','PAY_3':'PAY_JUL','PAY_4':'PAY_JUN','PAY_5':'PAY_MAY','PAY_6':'PAY_APR'},inplace=True)
            dataframe.rename(columns={'BILL_AMT1':'BILL_AMT_SEPT','BILL_AMT2':'BILL_AMT_AUG','BILL_AMT3':'BILL_AMT_JUL','BILL_AMT4':'BILL_AMT_JUN','BILL_AMT5':'BILL_AMT_MAY','BILL_AMT6':'BILL_AMT_APR'}, inplace = True)
            dataframe.rename(columns={'PAY_AMT1':'PAY_AMT_SEPT','PAY_AMT2':'PAY_AMT_AUG','PAY_AMT3':'PAY_AMT_JUL','PAY_AMT4':'PAY_AMT_JUN','PAY_AMT5':'PAY_AMT_MAY','PAY_AMT6':'PAY_AMT_APR'},inplace=True)
            dataframe.rename(columns={'default.payment.next.month':'Default_Pay'},inplace=True)
                
            replace = (dataframe.EDUCATION == 5) | (dataframe.EDUCATION == 6) | (dataframe.EDUCATION == 0)
            dataframe.loc[replace, 'EDUCATION'] = 4
            
            dataframe.loc[dataframe.MARRIAGE == 0, 'MARRIAGE'] = 3
            
            dataframe.replace({'SEX': {1 : 'MALE', 2 : 'FEMALE'}, 
                        'EDUCATION' : {1 : 'Graduate_School', 2 : 'University', 3 : 'High_School', 4 : 'Others'}, 
                        'MARRIAGE' : {1 : 'Married', 2 : 'Single', 3 : 'Others'}},
                    inplace=True)
            
            pay_col = ['PAY_SEPT','PAY_AUG','PAY_JUL','PAY_JUN','PAY_MAY','PAY_APR']
            for col in pay_col:
                dataframe.replace({col : {-2 : 'No_Consumption', -1 : 'Paid_In_Full', 0 : 'Revolving_Credit', 1 : '1MDelay', 2 : '2MDelay', 3 : '3MDelay', 4 : '4MDelay', 5 : '5MDelay', 6 : '6MDelay', 7 : '7MDelay', 8 : '8MDelay', 9 : '9MDelay'}},
                        inplace=True)
            return dataframe
        except  Exception as e:
            raise  CreditDefaultException(e,sys)
    
        
    def export_data_into_feature_store(self, dataframe: DataFrame) -> DataFrame:
        """
        Export mongo db collection record as data frame into feature
        """
        try:
            logging.info("Exporting encoded pre-processed data to the feature store...")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path            
            
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except  Exception as e:
            raise  CreditDefaultException(e,sys)
        

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Feature store dataset will be split into train and test file
        """

        try:
            X = dataframe.drop('Default_Pay',axis=1)
            y = dataframe['Default_Pay']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X,y, test_size=self.data_ingestion_config.train_test_split_ratio,
                stratify=y, random_state=123)
            
            train_set = pd.concat([X_train, y_train], axis=1)
            test_set = pd.concat([X_test, y_test], axis=1)

            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test data into the data ingestion folder...")
        except Exception as e:
            raise CreditDefaultException(e,sys)
    

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            raw_dataframe = self.fetch_data_from_DB()
            preprocessed_dataframe = self.initial_preprocess(dataframe=raw_dataframe)
            dataframe = self.export_data_into_feature_store(dataframe = preprocessed_dataframe)
            self.split_data_as_train_test(dataframe=dataframe)
            
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                            test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise CreditDefaultException(e,sys)