import os
import pymongo
from credit_default.logger import logging
from credit_default.constant.database import DATABASE_NAME
from credit_default.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from dotenv import load_dotenv
load_dotenv()

import certifi
ca = certifi.where()

import pandas as pd
class MongodbOperation:

    def __init__(self) -> None:

        #self.client = pymongo.MongoClient(os.getenv('MY_MONGO_DB_URL'),tlsCAFile=ca)
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.db_name=DATABASE_NAME

    def insert_many(self,collection_name,records:list):
        self.client[self.db_name][collection_name].insert_many(records)

    def insert(self,collection_name,record):
        self.client[self.db_name][collection_name].insert_one(record)

datapath = open(r"D:\Credit_Card_Default_Prediction\UCI_Credit_Card.csv")

def main():
    try:
         data = pd.read_csv(datapath)
         records = data.to_dict('records')
         mongodb = MongodbOperation()
         logging.info(f"Connected to database successfully. Starting with data exporting...")
         mongodb.insert_many(collection_name=DATA_INGESTION_COLLECTION_NAME, records=records)
         logging.info(f"Data exported to DB successfully")

    except Exception as e:
            raise e

if __name__=="__main__":
    main()

