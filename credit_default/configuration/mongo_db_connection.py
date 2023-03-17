import sys
import certifi
import pymongo
from credit_default.constant.database import DATABASE_NAME 
from credit_default.exception import CreditDefaultException
#from credit_default.constant.env_variable import MONGODB_URL_KEY

ca = certifi.where()

class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                #mongo_db_url = os.getenv(MONGODB_URL_KEY)
                #mongo_db_url = "mongodb+srv://Bonny:Mihali2342@cluster0.3uxym7b.mongodb.net/?retryWrites=true&w=majority"
                mongo_db_url = "mongodb://localhost:27017/"
                '''
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                '''
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)

            self.client = MongoDBClient.client

            self.database = self.client[database_name]

            self.database_name = database_name

        except Exception as e:
            raise CreditDefaultException(e,sys)