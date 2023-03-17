from credit_default.configuration.mongo_db_connection import MongoDBClient
from credit_default.entity.artifact_entity import PredictionArtifact

class PredictionArtifactData:

    def __init__(self):
        self.client = MongoDBClient()
        self.collection_name = "prediction_artifact"
        self.collection = self.client.database[self.collection_name]

    def save_prediction_artifact(self, prediction_artifact: PredictionArtifact):
        self.collection.insert_one(prediction_artifact.to_dict())

    def get_prediction_artifact(self, query):
        self.collection.find_one(query)

    def update_prediction_artifact(self, query, prediction_artifact: PredictionArtifact):
        self.collection.update_one(query, prediction_artifact.to_dict())

    def remove_prediction_artifact(self, query):
        self.collection.delete_one(query)

    def remove_prediction_artifacts(self, query):
        self.collection.delete_many(query)

    def get_prediction_artifacts(self, query):
        self.collection.find(query)
