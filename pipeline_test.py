from credit_default.pipeline import training_pipeline # for pipeline test 
from credit_default.pipeline.training_pipeline import TrainPipeline 
from credit_default.logger import logging

if __name__ == '__main__':
    '''
    # entity test
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config= DataIngestionConfig(training_pipeline_config=TrainingPipelineConfig())
    #print(data_ingestion_config.__dict__)
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()
    logging.info("Data ingestion failed")
    '''
    # pipeline test
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()
