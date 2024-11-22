import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """Data ingestion configuration class"""
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
    def initiateIngestionData(self):
        logging.info("Entered the Ingestion Data Phase")
        try:
            # Read the data from the given paths
            df=pd.read_csv("notebook\\stud.csv")
            logging.info("Read the Dataset as Dataframe....")
            # Save the data into the artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Initiated Train test Split on dataframe")
            train_data, test_data= train_test_split(df, test_size=0.30, random_state=7)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error in Ingestion Data Phase: {str(e)}")
            raise CustomException(e, sys)

if __name__=="__main__":
    obj= DataIngestion()
    train_data, test_data=obj.initiateIngestionData()
    
    dataobtained= DataTransformation()
    train_arr, test_arr, preprocess_path=dataobtained.initiateDataTransformation(train_data, test_data)
    
    modeltraining= ModelTrainer()
    score,name= modeltraining.initiate_traing(train_arr, test_arr, preprocess_path)
    print(score, name)