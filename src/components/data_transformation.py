import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    preprocessor_obj_path= os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_config= DataTransformationConfig()
    
    def get_preprocessor_obj(self):
        try:
            numerical_cols=["writing_score","reading_score"]
            cat_cols=[
                "race_ethnicity",
                "gender",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())   
                ]
            )
            cat_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder())
                ]
            )
            logging.info(f"Categorical columns: {cat_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")
            
            preprocessor= ColumnTransformer(transformers=[
                ('num',num_pipeline,numerical_cols),
                ('cat',cat_pipeline,cat_cols)
            ])
            logging.info("Preprocessing object is created ...")
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiateDataTransformation(self, train_data, test_data):
        try:
            target_cols="math_score"
            preprocessor_obj= self.get_preprocessor_obj()
            train_df= pd.read_csv(train_data)
            test_df=pd.read_csv(test_data)
            input_feature_train= train_df.drop(columns=target_cols,axis=1)
            target_feature_train= train_df[target_cols]
            input_feature_test= test_df.drop(columns=target_cols,axis=1)
            target_feature_test= test_df[target_cols]
            logging.info("Data is loaded without target cols train and test is loaded...")
            train_data_obj= preprocessor_obj.fit_transform(input_feature_train)
            test_data_obj= preprocessor_obj.transform(input_feature_test)
            logging.info("Data is transformed...")
            train_arr = np.c_[train_data_obj, np.array(target_feature_train)]
            test_arr= np.c_[test_data_obj, np.array(target_feature_test)]
            
            
            save_object(
                filepath=self.preprocessor_obj_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )
            logging.info("saved Preprocessing data of object..")
            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_config.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e,sys)
