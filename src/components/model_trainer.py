import os
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_model_config= ModelTrainerConfig()
    
    def initiate_traing(self, train_arr, test_arr,preprocessor_path):
        try:
            logging.info("splitting training and test data ")
            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1],  
                test_arr[:,:-1],
                train_arr[:,-1], 
                test_arr[:,-1]
            )
            models={
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "SVR": SVR(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                # "KNeighborsRegressor": KNeighborsRegressor()
            }
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict = evaluate_models(X_train,X_test,y_train,y_test, models,params)  
            best_model_score= max(sorted(model_report.values()))
            ## to get the best model name
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model= models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("Model score is less than 0.6 , no best model found..", sys)
            logging.info(f"Best model found on both training dattaset and test dataset ")
            
            save_object(
                filepath=self.trainer_model_config.trained_model_file_path,
                obj=best_model 
            ) 
            predicted_op= best_model.predict(X_test)
            r2= r2_score(y_test,predicted_op)
            logging.info(f"R2 score of the best model is {r2}")
            return r2, best_model_name
        except Exception as e:
            raise CustomException(e,sys)