import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_object(filepath, obj):
    try:
        # dirpath= os.path.dirname(filepath)
        # os.makedirs(dirpath, exist_ok=True)
        
        with open(filepath, "wb") as fp:
            pickle.dump(obj, fp)
        logging.info("Pickle object is created ")
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,X_test,y_train,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            name= list(models.keys())[i]
            model= list(models.values())[i]
            if name in params.keys():
                para=params[name]
                gs = GridSearchCV(model,param_grid=para,cv=3,verbose=2, n_jobs=-1)
                gs.fit(X_train,y_train)
            else:
                print(f"No parameters found for this model{name}")

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            ## predictions
            y_train_pred= model.predict(X_train)
            y_test_pred= model.predict(X_test)
            
            train_model_score= r2_score(y_train,y_train_pred)
            test_model_score= r2_score(y_test,y_test_pred)
            report[name]=test_model_score
            logging.info("Getting the model names with r2 score evaluating....")
        return report
    except Exception as e:
        raise CustomException(e,sys)