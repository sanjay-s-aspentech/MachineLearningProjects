import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging

def save_object(filepath, obj):
    try:
        # dirpath= os.path.dirname(filepath)
        # os.makedirs(dirpath, exist_ok=True)
        
        with open(filepath, "wb") as fp:
            pickle.dump(obj, fp)
        logging.info("Pickle object is created ")
    except Exception as e:
        raise CustomException(e,sys)