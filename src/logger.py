import logging
import os
from datetime import datetime
from exceptions import CustomException
import sys

log_file_format= f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

log_path= os.path.join(os.getcwd(),"logs",log_file_format)
os.makedirs(log_path,exist_ok=True)

Log_File_Path=os.path.join(log_path,log_file_format)

logging.basicConfig(
    filename=Log_File_Path,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d - %(name)s %(levelname)s: %(message)s"
)

if __name__=="__main__":
    
    try:
        a=1/0
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise CustomException(e,sys)