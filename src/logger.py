import logging
import os
from datetime import datetime
import sys

log_file_format= f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

log_path= os.path.join(os.getcwd(),"logs",log_file_format)
os.makedirs(log_path,exist_ok=True)

Log_File_Path=os.path.join(log_path,log_file_format)

logging.basicConfig(
    filename=Log_File_Path,
    format='[%(asctime)s] %(lineno)d - %(name)s- %(levelname)s: %(message)s',
    level=logging.INFO
)