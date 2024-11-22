import sys
def get_error_message(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename= str(exc_tb.tb_frame.f_code.co_filename)
    line_no=exc_tb.tb_lineno
    error_message= f"Error has occured in the file {filename} at line no {line_no} error_message : {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=get_error_message(error=error_message, error_detail=error_detail)
    
    def __str__(self) -> str:
        return self.error_message