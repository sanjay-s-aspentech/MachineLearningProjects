from setuptools import find_packages, setup

## Method -1
def get_requirements(file:str):
    with open(file, "r") as f:
        lines=[]
        while(True):
            line = f.readline()
            if line == "-e .":
                break
            else:
                lines.append(line.strip('\n'))
        return lines
    
## Method -2 
# from typing import List

# def get_requirements(file_path:str) -> List[str]:
#     requirements=[]
#     with open(file_path, "r") as f:
#         requirements= f.readlines()
#         requirements = [req.replace("\n","") for req in requirements]
#         if "-e ." in requirements:
#             requirements.remove("-e .")
#         return requirements
        
setup(
    name='mlproject',
    version='1.0.0',
    author="Sanjay.S",
    author_email="sanjaysundarmurthy@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements("./requirements.txt")  
)