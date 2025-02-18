from setuptools import find_packages,setup
from typing import List

HYPHOEN_E_DOT = "-e ."
def get_requirements(path:str)->List[str]:
    requirements = []
    with open(path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n',"") for req in requirements]

        if HYPHOEN_E_DOT in requirements:
            requirements.remove(HYPHOEN_E_DOT)
        
    return requirements

setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Vainavi',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)