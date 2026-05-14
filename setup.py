from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements from a given file.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newline characters and pip flags
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("-")]
            
    return requirements

setup(
    name='docquery',
    version='0.1.0',
    description='RAG based document query system',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)