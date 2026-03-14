from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements from a given file.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newline characters
        requirements = [req.replace("\n", "") for req in requirements]

        # Ignore '-e .' if it is present
        if "-e ." in requirements:
            requirements.remove("-e .")
            
    return requirements

setup(
    name='docquery',
    version='0.1.0',
    description='RAG based document query system',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)