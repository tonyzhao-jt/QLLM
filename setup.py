from setuptools import setup, find_packages
from torch.utils import cpp_extension

from pathlib import Path
import os

from typing import List

ROOT_DIR = Path(__file__).parent

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

# requirements
def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

setup(
    name='qllm',
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests', 'bench', 'example']),
    install_requires=get_requirements(),
)