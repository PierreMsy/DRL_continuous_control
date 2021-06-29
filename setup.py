import os
from setuptools import find_packages, setup

os.chdir(os.path.dirname(os.path.abspath(__file__)))

setup(
    name='drl_cont_control',
    packages=find_packages(),
    version='0.0.1',
    description='Attempts at solving the Unity ML-Agents Reacher Environment for single or multiples agents.',
    author='Pierre Massey',
    license='MIT',
    url="https://github.com/PierreMsy/DRL_continuous_control.git",
    include_package_data=True
)