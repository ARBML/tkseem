import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='tkseem',
      version='0.1',
      description='Arabic Tokenization Library',
      url='https://github.com/MagedSaeed/tkseem',
      author='Zaid Alyafeai',
      author_email='alyafey22@gmail.com',
      license='MIT',
      packages=['tkseem'],
      install_requires=required,
      zip_safe=False)