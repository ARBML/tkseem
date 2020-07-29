import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

def readme():
    with open('README.md', encoding="utf8") as f:
        return f.read()

setup(name='tkseem',
      version='0.0.1',
      description='Arabic Tokenization Library',
      long_description_content_type='text/markdown',
      long_description=readme(),
      url='https://github.com/MagedSaeed/tkseem',
      author='Zaid Alyafeai, Maged Saeed',
      author_email='alyafey22@gmail.com, mageedsaeed1@gmail.com',
      license='MIT',
      packages=['tkseem'],
      install_requires=required,
      python_requires=">=3.6",
      zip_safe=False)
