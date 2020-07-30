import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name='tkseem',
      version='0.0.1',
      discription="Arabic Tokenization Library",
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/ARBML/tkseem',
      author='Zaid Alyafeai, Maged Saeed',
      author_email='arabicmachinelearning@gmail.com',
      license='MIT',
      packages=['tkseem'],
      install_requires=required,
      include_package_data=True,
      zip_safe=False,
      )