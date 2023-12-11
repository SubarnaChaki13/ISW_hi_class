from setuptools import setup
import os

file_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(file_dir)

setup(name="ISW_likelihood_estimation",
      version='1.0',
      description='Example external Cobaya likelihood package',
      zip_safe=True,  # set to false if you want to easily access bundled package data files
      packages=['ISW_likelihood_estimation'],
      package_data={'ISW_likelihood_estimation': ['*.yaml']},
      install_requires=['cobaya (>=2.0.5)'],
      test_suite='ISW_likelihood_estimation.tests',
      )
