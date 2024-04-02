# From package directory and right environment, install package using: "pip install -e ."   
from setuptools import setup
import os

r_script_sourcepath = os.path.join("msfeast", "runStats.R")

# Read the version string from version.py
version = {}
with open(os.path.join('msfeast', 'version.py')) as fp:
  exec(fp.read(), version)

setup(
  name = 'msfeast',
  version = version["__version__"],
  include_package_data = True, 
  packages = ['msfeast'],
  scripts = ['msfeast/runStats.R'],
  python_requires = '>=3.10,<3.11',
  install_requires = [
    'numpy', 
    'jupyter',
    "ipykernel",
    'scikit-learn',
    'scipy',
    'plotly',
    'pandas',
    'matchms==0.24.1',
    "matchmsextras==0.4.0",
    'kmedoids==0.5.0',
    # 'spec2vec==0.8.0',
    # 'ms2deepscore==0.4.0',
  ],  
  extras_require={
    'dev': ['pytest']
  }
)