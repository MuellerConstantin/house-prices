"""
Setup file for the project source code package.
"""

from setuptools import setup, find_packages

setup(
  name="house-prices",
  version="0.1.0",
  packages=find_packages(include=["house_prices", "house_prices.*"]),
  install_requires=[
    "numpy",
    "pandas",
    "tqdm",
    "scikit-learn",
  ],
  entry_points={
    "console_scripts": [
      "preprocess=house_prices.etl.preprocess:main",
      "split=house_prices.etl.split:main",
      "train_lm=house_prices.modelling.train_lm:main",
      "train_lm_ridge=house_prices.modelling.train_lm_ridge:main",
      "train_random_forest=house_prices.modelling.train_random_forest:main",
    ]
  }
)
