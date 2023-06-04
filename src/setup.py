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
      "preprocess=house_prices.preprocess:main",
      "split=house_prices.split:main",
      "linear_regression=house_prices.ml.linear_regression:main",
      "linear_ridge_regression=house_prices.ml.linear_ridge_regression:main",
      "random_forest=house_prices.ml.random_forest:main",
      "gradient_boosting=house_prices.ml.gradient_boosting:main",
    ]
  }
)
