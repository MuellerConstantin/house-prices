from setuptools import setup, find_packages

setup(
    name="house-prices",
    version="0.1.0",
    packages=find_packages(include=["house_prices", "house_prices.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "tqdm"
    ],
    entry_points={
        "console_scripts": []
    }
)
