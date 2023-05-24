"""
Module for training a linear regression model.
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from house_prices.modelling.feature_selection import get_ordinal_feature_names, get_binary_feature_names, get_numerical_feature_names, ORDINAL_FEATURE_MAPPINGS

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def build_model(df: pd.DataFrame):
  """
  Builds a linear regression model.
  """

  vprint("Building model ...")

  ordinal_features = get_ordinal_feature_names(df)
  binary_features = get_binary_feature_names(df)
  numerical_features = get_numerical_feature_names(df)

  ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[value for key, value in ORDINAL_FEATURE_MAPPINGS.items()], dtype=int)),
  ])

  binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
  ])

  numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
  ])

  transformer = ColumnTransformer([
    ("ordinal", ordinal_pipeline, ordinal_features),
    ("binary", binary_pipeline, binary_features),
    ("numerical", numerical_pipeline, numerical_features),
  ])

  estimator = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp)

  model = Pipeline([
    ("transformer", transformer),
    ("estimator", estimator),
  ])

  return model

def train_model(x: pd.DataFrame, y: pd.Series):
  """
  Trains a linear regression model.
  """

  model = build_model(x)

  vprint("Training model ...")

  model.fit(x, y)

  return model

def main():
  parser = argparse.ArgumentParser(prog="train_lm.py",
                                   formatter_class=argparse.RawTextHelpFormatter,
                                   description="Trains a linear regression model.")

  parser.add_argument("-v", "--verbose", action="store_true",
                      help="Print out verbose messages.")
  parser.add_argument("-i", "--input", type=str, required=True,
                      help="Input file path of train set. Must be a CSV file.")
  parser.add_argument("-o", "--output", type=str, required=True,
                      help="Output file path. Must be a JOBLIB file.")

  args = parser.parse_args()

  if args.verbose:
    global vprint
    vprint = print

  vprint(f"Loading data from '{args.input}' ...")

  df = pd.read_csv(args.input)
  model = train_model(df.drop("SalePrice", axis=1), df["SalePrice"])

  vprint(f"Saving model to '{args.output}' ...")

  joblib.dump(model, args.output)

if __name__ == "__main__":
  main()
