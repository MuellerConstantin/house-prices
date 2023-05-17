"""
Module for training a linear regression model.
"""

import argparse
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def build_model():
  """
  Builds a linear regression model.
  """

  vprint("Building model ...")

  categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
  ])

  numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
  ])

  transformer = make_column_transformer(
    (categorical_pipeline, make_column_selector(dtype_include="object")),
    (numerical_pipeline, make_column_selector(dtype_include="number")),
  )

  model = Pipeline([
    ("transformer", transformer),
    ("regressor", LinearRegression()),
  ])

  return model

def train_model(x: pd.DataFrame, y: pd.Series):
  """
  Trains a linear regression model.
  """

  vprint("Training model ...")

  model = build_model()
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
