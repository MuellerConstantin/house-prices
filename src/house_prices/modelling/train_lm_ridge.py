"""
Module for training a linear regression model with ridge regularization.
"""

import argparse
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def build_model():
  """
  Builds a linear regression model with ridge regularization.
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
    ("regressor", Ridge()),
  ])

  return model

def train_model(x: pd.DataFrame,
                y: pd.Series,
                param_distributions,
                n_iter=10,
                n_folds=10,
                n_jobs=None,
                verbose=0,
                random_state=42):
  """
  Trains a linear regression model with ridge regularization.
  """

  model = build_model()
  cv = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=n_folds,
                          n_jobs=n_jobs, verbose=verbose, random_state=random_state)

  vprint("Training model ...")

  cv.fit(x, y)

  return cv

def main():
  parser = argparse.ArgumentParser(prog="train_lm_ridge.py",
                                   formatter_class=argparse.RawTextHelpFormatter,
                                   description="Trains a linear ridge regression model.")

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

  hyperparameters = {
    "regressor__alpha": [0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
  }

  model = train_model(df.drop("SalePrice", axis=1), df["SalePrice"], hyperparameters,
                      verbose=5 if args.verbose else 0)

  vprint(f"Saving model to '{args.output}' ...")

  joblib.dump(model, args.output)

if __name__ == "__main__":
  main()
