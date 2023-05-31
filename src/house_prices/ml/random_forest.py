"""
Module for training a random forest model.
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from house_prices.modelling import build_transformer, get_ordinal_feature_mappings

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def train_model(x: pd.DataFrame,
                y: pd.Series,
                param_distributions,
                n_iter=10,
                n_folds=10,
                n_jobs=None,
                verbose=0,
                random_state=42):
  """
  Trains a random forest model.
  """

  vprint("Building model ...")

  ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=get_ordinal_feature_mappings(x), dtype=int)),
  ])

  binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", categories="auto")),
  ])

  numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
  ])

  estimator = RandomForestRegressor()

  transformer = build_transformer(x, ordinal_pipeline, binary_pipeline, numerical_pipeline)
  model = Pipeline([
    ("transformer", transformer),
    ("estimator", estimator),
  ])

  cv = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=n_folds,
                          n_jobs=n_jobs, verbose=verbose, random_state=random_state, return_train_score=True)

  vprint("Training model ...")

  cv.fit(x, y)

  return cv

def main():
  parser = argparse.ArgumentParser(prog="random_forest.py",
                                   formatter_class=argparse.RawTextHelpFormatter,
                                   description="Trains a random forest model.")

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
    "estimator__n_estimators": np.arange(100, 1000, 100),
    "estimator__max_depth": np.arange(10, 100, 10),
    "estimator__min_samples_leaf": np.arange(1, 50, 5),
    "estimator__max_features": [None, "sqrt", "log2"],
    "estimator__random_state": [42],
  }

  model = train_model(df.drop("SalePrice", axis=1), df["SalePrice"], hyperparameters,
                      verbose=5 if args.verbose else 0, n_jobs=-2)

  vprint(f"Saving model to '{args.output}' ...")

  joblib.dump(model, args.output)

if __name__ == "__main__":
  main()
