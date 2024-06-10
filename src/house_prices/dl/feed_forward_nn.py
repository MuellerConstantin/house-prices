"""
Module for training a simple neural network regression model.
"""

import argparse
import keras
import joblib
import pandas as pd
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from house_prices.transformation import build_transformer, get_ordinal_feature_mappings

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def create_model(meta):
  """
  Creates a simple feed forward neural network model.
  """

  n_features_in_ = meta["n_features_in_"]

  model = keras.models.Sequential([
    keras.layers.Input(shape=(n_features_in_,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
  ])

  model.compile(optimizer="adam", loss="mean_squared_error", metrics=["root_mean_squared_error"])

  return model

def train_model(x: pd.DataFrame,
                y: pd.Series):
  """
  Trains the neural network regression model.
  """

  vprint("Building model ...")

  ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=get_ordinal_feature_mappings(x), dtype=int)),
  ])

  nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
  ])

  numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
  ])

  estimator = KerasRegressor(model=create_model, epochs=250, batch_size=32)

  transformer = build_transformer(x, ordinal_pipeline, nominal_pipeline, numerical_pipeline)
  model = Pipeline([
    ("transformer", transformer),
    ("estimator", estimator),
  ])

  vprint("Training model ...")

  model.fit(x, y)

  return model

def main():
  parser = argparse.ArgumentParser(prog="feed_forward_nn.py",
                                   formatter_class=argparse.RawTextHelpFormatter,
                                   description="Trains a simple neural network regression model.")

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
