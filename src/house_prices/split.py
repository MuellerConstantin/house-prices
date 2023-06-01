"""
Module for splitting the data into train and test/validation sets.
"""

import argparse
import pandas as pd
import numpy as np
import dvc.api
from sklearn.model_selection import train_test_split

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def combine_single_value_bins(y_binned):
  """
  Combine bins with only one value with the closest bin.
  """

  levels = np.unique(y_binned)

  for level_index, level in enumerate(levels):
    level_count = np.sum(y_binned == level)

    if level_count == 1:
      prev_level_index = level_index - 1
      next_level_index = level_index + 1

      while prev_level_index >= 0 and np.sum(y_binned == levels[prev_level_index]) == 0:
        prev_level_index -= 1

      while next_level_index < len(levels) and np.sum(y_binned == levels[next_level_index]) == 0:
        next_level_index += 1

      prev_level_exists = prev_level_index >= 0
      next_level_exists = next_level_index < len(levels)

      prev_level = levels[prev_level_index] if prev_level_exists else None
      next_level = levels[next_level_index] if next_level_exists else None

      prev_level_count = np.sum(y_binned == prev_level) if prev_level_exists else None
      next_level_count = np.sum(y_binned == next_level) if next_level_exists else None

      if prev_level_exists and next_level_exists:
        y_binned[y_binned == level] = prev_level if prev_level_count < next_level_count else next_level
      elif prev_level_exists:
        y_binned[y_binned == level] = prev_level
      elif next_level_exists:
        y_binned[y_binned == level] = next_level

  return y_binned

def stratified_train_test_split(x, y, test_size=0.2, random_state=None):
  """
  Perform stratified train-test split for continuous target variables.
  """

  num_bins = int(np.sqrt(len(y))) + 1
  bins = np.linspace(min(y), max(y), num_bins)

  y_binned = np.digitize(y, bins)
  y_binned = combine_single_value_bins(y_binned)

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                      random_state=random_state, stratify=y_binned)

  return x_train, x_test, y_train, y_test

def main():
  parser = argparse.ArgumentParser(prog="split.py",
                                   formatter_class=argparse.RawTextHelpFormatter,
                                   description="Split data into train and test/validation sets.")

  parser.add_argument("-v", "--verbose", action="store_true",
                      help="Print out verbose messages.")
  parser.add_argument("-i", "--input", type=str, required=True,
                      help="Input file path. Must be a CSV file.")
  parser.add_argument("-t", "--train", type=str, required=True,
                      help="Output file path of train set. Must be a CSV file.")
  parser.add_argument("-e", "--test", type=str, required=True,
                      help="Output file path of test/validation set. Must be a CSV file.")

  args = parser.parse_args()

  if args.verbose:
    global vprint
    vprint = print

  vprint(f"Loading data from '{args.input}' ...")

  df = pd.read_csv(args.input)
  params = dvc.api.params_show()

  x_train, x_test, y_train, y_test = stratified_train_test_split(df.drop("SalePrice", axis=1), df["SalePrice"],
                                                                 test_size=params["split"]["test_size"],
                                                                 random_state=params["split"]["random_state"])

  train_df = pd.concat([x_train, y_train], axis=1)
  test_df = pd.concat([x_test, y_test], axis=1)

  vprint(f"Saving data to '{args.train}' and '{args.test}' ...")

  train_df.to_csv(args.train, index=False)
  test_df.to_csv(args.test, index=False)

if __name__ == "__main__":
  main()
