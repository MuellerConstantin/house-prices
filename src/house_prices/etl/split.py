"""
Module for splitting the data into train and test/validation sets.
"""

import argparse
import pandas as pd
import dvc.api
from sklearn.model_selection import train_test_split

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

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
  train_df, test_df = train_test_split(df, test_size=params["split"]["test_size"], random_state=params["split"]["seed"])

  vprint(f"Saving data to '{args.train}' and '{args.test}' ...")

  train_df.to_csv(args.train, index=False)
  test_df.to_csv(args.test, index=False)

if __name__ == "__main__":
  main()
