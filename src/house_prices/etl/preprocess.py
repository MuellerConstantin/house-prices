"""
Module for preparing data for modeling.
"""

import pandas as pd
import argparse

# pylint: disable=unnecessary-lambda-assignment
vprint = lambda *a, **k: None

def preprocess(df: pd.DataFrame):
  """
  Preprocess data for modeling.
  """

  vprint("Preprocessing data ...")

  # Building properties

  df["MSSubClass"] = df["MSSubClass"].replace({
    20: "1S1946-NEW", # 1-STORY 1946 & NEWER ALL STYLES
    30: "1S1945-OLD", # 1-STORY 1945 & OLDER
    40: "1SFIN-ALL", # 1-STORY W/FINISHED ATTIC ALL AGES
    45: "1-1/2UNF-ALL", # 1-1/2 STORY - UNFINISHED ALL AGES
    50: "1-1/2FIN-ALL", # 1-1/2 STORY FINISHED ALL AGES
    60: "2S1946-NEW", # 2-STORY 1946 & NEWER
    70: "2S1945-OLD", # 2-STORY 1945 & OLDER
    75: "2-1/2S-ALL", # 2-1/2 STORY ALL AGES
    80: "SPLT-ALL", # SPLIT OR MULTI-LEVEL
    85: "SPLT-FOYER", # SPLIT FOYER
    90: "DUPLEX-ALL", # DUPLEX - ALL STYLES AND AGES
    120: "1SPUD1946-NEW", # 1-STORY PUD (Planned Unit Development) - 1946 & NEWER
    150: "1-1/2SPUD-ALL", # 1-1/2 STORY PUD - ALL AGES
    160: "2SPUD1946-NEW", # 2-STORY PUD - 1946 & NEWER
    180: "PUD-ALL", # PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
    190: "2FAM-ALL"}) # 2 FAMILY CONVERSION - ALL STYLES AND AGES

  df["MasVnrType"] = df["MasVnrType"].fillna("None")

  # Lot properties

  df["LotFrontage"] = df["LotFrontage"].fillna(0)

  # Utility properties

  df["Alley"] = df["Alley"].fillna("None")

  # Neighborhood properties

  # Garage properties

  df.loc[df["GarageQual"].isna(), "GarageFinish"] = "None"
  df.loc[df["GarageQual"].isna(), "GarageType"] = "None"
  df.loc[df["GarageQual"].isna(), "GarageCond"] = "None"
  df.loc[df["GarageQual"].isna(), "GarageArea"] = 0
  df.loc[df["GarageQual"].isna(), "GarageCars"] = 0
  df.loc[df["GarageQual"].isna(), "GarageYrBlt"] = 0
  df["GarageQual"] = df["GarageQual"].fillna("None")

  # Supplies properties

  # Basement properties

  df.loc[df["BsmtQual"].isna(), "BsmtCond"] = "None"
  df.loc[df["BsmtQual"].isna(), "BsmtExposure"] = "None"
  df.loc[df["BsmtQual"].isna(), "BsmtFinType1"] = "None"
  df.loc[df["BsmtQual"].isna(), "BsmtFinType2"] = "None"
  df.loc[df["BsmtQual"].isna(), "BsmtFinSF1"] = 0
  df.loc[df["BsmtQual"].isna(), "BsmtFinSF2"] = 0
  df.loc[df["BsmtQual"].isna(), "BsmtUnfSF"] = 0
  df.loc[df["BsmtQual"].isna(), "TotalBsmtSF"] = 0
  df.loc[df["BsmtQual"].isna(), "BsmtFullBath"] = 0
  df.loc[df["BsmtQual"].isna(), "BsmtHalfBath"] = 0
  df["BsmtQual"] = df["BsmtQual"].fillna("None")

  # Outdoor area properties

  df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
  df["PoolQC"] = df["PoolQC"].fillna("None")
  df["Fence"] = df["Fence"].fillna("None")
  df["MiscFeature"].fillna("None", inplace=True)

  # Kitchen properties

  df = df.dropna()

  return df

def main():
  parser = argparse.ArgumentParser(prog="preprocess.py",
                                   formatter_class=argparse.RawTextHelpFormatter,
                                   description="Preprocess data for modeling.")

  parser.add_argument("-v", "--verbose", action="store_true",
                      help="Print out verbose messages.")
  parser.add_argument("-i", "--input", type=str, required=True,
                      help="Input file path. Must be a CSV file.")
  parser.add_argument("-o", "--output", type=str, required=True,
                      help="Output file path. Must be a CSV file.")

  args = parser.parse_args()

  if args.verbose:
    global vprint
    vprint = print

  vprint(f"Loading data from ({args.input}) ...")

  df = pd.read_csv(args.input)
  df = preprocess(df)

  vprint(f"Saving data to ({args.output}) ...")

  df.to_csv(args.output, index=False)

if __name__ == "__main__":
  main()
