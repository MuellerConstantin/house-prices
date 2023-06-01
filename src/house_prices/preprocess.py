"""
Module for preparing data for modeling. It contains functions which are designed for the preprocessing
of the data set of this project.
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

  # Building properties transformations

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

  df["MasVnrType"] = df["MasVnrType"].fillna("XX")

  df["OverallCond"] = df["OverallCond"].replace({
    10: "VEx",
    9: "Ex",
    8: "VGd",
    7: "Gd",
    6: "AAvg",
    5: "Avg",
    4: "BAvg",
    3: "Fa",
    2: "Po",
    1: "VPo"})

  df["OverallQual"] = df["OverallQual"].replace({
    10: "VEx",
    9: "Ex",
    8: "VGd",
    7: "Gd",
    6: "AAvg",
    5: "Avg",
    4: "BAvg",
    3: "Fa",
    2: "Po",
    1: "VPo"})

  # Lot properties transformations

  df["LotFrontage"] = df["LotFrontage"].fillna(0)

  # Utility properties transformations

  df["Alley"] = df["Alley"].fillna("XX")

  # Neighborhood properties transformations

  # Garage properties transformations

  df.loc[df["GarageQual"].isna(), "GarageFinish"] = "XX"
  df.loc[df["GarageQual"].isna(), "GarageType"] = "XX"
  df.loc[df["GarageQual"].isna(), "GarageCond"] = "XX"
  df.loc[df["GarageQual"].isna(), "GarageArea"] = 0
  df.loc[df["GarageQual"].isna(), "GarageCars"] = 0
  df.loc[df["GarageQual"].isna(), "GarageYrBlt"] = 0
  df["GarageQual"] = df["GarageQual"].fillna("XX")

  # Supplies properties transformations

  # Basement properties transformations

  df.loc[df["BsmtQual"].isna(), "BsmtCond"] = "XX"
  df.loc[df["BsmtQual"].isna(), "BsmtExposure"] = "XX"
  df.loc[df["BsmtQual"].isna(), "BsmtFinType1"] = "XX"
  df.loc[df["BsmtQual"].isna(), "BsmtFinType2"] = "XX"
  df.loc[df["BsmtQual"].isna(), "BsmtFinSF1"] = 0
  df.loc[df["BsmtQual"].isna(), "BsmtFinSF2"] = 0
  df.loc[df["BsmtQual"].isna(), "BsmtUnfSF"] = 0
  df.loc[df["BsmtQual"].isna(), "TotalBsmtSF"] = 0
  df.loc[df["BsmtQual"].isna(), "BsmtFullBath"] = 0
  df.loc[df["BsmtQual"].isna(), "BsmtHalfBath"] = 0
  df["BsmtQual"] = df["BsmtQual"].fillna("XX")

  # Outdoor area properties transformations

  df["FireplaceQu"] = df["FireplaceQu"].fillna("XX")
  df["PoolQC"] = df["PoolQC"].fillna("XX")
  df["Fence"] = df["Fence"].fillna("XX")
  df["MiscFeature"] = df["MiscFeature"].fillna("XX")

  # Kitchen properties transformations

  # General clean up and feature transformations

  df["Age"] = df["YrSold"] - df["YearBuilt"]
  df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
  df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]
  df["TotalBathAbvGr"] = df["FullBath"] + df["HalfBath"] * 0.5
  df["TotalBathBsmt"] = df["BsmtFullBath"] + df["BsmtHalfBath"] * 0.5
  df["AvgRoomSF"] = round(df["GrLivArea"] / (df["TotRmsAbvGrd"] + df["TotalBathAbvGr"]), 2)

  df = df.dropna()
  df = df.drop(columns=["Id"])

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

  vprint(f"Loading data from '{args.input}' ...")

  df = pd.read_csv(args.input)
  df = preprocess(df)

  vprint(f"Saving data to '{args.output}' ...")

  df.to_csv(args.output, index=False)

if __name__ == "__main__":
  main()
