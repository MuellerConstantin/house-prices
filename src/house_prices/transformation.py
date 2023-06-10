"""
Module contains utility functions for data transformation.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

EXCLUDED_FEATURES = ["Id"]

ORDINAL_FEATURE_MAPPINGS = {
  "OverallQual": ["VPo", "Po", "Fa", "BAvg", "Avg", "AAvg", "Gd", "VGd", "Ex", "VEx"],
  "OverallCond": ["VPo", "Po", "Fa", "BAvg", "Avg", "AAvg", "Gd", "VGd", "Ex", "VEx"],
  "OverallGrade": ["VPo", "Po", "Fa", "BAvg", "Avg", "AAvg", "Gd", "VGd", "Ex", "VEx"],
  "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
  "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
  "BsmtQual": ["No", "Po", "Fa", "TA", "Gd", "Ex"],
  "BsmtCond": ["No", "Po", "Fa", "TA", "Gd", "Ex"],
  "BsmtFinType1": ["No", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
  "BsmtFinType2": ["No", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
  "BsmtExposure": ["No", "No", "Mn", "Av", "Gd"],
  "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
  "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
  "FireplaceQu": ["No", "Po", "Fa", "TA", "Gd", "Ex"],
  "GarageQual": ["No", "Po", "Fa", "TA", "Gd", "Ex"],
  "GarageCond": ["No", "Po", "Fa", "TA", "Gd", "Ex"],
  "PoolQC": ["No", "Fa", "TA", "Gd", "Ex"],
  "LandSlope": ["Sev", "Mod", "Gtl"],
  "LotShape": ["IR3", "IR2", "IR1", "Reg"],
  "PavedDrive": ["N", "P", "Y"],
  "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
  "MoSold": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
}

def get_numerical_feature_names(df: pd.DataFrame) -> list:
  """
  Returns a list of numerical features.
  """

  return df.select_dtypes(include=["number"]).columns.difference(EXCLUDED_FEATURES).tolist()

# pylint: disable=unused-argument
def get_ordinal_feature_names(df: pd.DataFrame) -> list:
  """
  Returns a list of ordinal features.
  """

  return [key for key in ORDINAL_FEATURE_MAPPINGS if key not in EXCLUDED_FEATURES]

def get_ordinal_feature_mappings(df: pd.DataFrame) -> list:
  """
  Returns a list of feature mappings for selected ordinal features.
  """

  return [value for key, value in ORDINAL_FEATURE_MAPPINGS.items() if key not in EXCLUDED_FEATURES]

def get_nominal_feature_names(df: pd.DataFrame) -> list:
  """
  Returns a list of nominal features.
  """

  return (df.select_dtypes(include="object").columns.difference(get_ordinal_feature_names(df) + EXCLUDED_FEATURES)
            .tolist())

def build_transformer(df: pd.DataFrame,
                      ordinary_pipeline: Pipeline,
                      nominal_pipeline: Pipeline,
                      numerical_pipeline: Pipeline):
  """
  Builds a transformer for the given data frame.
  """

  ordinal_features = get_ordinal_feature_names(df)
  nominal_features = get_nominal_feature_names(df)
  numerical_features = get_numerical_feature_names(df)

  transformer = ColumnTransformer([
    ("ordinal", ordinary_pipeline, ordinal_features),
    ("nominal", nominal_pipeline, nominal_features),
    ("numerical", numerical_pipeline, numerical_features),
  ])

  return transformer
