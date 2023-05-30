"""
Module contains utility functions for constructing a project specific machine learning pipeline.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

ORDINAL_FEATURE_MAPPINGS = {
  "OverallQual": ["VPo", "Po", "Fa", "BAvg", "Avg", "AAvg", "Gd", "VGd", "Ex", "VEx"],
  "OverallCond": ["VPo", "Po", "Fa", "BAvg", "Avg", "AAvg", "Gd", "VGd", "Ex", "VEx"],
  "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
  "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
  "BsmtQual": ["XX", "Po", "Fa", "TA", "Gd", "Ex"],
  "BsmtCond": ["XX", "Po", "Fa", "TA", "Gd", "Ex"],
  "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
  "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
  "FireplaceQu": ["XX", "Po", "Fa", "TA", "Gd", "Ex"],
  "GarageQual": ["XX", "Po", "Fa", "TA", "Gd", "Ex"],
  "GarageCond": ["XX", "Po", "Fa", "TA", "Gd", "Ex"],
  "PoolQC": ["XX", "Fa", "TA", "Gd", "Ex"],
}

def get_numerical_feature_names(df: pd.DataFrame) -> list:
  """
  Returns a list of numerical features.
  """

  return df.select_dtypes(include="number").columns.tolist()

# pylint: disable=unused-argument
def get_ordinal_feature_names(df: pd.DataFrame) -> list:
  """
  Returns a list of ordinal features.
  """

  return list(ORDINAL_FEATURE_MAPPINGS.keys())

def get_binary_feature_names(df: pd.DataFrame) -> list:
  """
  Returns a list of binary features.
  """

  return df.select_dtypes(include="object").columns.difference(get_ordinal_feature_names(df)).tolist()

def build_transformer(df: pd.DataFrame,
                      ordinary_pipeline: Pipeline,
                      binary_pipeline: Pipeline,
                      numerical_pipeline: Pipeline):
  """
  Builds a transformer for the given data frame.
  """

  ordinal_features = get_ordinal_feature_names(df)
  binary_features = get_binary_feature_names(df)
  numerical_features = get_numerical_feature_names(df)

  transformer = ColumnTransformer([
    ("ordinal", ordinary_pipeline, ordinal_features),
    ("binary", binary_pipeline, binary_features),
    ("numerical", numerical_pipeline, numerical_features),
  ])

  return transformer
