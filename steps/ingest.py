"""
This module defines the following routines used by the 'ingest' step:

- ``load_dataset``: Defines customizable logic for parsing the dataset
"""
import pandas as pd


def load_dataset(location: str, *args) -> pd.DataFrame:
    """
    Load content from the specified dataset file as a `pd.DataFrame`.

    This method is used to load dataset types that are not natively  managed by MLflow Recipes
    (datasets that are not in Parquet, Delta Table, or Spark SQL Table format). This method is
    called once for each file in the dataset, and MLflow Recipes automatically combines the
    resulting DataFrames together.

    :param location: The path to the dataset file.
    :args: The dataset reading using method
    :return: pd.DataFrame representing the content of the specified file.
    """
    columns = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Fare", "Age", "Embarked"]
    df = pd.read_csv(location, index_col=0)
    return df[columns].copy(deep=True)


def load_testset(location: str, *args) -> pd.DataFrame:
    """
    Load the test set as a `pd.DataFrame`

    :param location: The path to the dataset file.
    :args: The dataset reading using method
    :return: pd.DataFrame representing the content of the specified file.
    """
    columns = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age", "Embarked"]
    df = pd.read_csv(location, index_col=0)
    return df[columns].copy(deep=True)
