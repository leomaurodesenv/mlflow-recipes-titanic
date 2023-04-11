"""
This module defines the following routines used by the 'transform' step:

- ``transformer_fn``: Defines transforming logic to use before the estimator.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


def return_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    pass


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    return Pipeline(
        steps=[
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "onehot",
                            OneHotEncoder(categories="auto"),
                            ["Sex", "Embarked"],
                        ),
                        (
                            "ordinal",
                            OrdinalEncoder(categories="auto"),
                            ["Sex", "Embarked"],
                        ),
                    ]
                ),
            ),
        ]
    )
