"""
This module defines the following routines used by the 'transform' step:

- ``transformer_fn``: Defines transforming logic to use before the estimator.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    categorical_features = ["Pclass", "Sex", "Embarked"]
    numeric_features = ["SibSp", "Parch", "Fare", "Age"]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # return preprocessor
    return ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(categories="auto"), categorical_features),
            ("ordinal", OrdinalEncoder(categories="auto"), categorical_features),
            ("numerical", numeric_pipeline, numeric_features),
        ]
    )
