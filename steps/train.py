"""
This module defines the following routines used by the 'train' step:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Any, Dict

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.model = LogisticRegression(random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def estimator_fn(estimator_params: Dict[str, Any] = None) -> Any:
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    return TemplateClassifier()
