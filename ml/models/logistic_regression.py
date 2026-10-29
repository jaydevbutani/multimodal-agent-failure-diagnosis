"""
Logistic Regression Baseline Model Definition
"""
from sklearn.linear_model import LogisticRegression
from ml.training.config import LOGISTIC_REGRESSION_PARAMS

def create_logistic_regression_model():
    """
    Instantiates the audited Logistic Regression baseline model with explicit hyperparameters.
    """
    return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
