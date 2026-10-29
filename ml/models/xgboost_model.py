"""
XGBoost Classifier Model Definition
"""
from xgboost import XGBClassifier
from ml.training.config import XGBOOST_PARAMS

def create_xgboost_model():
    """
    Instantiates the audited XGBoost classifier with explicit hyperparameters.
    """
    return XGBClassifier(**XGBOOST_PARAMS)
