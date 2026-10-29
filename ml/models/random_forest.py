"""
Random Forest Classifier Model Definition
"""
from sklearn.ensemble import RandomForestClassifier
from ml.training.config import RANDOM_FOREST_PARAMS

def create_random_forest_model():
    """
    Instantiates the audited Random Forest classifier with explicit hyperparameters.
    """
    return RandomForestClassifier(**RANDOM_FOREST_PARAMS)
