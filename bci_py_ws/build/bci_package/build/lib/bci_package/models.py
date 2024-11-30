import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class VotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
        print("VotingEnsemble __init__")
        
    def fit(self, X, y):
        for model in self.models:
            print(model)
            model.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=1, 
            arr=predictions)