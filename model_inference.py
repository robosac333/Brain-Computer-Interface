from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np

class VotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        for model in self.models:
            print(model)
            model.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        # Majority voting
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=1, 
            arr=predictions)
    
clf = LogisticRegression(random_state=0, max_iter=1000)
neigh = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear')
rf = RandomForestClassifier(max_depth=2, random_state=0)
dt = DecisionTreeClassifier(random_state=0)

from sklearn.metrics import accuracy_score

ensemble_model = VotingEnsemble(models=[clf, neigh, rf, dt])
ensemble_model.fit(X_train.iloc[:, 1:], y_train.values.ravel())

# Store predictions
y_pred = ensemble_model.predict(X_test.iloc[:, 1:])

# Calculate accuracy
print(accuracy_score(y_test, y_pred))


