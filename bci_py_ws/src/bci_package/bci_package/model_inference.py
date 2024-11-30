import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import pickle

# Dataschema for pandas

data_schema_features = {
    'TimeStamp': 'datetime64[ns]',
    'Delta_TP9': 'float64',
    'Delta_AF7': 'float64',
    'Delta_AF8': 'float64',
    'Delta_TP10': 'float64',
    'Theta_TP9': 'float64',
    'Theta_AF7': 'float64',
    'Theta_AF8': 'float64',
    'Theta_TP10': 'float64',
    'Alpha_TP9': 'float64',
    'Alpha_AF7': 'float64',
    'Alpha_AF8': 'float64',
    'Alpha_TP10': 'float64',
    'Beta_TP9': 'float64',
    'Beta_AF7': 'float64',
    'Beta_AF8': 'float64',
    'Beta_TP10': 'float64',
    'Gamma_TP9': 'float64',
    'Gamma_AF7': 'float64',
    'Gamma_AF8': 'float64',
    'Gamma_TP10': 'float64',
}

data_schema_labels = 'int64'  

left_hand_path = "/home/sachin33sj/BCI project/bci-ros2/Trial_2/Left_hand"
right_hand_path = "/home/sachin33sj/BCI project/bci-ros2/Trial_2/Right_hand"
columns = ["TimeStamp", "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
    "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
    "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
    "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
    "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10"]

class CreateDataset:
    def __init__(self, path, columns, labels, data_schema_features, data_schema_labels):
        self.path = path
        self.columns = columns
        self.labels = labels
        self.data_schema_features = data_schema_features
        self.data_schema_labels = data_schema_labels

    def create_dataset(self):
        dataset = pd.DataFrame(columns=self.columns)
        for file in os.listdir(self.path):
            if file.endswith(".csv"):
                file_path = os.path.join(self.path, file)
                data = pd.read_csv(file_path, 
                                dtype={col: self.data_schema_features[col] for col in self.columns if col != 'TimeStamp'})
                # Convert TimeStamp using mixed format
                # data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], format='mixed')
                data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')

                data['TimeStamp'] = data['TimeStamp'].dt.second + data['TimeStamp'].dt.microsecond/1e6
                data = data[self.columns]
                dataset = pd.concat([dataset, data], axis=0)
                dataset = dataset.dropna(thresh=len(self.columns)-1)
        return dataset

    def create_labeled_dataset(self):
        features = self.create_dataset()
        labels = pd.DataFrame([self.labels[0]] * features.shape[0], columns = ["Label"], dtype = self.data_schema_labels)
        return features, labels
        

lh_features, lh_labels = CreateDataset(left_hand_path, columns, [0], data_schema_features, data_schema_labels).create_labeled_dataset()
rh_features, rh_labels = CreateDataset(right_hand_path, columns, [1], data_schema_features, data_schema_labels).create_labeled_dataset()

X_test = pd.concat([lh_features, rh_features], axis = 0)
Y_test = pd.concat([lh_labels, rh_labels], axis = 0)

    
from sklearn.preprocessing import StandardScaler

def load_scaler(path: str='/home/sachin33sj/BCI project/bci-ros2/bci_py_ws/src/bci_package/weights/scaler.pkl') -> StandardScaler:
    """Load saved scaler from disk"""
    with open(path, 'rb') as f:
        return pickle.load(f)
    
scaler = load_scaler()
    
X_test.iloc[:, 1:] = scaler.transform(X_test.iloc[:, 1:])

from sklearn.base import BaseEstimator, ClassifierMixin
import pickle

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
    
    def save(self, filename):
        """Saves the ensemble model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

from sklearn.metrics import accuracy_score

# Load the pre-trained ensemble model
MODEL_PATH = '/home/sachin33sj/BCI project/bci-ros2/bci_py_ws/src/bci_package/weights/ensemble_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    ensemble_model = pickle.load(f)
        
# Store predictions
y_pred = ensemble_model.predict(X_test.iloc[:, 1:])

# Calculate accuracy
print(accuracy_score(Y_test, y_pred))