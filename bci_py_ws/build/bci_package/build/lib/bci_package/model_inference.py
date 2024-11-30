import numpy as np
import pandas as pd
import os
import pickle
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sklearn.preprocessing import StandardScaler
from .models import VotingEnsemble  # Import from your existing models.py

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

# Get package directory for weights
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
left_hand_path = "/home/abhishek/Brain-Computer-Interface/Trial_2/Left_hand"
right_hand_path = "/home/abhishek/Brain-Computer-Interface/Trial_2/Right_hand"
columns = ["TimeStamp", "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
          "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
          "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
          "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
          "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP20"]

class CreateDataset:
    def __init__(self, path, columns, labels, data_schema_features, data_schema_labels):
        self.path = path
        self.columns = columns
        self.labels = labels
        self.data_schema_features = data_schema_features
        self.data_schema_labels = data_schema_labels

    def create_dataset(self):
        all_data = []
        for file in os.listdir(self.path):
            if file.endswith(".csv"):
                file_path = os.path.join(self.path, file)
                data = pd.read_csv(file_path, 
                                dtype={col: self.data_schema_features[col] 
                                      for col in self.columns if col != 'TimeStamp'})
                
                data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], format='mixed')
                data['TimeStamp'] = data['TimeStamp'].dt.second + data['TimeStamp'].dt.microsecond/1e6
                
                data = data[self.columns].dropna(how='all')
                
                if not data.empty:
                    all_data.append(data)
        
        if all_data:
            dataset = pd.concat(all_data, axis=0, ignore_index=True)
            return dataset.dropna(thresh=len(self.columns)-1)
        return pd.DataFrame(columns=self.columns)

    def create_labeled_dataset(self):
        features = self.create_dataset()
        labels = pd.DataFrame([self.labels[0]] * features.shape[0], 
                            columns=["Label"], 
                            dtype=self.data_schema_labels)
        return features, labels

def load_scaler(path: str=os.path.join(PACKAGE_DIR, 'weights', 'scaler.pkl')) -> StandardScaler:
    with open(path, 'rb') as f:
        return pickle.load(f)

class BCIPredictor(Node):
    def __init__(self):
        super().__init__('bci_predictor')
        self.prediction_publisher = self.create_publisher(
            String,
            'bci_predictions',
            10
        )
        self.get_logger().info('BCI Predictor Node has been started')
        self.process_and_predict()

    def process_and_predict(self):
        try:
            # Create datasets
            lh_features, lh_labels = CreateDataset(
                left_hand_path, columns, [0], 
                data_schema_features, data_schema_labels
            ).create_labeled_dataset()
            
            rh_features, rh_labels = CreateDataset(
                right_hand_path, columns, [1], 
                data_schema_features, data_schema_labels
            ).create_labeled_dataset()

            X_test = pd.concat([lh_features, rh_features], axis=0)
            Y_test = pd.concat([lh_labels, rh_labels], axis=0)

            # Load and apply scaler
            scaler = load_scaler()
            X_test.iloc[:, 1:] = scaler.transform(X_test.iloc[:, 1:])

            # Load model and predict
            MODEL_PATH = os.path.join(PACKAGE_DIR, 'weights', 'ensemble_model.pkl')
            self.get_logger().info(f'Loading model from: {MODEL_PATH}')
            
            with open(MODEL_PATH, 'rb') as f:
                ensemble_model = pickle.load(f)

            y_pred = ensemble_model.predict(X_test.iloc[:, 1:])

            # Publish predictions
            for timestamp, prediction in zip(X_test['TimeStamp'], y_pred):
                msg = String()
                msg.data = f"Time: {timestamp:.2f}s - {'Left Hand' if prediction == 0 else 'Right Hand'}"
                self.prediction_publisher.publish(msg)
                self.get_logger().info(f'Published: {msg.data}')
                
        except Exception as e:
            self.get_logger().error(f'Error in process_and_predict: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    bci_predictor = BCIPredictor()
    
    try:
        rclpy.spin(bci_predictor)
    except KeyboardInterrupt:
        print('Stopped by keyboard interrupt')
    finally:
        bci_predictor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()