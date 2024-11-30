#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import pickle
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from bci_package.models import VotingEnsemble


# class VotingEnsemble(BaseEstimator, ClassifierMixin):
#     def __init__(self, models):
#         self.models = models
        
#     def fit(self, X, y):
#         for model in self.models:
#             print(model)
#             model.fit(X, y)
#         return self
        
#     def predict(self, X):
#         predictions = np.column_stack([
#             model.predict(X) for model in self.models
#         ])
#         return np.apply_along_axis(
#             lambda x: np.bincount(x).argmax(), 
#             axis=1, 
#             arr=predictions)


class BCIModelPublisher(Node):
    def __init__(self):
        super().__init__('bci_model_publisher')
        
        # Create publisher
        self.publisher_ = self.create_publisher(String, 'bci_commands', 10)
        
        # Initialize paths
        self.left_hand_path = "/home/sachin33sj/BCI project/bci-ros2/Trial_2/Left_hand"
        self.right_hand_path = "/home/sachin33sj/BCI project/bci-ros2/Trial_2/Right_hand"
        
        # Initialize data schema
        self.data_schema_features = {
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
            'Gamma_TP20': 'float64',
        }
        
        self.columns = ["TimeStamp", "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
                       "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
                       "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
                       "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
                       "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP20"]

        # Initialize data iterator
        self.current_file_index = 0
        self.current_row_index = 0
        self.current_dataset = None
        self.file_list = []
        
        # Load models and file list
        self.load_models()
        self.load_file_list()
        
        # Set up timer for periodic predictions (every 1 second)
        self.timer = self.create_timer(1.0, self.predict_and_publish)
        
        print('BCI Model Publisher Node has started')
        self.get_logger().info('BCI Model Publisher Node has started')

    def load_models(self):
        try:
            # Get the path to the source directory where weights are stored
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            SCALER_PATH = os.path.join(package_dir, 'weights', 'scaler.pkl')
            MODEL_PATH = os.path.join(package_dir, 'weights', 'ensemble_model.pkl')  # Note: using ensemble_model.pkl instead of ensemble_models.pkl
            
            self.get_logger().info(f'Loading scaler from: {SCALER_PATH}')
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)

            # Load trained models and create new ensemble
            self.get_logger().info(f'Loading models from: {MODEL_PATH}')
            with open(MODEL_PATH, 'rb') as f:
                trained_models = pickle.load(f)
            
            # Create new ensemble with loaded models
            # self.ensemble_model = VotingEnsemble(models=trained_models)
            
            self.get_logger().info('Models loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Error loading models: {str(e)}')
            raise

    def load_file_list(self):
        try:
            # Get list of all CSV files from both directories
            left_files = [os.path.join(self.left_hand_path, f) 
                         for f in os.listdir(self.left_hand_path) 
                         if f.endswith('.csv')]
            right_files = [os.path.join(self.right_hand_path, f) 
                          for f in os.listdir(self.right_hand_path) 
                          if f.endswith('.csv')]
            self.file_list = left_files + right_files
            print(f'Loaded {len(self.file_list)} files for processing')
            self.get_logger().info(f'Loaded {len(self.file_list)} files for processing')
        except Exception as e:
            print(f'Error loading file list: {str(e)}')
            self.get_logger().error(f'Error loading file list: {str(e)}')
            raise

    def get_current_eeg_data(self):
        try:
            # If we haven't loaded any data yet or have finished current dataset
            if self.current_dataset is None or self.current_row_index >= len(self.current_dataset):
                # Move to next file
                if self.current_file_index >= len(self.file_list):
                    # Reset to start if we've processed all files
                    self.current_file_index = 0
                    print('Restarting dataset from beginning')
                    self.get_logger().info('Restarting dataset from beginning')
                
                # Load new file
                file_path = self.file_list[self.current_file_index]
                print(f'Loading new file: {file_path}')
                self.get_logger().info(f'Loading new file: {file_path}')
                
                # Read and process the file
                data = pd.read_csv(file_path, 
                                 dtype={col: self.data_schema_features[col] 
                                      for col in self.columns 
                                      if col != 'TimeStamp'})
                
                # Process timestamp
                data['TimeStamp'] = pd.to_datetime(data['TimeStamp'], 
                                                 format='%Y-%m-%d %H:%M:%S.%f')
                data['TimeStamp'] = data['TimeStamp'].dt.second + data['TimeStamp'].dt.microsecond/1e6
                
                self.current_dataset = data
                self.current_row_index = 0
                self.current_file_index += 1

            # Get current row
            current_row = self.current_dataset.iloc[[self.current_row_index]]
            self.current_row_index += 1
            
            return current_row
        except Exception as e:
            print(f'Error getting EEG data: {str(e)}')
            self.get_logger().error(f'Error getting EEG data: {str(e)}')
            raise

    def process_data(self, data):
        # Scale the features (excluding timestamp)
        scaled_features = self.scaler.transform(data.iloc[:, 1:])
        return scaled_features

    def prediction_to_command(self, prediction):
        # Convert model prediction to robot command
        if prediction == 0:
            return 'move left'  # For left hand
        else:
            return 'move right'  # For right hand

    def predict_and_publish(self):
        try:
            # Get single row of data
            current_data = self.get_current_eeg_data()
            
            # Process the data
            processed_data = self.process_data(current_data)
            
            # Make prediction
            prediction = self.ensemble_model.predict(processed_data)
            
            # Convert prediction to command
            command = self.prediction_to_command(prediction[0])
            
            # Create and publish message
            msg = String()
            msg.data = command
            self.publisher_.publish(msg)
            
            # Print prediction information
            print(f'Prediction: {prediction[0]} -> Command: {command}')
            print(f'File {self.current_file_index}/{len(self.file_list)}, Row {self.current_row_index}')
            
            self.get_logger().info(
                f'Published command: {command} '
                f'(File {self.current_file_index}/{len(self.file_list)}, '
                f'Row {self.current_row_index})'
            )
            
        except Exception as e:
            print(f'Error in prediction: {str(e)}')
            self.get_logger().error(f'Error in prediction: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = BCIModelPublisher()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
