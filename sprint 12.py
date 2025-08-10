# libraries
import pandas as pd
import numpy as np
import datetime as dt
from DataProcessor import DataProcessor
from DataModeler2 import DataModeler2

# Load the data
train_features_vectorized = pd.read_csv('car_data_train_features_vectorized.csv', header=None)
train_target_vectorized = pd.read_csv('car_data_train_target_vectorized.csv', header=None)
valid_features_vectorized = pd.read_csv('car_data_valid_features_vectorized.csv', header=None)
valid_target_vectorized = pd.read_csv('car_data_valid_target_vectorized.csv', header=None)
print(f"Loaded training features shape: {train_features_vectorized.shape}")
print(f"Loaded training target shape: {train_target_vectorized.shape}")
print(f"Loaded validation features shape: {valid_features_vectorized.shape}")
print(f"Loaded validation target shape: {valid_target_vectorized.shape}")
print(f"Loaded training features shape: {train_features_vectorized.head(1)}")
print(f"Loaded training target shape: {train_target_vectorized.head(1)}")
print(f"Loaded validation features shape: {valid_features_vectorized.head(1)}")
print(f"Loaded validation target shape: {valid_target_vectorized.head(1)}")
# Saving data
np.savetxt('car_data_train_features_vectorized.csv', train_features_vectorized, delimiter=',', fmt='%.6f')
np.savetxt('car_data_train_target_vectorized.csv', train_target_vectorized, delimiter=',', fmt='%.6f')
np.savetxt('car_data_valid_features_vectorized.csv', valid_features_vectorized, delimiter=',', fmt='%.6f')
np.savetxt('car_data_valid_target_vectorized.csv', valid_target_vectorized, delimiter=',', fmt='%.6f')

# Numpy: Loading data
train_features_vectorized = np.loadtxt('car_data_train_features_vectorized.csv', delimiter=',', dtype=np.float32)
train_target_vectorized = np.loadtxt('car_data_train_target_vectorized.csv', delimiter=',', dtype=np.float32)
valid_features_vectorized = np.loadtxt('car_data_valid_features_vectorized.csv', delimiter=',', dtype=np.float32)
valid_target_vectorized = np.loadtxt('car_data_valid_target_vectorized.csv', delimiter=',', dtype=np.float32)

train_features_vectorized = np.ascontiguousarray(train_features_vectorized)
train_target_vectorized = np.ascontiguousarray(train_target_vectorized)
valid_features_vectorized = np.ascontiguousarray(valid_features_vectorized)
valid_target_vectorized = np.ascontiguousarray(valid_target_vectorized)

# Verify shapes
print(f"Loaded training features shape: {train_features_vectorized.shape}")
print(f"Loaded training target shape: {train_target_vectorized.shape}")
print(f"Loaded validation features shape: {valid_features_vectorized.shape}")
print(f"Loaded validation target shape: {valid_target_vectorized.shape}")

dm = DataModeler2()

dm.fit(
    train_features_vectorized=train_features_vectorized,
    train_target_vectorized=train_target_vectorized,
    model_type='regression',
    model_name='xgboost'
    )

scores = dm.score(
    valid_features_vectorized=valid_features_vectorized,
    valid_target_vectorized=valid_target_vectorized,
    metric='RMSE',
    verbose=True)
