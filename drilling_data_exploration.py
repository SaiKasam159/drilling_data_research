# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:14:35 2025

@author: ezask15
"""

import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Model


training_data_path = 'C:/Users/ezask15/Downloads/drilling_data/20250513_130643/0/training'
anomalous_data_path = 'C:/Users/ezask15/Downloads/drilling_data/20250513_130643/0/anomalies'

column_names = ["Index", "Time", "INORM", "IEFF", "TCMD", "POSF"]

# Defining the AutoEncoder class
class AutoEncoder(Model):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # Defining the encoder and decoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])
    
        self.decoder = tf.keras.Sequential([
          layers.Dense(16, activation="relu"),
          layers.Dense(32, activation="relu"),
          layers.Dense(16000, activation="sigmoid")])
    
    # Defining the predict function
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AutoEncoder()
scaler = MinMaxScaler()
data_dict = {}
X = None

# ------------------------------------ #

print('Loading the data')
for file_name in os.listdir(training_data_path):
    file_path = os.path.join(training_data_path, file_name)
    if file_name.endswith(".csv"):
        try:
            
            # Read data from line 4 onward (skip metadata lines)
            df = pd.read_csv(file_path, skiprows=3, names=column_names)

            # Drop rows with missing or non-numeric time/inorm values
            df = df[["Time", "INORM"]].apply(pd.to_numeric, errors='coerce').dropna()
            inorm = pd.to_numeric(df["INORM"], errors='coerce').dropna().values
            
            if len(inorm) == 16000:
                inorm = inorm[:16000].reshape(-1, 1)
                inorm_scaled = scaler.fit_transform(inorm).flatten()
                data_dict[file_name] = inorm_scaled.reshape(1, -1)
            
        except Exception as e:
            print(f"Received the error {e}")
            
# ------------------------------------ #

print('Training the AutoEncoder using Cross Validation')

errors = []

for i in range(10):
    
    test_data = random.choice(list(data_dict.keys()))
    train_data = [f for f in data_dict if f != test_data]

    
    X = np.vstack([data_dict[f] for f in train_data])
        
    autoencoder = AutoEncoder()
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=1000, verbose=0)
    
    # Evaluate on test file
    X_test = data_dict[test_data]
    reconstructed = autoencoder.predict(X_test)
    mse = mean_squared_error(X_test.flatten(), reconstructed.flatten())
    errors.append(mse)

    print(f"Iteration {i+1}: Tested on {test_data}, MSE = {mse:.6f}")
    
avg_train_mse = np.mean(errors)
print(f"\nAverage MSE on the Training Data: {avg_train_mse:.6f}")
plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-')

# ------------------------------------ #

print('Viewing the results')

errors = []

for file_name in os.listdir(anomalous_data_path):
    file_path = os.path.join(anomalous_data_path, file_name)
    if file_name.endswith(".csv"):
        try:
            
            # Read data from line 4 onward (skip metadata lines)
            df = pd.read_csv(file_path, skiprows=3, names=column_names)

            # Drop rows with missing or non-numeric time/inorm values
            df = df[["Time", "INORM"]].apply(pd.to_numeric, errors='coerce').dropna()
            inorm = pd.to_numeric(df["INORM"], errors='coerce').dropna().values

            X = np.array(inorm, dtype=np.float32)
            X = X.reshape(-1, 1)
            X = scaler.fit_transform(X).flatten()
            X = X.reshape(1, -1)

            reconstructed = autoencoder.predict(X)
            mse = mean_squared_error(X.flatten(), reconstructed.flatten())
            errors.append((file_name, mse))
            
            print(f'----{file_name}----')
            plt.figure(figsize=(12, 4))
            plt.plot(X.flatten(), label='Original')
            plt.plot(reconstructed.flatten(), label='Reconstructed')
            plt.xlabel("Time (msec)")
            plt.ylabel("INORM (A(p))")
            plt.title("Reconstruction of Training Data")
            plt.legend()
            plt.tight_layout()
            plt.show()
            

        except Exception as e:
            print(f"Received the error {e}")

# Compute average MSE across anomalous data
if errors:
    avg_anomalous_mse = np.mean([e[1] for e in errors])
    print(f"\nAverage MSE across anomalous files: {avg_anomalous_mse:.6f}")
else:
    print("No valid files were processed for error computation.")


# Comparing Anomalous MSE to Training Data MSE
labels = ['Training Data', 'Anomalous Data']
mse_values = [avg_train_mse, avg_anomalous_mse]

plt.figure(figsize=(6, 5))
plt.bar(labels, mse_values, color=['green', 'red'])
plt.ylabel('Average MSE')
plt.title('Comparison of Error on Anomalous Data')
plt.tight_layout()
plt.show()