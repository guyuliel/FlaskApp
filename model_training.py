import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from car_data_prep import prepare_data

# Load data (replace with actual data loading code)
processed_data = pd.read_csv('C:\\Users\\guyul\\ProjectP3\\training_data.csv')


# Split data into training and test sets
X = processed_data.drop('Price', axis=1)
y = processed_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train the model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)


# Predict and evaluate performance
y_pred = elastic_net.predict(X_test)


# Save the model using PKL
model_filename = 'trained_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(elastic_net, file)


# Save the Scaler for future use
scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)