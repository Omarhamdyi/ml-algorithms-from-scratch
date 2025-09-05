import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

def load_data_scaled():
    """
        Loads the California housing dataset,
        scales the features with MinMaxScaler,
        and returns (X, y).
    """
    housing = fetch_california_housing()
    X = housing.data        # feature matrix size (20640, 8)
    t = housing.target      # target vector size  (20640)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled,t
