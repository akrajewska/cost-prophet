import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Union

"""
Matrix completion error matrix
"""

def _to_vectors(X_predicted: np.ndarray, X_actual:np.ndarray, test_indices:list) -> tuple:
    y_actual = np.array([X_actual[tuple(idx)] for idx in test_indices])
    y_predicted = np.array([X_predicted[tuple(idx)] for idx in test_indices])
    return y_actual, y_predicted


def rmse(X_output:np.ndarray, X_input:np.ndarray, test_indices: list) -> Union[float, np.ndarray]:
    y_actual, y_predicted = _to_vectors(X_output, X_input, test_indices)
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    return rmse

def nrmse(X_output:np.ndarray, X_input:np.ndarray, test_indices: list) -> Union[float, np.ndarray]:
    y_actual, y_predicted = _to_vectors(X_output, X_input, test_indices)
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    return rmse/(max(y_actual) - min(y_actual))

def mae(X_output:np.ndarray, X_input:np.ndarray, test_indices: list) -> Union[float, np.ndarray]:
    y_actual, y_predicted = _to_vectors(X_output, X_input, test_indices)
    mae = mean_squared_error(y_actual, y_predicted, squared=True)
    return mae

def mape(X_output:np.ndarray, X_input:np.ndarray, test_indices: list) -> Union[float, np.ndarray]:
    y_actual, y_predicted = _to_vectors(X_output, X_input, test_indices)
    return np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100

def test_error(X_output:np.ndarray, X_input:np.ndarray, test_indices: list) -> Union[float, np.ndarray]:
    y_actual, y_predicted = _to_vectors(X_output, X_input, test_indices)
    difference = y_actual - y_predicted
    ssd = np.sum(difference ** 2)
    return ssd/np.sum(y_actual ** 2)
