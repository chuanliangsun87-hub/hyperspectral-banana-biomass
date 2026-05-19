"""
评估指标
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mbe(y_true, y_pred):
    return np.mean(y_pred - y_true)

def evaluate_all(y_true, y_pred):
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mbe': mbe(y_true, y_pred)
    }
