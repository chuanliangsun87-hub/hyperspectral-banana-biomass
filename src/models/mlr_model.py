"""
多元线性回归 (MLR)
"""

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def train_mlr(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate(model, X, y):
    y_pred = model.predict(X)
    return {'r2': r2_score(y, y_pred), 'rmse': np.sqrt(mean_squared_error(y, y_pred))}
