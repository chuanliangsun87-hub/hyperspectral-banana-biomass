"""
随机森林回归
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_rf(X_train, y_train, random_state=42):
    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X, y):
    pred = model.predict(X)
    from sklearn.metrics import r2_score, mean_squared_error
    return {'r2': r2_score(y, pred), 'rmse': (mean_squared_error(y, pred) ** 0.5)}
