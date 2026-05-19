"""
支持向量回归 (SVR)
"""

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_svr(X_train, y_train, kernel='rbf', C=1.0, epsilon=0.1):
    model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, epsilon=epsilon))
    model.fit(X_train, y_train)
    return model
