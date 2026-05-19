"""
绘制散点图、残差、时间序列等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_scatter_true_pred(y_true, y_pred, title=None):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y_true, y=y_pred)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    plt.plot([mn,mx],[mn,mx], 'r--')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    if title:
        plt.title(title)
    plt.show()

def plot_residuals(y_true, y_pred):
    res = np.array(y_pred) - np.array(y_true)
    plt.figure(figsize=(6,4))
    sns.histplot(res, kde=True)
    plt.title('Residuals')
    plt.show()
