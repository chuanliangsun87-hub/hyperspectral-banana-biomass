"""
计算光谱/特征与生物量的相关性（Pearson、Spearman）
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def feature_target_correlations(df, feature_cols, target_col='biomass'):
    """
    df: DataFrame
    feature_cols: list of column names（若列为数组则需单独处理）
    返回 DataFrame: feature, pearson_r, pearson_p, spearman_r, spearman_p
    """
    records = []
    for col in feature_cols:
        series = df[col].dropna()
        if series.shape[0] < 3:
            continue
        x = series.values
        y = df.loc[series.index, target_col].values
        try:
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
        except Exception as e:
            pr = pp = sr = sp = np.nan
        records.append({'feature': col, 'pearson_r': pr, 'pearson_p': pp, 'spearman_r': sr, 'spearman_p': sp})
    return pd.DataFrame.from_records(records).sort_values('pearson_r', ascending=False)
