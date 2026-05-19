"""
全波段特征分析：计算常见植被指数（NDVI/NDRE/CI等）、全波段统计量、主成分等
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def compute_index(data, band_n, band_m):
    """简单波段比率 (band_n - band_m)/(band_n + band_m)"""
    n = data[..., band_n]
    m = data[..., band_m]
    denom = (n + m)
    denom[denom == 0] = np.nan
    return (n - m) / denom

def compute_ndvi(data, red_band, nir_band):
    return compute_index(data, nir_band, red_band)

def compute_stats_from_spectrum(spectrum):
    """
    spectrum: (B,)
    返回均值、方差、峰值波段、斜率等统计特征
    """
    mean = np.nanmean(spectrum)
    std = np.nanstd(spectrum)
    maxi = np.nanmax(spectrum)
    mini = np.nanmin(spectrum)
    band_max = np.nanargmax(spectrum)
    band_min = np.nanargmin(spectrum)
    return dict(mean=mean, std=std, max=maxi, min=mini, band_max=band_max, band_min=band_min)

def pca_reduce(spectra_array, n_components=5):
    """
    spectra_array: N x B
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(spectra_array)
    return components, pca
