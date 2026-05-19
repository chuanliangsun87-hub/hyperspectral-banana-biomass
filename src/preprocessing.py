"""
预处理：读取高光谱影像（GeoTIFF/ENVI），波段去噪处理，光谱平滑，坏波段剔除
"""

import os
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter1d
import pandas as pd

def read_hyperspectral(path, driver_hint=None):
    """
    读取高光谱影像（支持 GeoTIFF 波段堆栈 或 ENVI）
    返回：data (H, W, B), meta
    """
    # Rasterio 可读取多波段 GeoTIFF；若是 ENVI 可配合 spectral 包
    with rasterio.open(path) as src:
        data = src.read()  # shape: (B, H, W)
        meta = src.meta
    data = np.transpose(data, (1, 2, 0))  # -> H, W, B
    return data.astype(np.float32), meta

def smooth_spectra(spectra, sigma=1.0):
    """
    对单个光谱（或谱集）进行高斯平滑
    spectra: (..., B)
    """
    return gaussian_filter1d(spectra, sigma=sigma, axis=-1)

def normalize_spectra(spectra):
    """光谱归一化（每个样本除以其最大值以抑制照度差异）"""
    maxv = np.nanmax(spectra, axis=-1, keepdims=True)
    maxv[maxv == 0] = 1.0
    return spectra / maxv

def mask_invalid_bands(data, valid_band_indices):
    """
    data: H, W, B
    valid_band_indices: list/array of indices to keep
    """
    return data[..., valid_band_indices]
