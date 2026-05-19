"""
连续小波变换（CWT）特征提取 —— 使用 PyWavelets 或 scipy.signal
这里使用 pywt.cwt（连续小波变换）
"""

import numpy as np
import pywt

def cwt_magnitude_features(spectrum, wavelet='morl', scales=None):
    """
    spectrum: 1D array (B,)
    返回：不同尺度下的幅值统计（mean, max, std）作为特征
    """
    if scales is None:
        # scales 的选取可根据波段数调整
        scales = np.arange(1, min(64, len(spectrum)//2))
    coeffs, freqs = pywt.cwt(spectrum, scales, wavelet)
    # coeffs shape: (len(scales), len(spectrum))
    mag = np.abs(coeffs)
    feats = {}
    feats['cwt_mean'] = np.nanmean(mag)
    feats['cwt_std'] = np.nanstd(mag)
    feats['cwt_max'] = np.nanmax(mag)
    # 也可按尺度提取均值向量
    feats['cwt_scale_mean_vector'] = np.nanmean(mag, axis=1)
    return feats
