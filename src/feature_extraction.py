"""
将光谱、纹理（GLCM）、结构（株高）等结合，生成训练样本特征表
"""

import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from scipy.ndimage import uniform_filter

from src.spectral_analysis import compute_stats_from_spectrum, pca_reduce
from src.wavelet_analysis import cwt_magnitude_features

def extract_pixel_spectrum(data, row, col):
    """从影像中提取像素光谱"""
    return data[row, col, :]

def compute_glcm_texture(patch, distances=[1], angles=[0], levels=256):
    """
    patch: 2D 灰度图（例如单波段或指数）
    返回：GLCM 常见纹理特征
    """
    # 归一化到 0..levels-1
    img = patch.copy()
    img = img - np.nanmin(img)
    mx = np.nanmax(img)
    if mx > 0:
        img = (img / mx * (levels-1)).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)
    glcm = greycomatrix(img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    feats = {}
    feats['contrast'] = greycoprops(glcm, 'contrast').mean()
    feats['dissimilarity'] = greycoprops(glcm, 'dissimilarity').mean()
    feats['homogeneity'] = greycoprops(glcm, 'homogeneity').mean()
    feats['ASM'] = greycoprops(glcm, 'ASM').mean()
    feats['energy'] = greycoprops(glcm, 'energy').mean()
    return feats

def extract_features_for_samples(hsi, samples_df, window=9, red_idx=30, nir_idx=60):
    """
    hsi: H, W, B
    samples_df: DataFrame，包含 row, col, height, biomass 等列
    返回特征 DataFrame（每行一个样点）
    """
    half = window // 2
    feats_list = []
    spectra_stack = []
    for _, row in samples_df.iterrows():
        r = int(row['row'])
        c = int(row['col'])
        # bounds check
        if r < 0 or c < 0 or r >= hsi.shape[0] or c >= hsi.shape[1]:
            continue
        spectrum = extract_pixel_spectrum(hsi, r, c)
        spectrum_sm = spectrum  # 可进行平滑
        spec_stats = compute_stats_from_spectrum(spectrum_sm)
        # CWT 特征
        cwt_feats = cwt_magnitude_features(spectrum_sm)
        # 局部纹理：用红波段或某指数
        r0 = max(0, r-half)
        r1 = min(hsi.shape[0], r+half+1)
        c0 = max(0, c-half)
        c1 = min(hsi.shape[1], c+half+1)
        patch_red = hsi[r0:r1, c0:c1, red_idx]
        tex_feats = compute_glcm_texture(patch_red)
        # PCA 波段降维特征 (简单示例)
        # 这里只演示，将光谱直接加入
        combined = {}
        combined.update(spec_stats)
        combined.update({f'cwt_{k}': v if np.isscalar(v) else None for k,v in cwt_feats.items()})
        combined.update(tex_feats)
        combined['biomass'] = row.get('biomass', np.nan)
        combined['height'] = row.get('height', np.nan)
        combined['row'] = r
        combined['col'] = c
        # 也保留原始光谱向量（可用于深度学习）
        combined['spectrum'] = spectrum_sm
        feats_list.append(combined)
    # 转换为 DataFrame（部分列为数组，稍后处理）
    df = pd.DataFrame(feats_list)
    return df
