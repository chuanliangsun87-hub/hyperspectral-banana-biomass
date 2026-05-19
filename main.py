"""
主流程示例脚本（示范如何串联各模块）
请先修改 config.yaml 指向正确数据路径与参数
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing import read_hyperspectral, smooth_spectra
from src.feature_extraction import extract_features_for_samples
from src.correlation_analysis import feature_target_correlations
from src.models import mlr_model, rf_model, svm_model, dnn_model, cnn_model
from src.utils.metrics import evaluate_all
from src.utils.visualization import plot_scatter_true_pred

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def main(config_path='config.yaml'):
    cfg = load_config(config_path)
    # 1) 读取 ground truth
    gt = pd.read_csv(cfg['data']['ground_truth_csv'])
    # 2) 读取对应时相的高光谱影像 —— 这里只示例单幅
    hsi_path = os.path.join(cfg['data']['hyperspectral_dir'], os.listdir(cfg['data']['hyperspectral_dir'])[0])
    hsi, meta = read_hyperspectral(hsi_path)
    # 3) 特征提取（示例）
    feats_df = extract_features_for_samples(hsi, gt, window=cfg['feature']['window_size'])
    # 需要把 'spectrum' 列展开/或删除，构造 X（tabular）
    # 这里示例：使用 mean, std, texture, height 作为特征
    candidate_cols = ['mean','std','contrast','dissimilarity','homogeneity','ASM','energy','height','biomass']
    # 处理并去除包含 None/array 的列
    X_df = feats_df.dropna(subset=['mean','std'])  # 简单筛选
    # 构造 X, y
    feature_cols = ['mean','std','contrast','homogeneity','energy','height']
    X = X_df[feature_cols].values
    y = X_df['biomass'].values
    # train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg['models']['test_size'], random_state=cfg['models']['random_state'])
    # MLR
    mlr = mlr_model.train_mlr(X_train, y_train)
    res_mlr = mlr_model.evaluate(mlr, X_test, y_test)
    print('MLR:', res_mlr)
    # RF
    rf = rf_model.train_rf(X_train, y_train, random_state=cfg['models']['random_state'])
    res_rf = rf_model.evaluate(rf, X_test, y_test)
    print('RF:', res_rf)
    # SVR
    svr = svm_model.train_svr(X_train, y_train)
    pred_svr = svr.predict(X_test)
    from src.utils.metrics import evaluate_all
    print('SVR:', evaluate_all(y_test, pred_svr))
    # DNN
    import tensorflow as tf
    dnn = dnn_model.build_dnn(X_train.shape[1], lr=cfg['training']['learning_rate'])
    dnn, hist = dnn_model.train_dnn(dnn, X_train, y_train, X_test, y_test, epochs=cfg['training']['epochs'], batch_size=cfg['training']['batch_size'])
    pred_dnn = dnn.predict(X_test).ravel()
    print('DNN:', evaluate_all(y_test, pred_dnn))
    # 可视化示例
    plot_scatter_true_pred(y_test, pred_dnn, title='DNN Observed vs Predicted')

if __name__ == '__main__':
    main()
