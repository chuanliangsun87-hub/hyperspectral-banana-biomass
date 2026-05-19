# Hyperspectral Banana Biomass Estimation

本仓库包含用于基于无人机高光谱影像反演香蕉树生物量的一套示例 Python 代码。包括：数据预处理、全波段特征提取、连续小波（CWT）分析、纹理/结构特征融合、相关性分析，以及多种回归模型（MLR、RF、SVR、DNN、CNN）的训练、评估与比较。

目录结构（已提交文件的主要部分）：

```
hyperspectral-banana-biomass/
├── README.md
├── requirements.txt
├── config.yaml
├── data/  # 占位：请上传高光谱影像和地面样本数据
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── spectral_analysis.py
│   ├── wavelet_analysis.py
│   ├── feature_extraction.py
│   ├── correlation_analysis.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlr_model.py
│   │   ├── rf_model.py
│   │   ├── svm_model.py
│   │   ├── dnn_model.py
│   │   └── cnn_model.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
└── main.py
```

快速开始：
1. 在 `config.yaml` 中设置数据路径与参数。
2. 把高光谱影像放入 `data/hyperspectral_images/`，地面样本 CSV 放入 `data/ground_truth/ground_truth.csv`。
3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 运行示例：

```bash
python main.py
```

注意：示例代码包含多处占位（TODO），需要根据你的数据格式（例如经纬度->像素坐标转换、波段波长文件）调整后方能直接跑通。有关坐标转换、patch 切割与多时相处理我可以继续帮你完善。
