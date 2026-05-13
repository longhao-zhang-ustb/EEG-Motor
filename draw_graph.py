import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
# 计算AUC
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# 绘图字体设置为times new roman
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
import random
import os
import joblib
import shap
import plotly.graph_objects as go
plt.rcParams['font.sans-serif'] = ['Times New Roman']

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    df = pd.read_csv(r'eeg_features_20260415_80hz_1s.csv')
    ###################################
    seeds = [42, 123, 230, 999, 2000]
    seed = seeds[0]
    band_name = 'alpha'
    dim = 30
    set_seed(seed)
    ###################################
    # 去掉label为2的记录
    df = df[df['label'] != 2]
    # 对同一患者的多条记录分别保存均值，最小值，最大值整合成一条记录, 不包括label
    df_new = df.drop(columns=['label']).groupby('patient_id').agg({"mean", "min", "max", "std", "var", "sum", "count"})
    df_new.columns = [f"{col}_{stat}" for col, stat in df_new.columns]
    df_new = df_new.copy()
    # 获取每个患者的第一个patient_id和label
    df_new['label'] = df.groupby('patient_id')['label'].first()
    df_new['patient_id'] = df.groupby('patient_id')['patient_id'].first()
    # 将df_new赋给df
    df = df_new.copy()
    df = df.reset_index(drop=True)
    # 读取dataset\patient_diagnosis_crs.csv
    df_diagnosis = pd.read_csv(r'dataset\\patient_diagnosis_crs.csv')
    df = pd.merge(df, df_diagnosis, on='patient_id', how='left')
    # 选择其中1个患者的数据做测试集，其余患者做训练集
    unique_pids = df['patient_id'].unique()
    feature_cols = [col for col in df.columns if f'{band_name}_psd' in col or f'{band_name}_coherence' in col or 'crs_diff_2' in col]
    feature_name = 'psd_coherence_crs_2'
    X = df[feature_cols]
    y = df['label']
    # ===================== 2. 标准化 + PCA =====================
    index = 1
    pca = joblib.load(f'experiment7_models\\pca_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')
    # 加载标准化器
    scaler = joblib.load(f'experiment7_models\\scaler_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')
    # 加载模型
    model = joblib.load(f'experiment7_models\\svm_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')
    # 获取训练数据和测试数据P06
    test_mask = df['patient_id'] == 'P' + ('0' if index <= 9 else '') + str(index)
    # 训练集 = 除了这个患者以外所有人
    train_mask = ~test_mask
    X_train = X[train_mask]
    X_test  = X[test_mask]
    y_train = y[train_mask]
    y_test  = y[test_mask]
    X_train = X_train[scaler.feature_names_in_]
    X_test  = X_test[scaler.feature_names_in_]
    # 标准化
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    # 使用PCA降维--SVM：30维度效果最好
    X_train = pca.transform(X_train)
    X_test  = pca.transform(X_test)
    loadings = pca.components_.T  # shape: (n_features, n_components)
    # ===================== 3. 构造桑基图数据 =====================
    labels = list(feature_cols) + [f"PC{i+1}" for i in range(2)]
    # labels = [""] * (len(feature_cols)) + [f"PC{i+1}" for i in range(dim)]  # 原始特征标签隐藏，图更简洁

    sources = []
    targets = []
    values = []

    n_feats = len(feature_cols)
    for feat_idx in range(n_feats):
        for pc_idx in range(2):
            val = abs(loadings[feat_idx, pc_idx])
            if val > 0.1:  # 过滤掉太小的贡献，图更干净
                sources.append(feat_idx)
                targets.append(n_feats + pc_idx)
                values.append(val)

    # ===================== 4. 画桑基图 =====================
    fig = go.Figure(go.Sankey(
        node=dict(
            label=labels,
            pad=20,
            thickness=20
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(200, 200, 200, 0.75)"
        ),
        arrangement="fixed"
        
    ))
    # PCA 特征流向桑基图：原始特征 → 主成分
    fig.update_layout(
        title="Raw Features → Principal Components", 
        font=dict(
            family="Times New Roman",  # 全局所有文字新罗马
            size=12,
            color="black",
            weight="bold"
        ),
        width=600
    )
    fig.show()
    # 保存为svg
    # fig.write_image(f'experiment6_res\\pca_sankey_patient_{index-1}_{seed}_{band_name}_{feature_name}.svg', scale=3)
