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

# 全局字体统一设置
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    df = pd.read_csv(r'eeg_features_20260415_80hz_1s.csv')
    sfreq = 250 # 原始数据采样频率
    for col in df.columns:
        if 'psd' in col:
            df[col] /= sfreq
    ###################################
    seeds = [42, 123, 230, 999, 2000]
    seed = seeds[0]
    band_name = 'alpha'
    dim = 45
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
    original_feature_names = scaler.feature_names_in_
    loadings = pca.components_.T  # shape: (n_features, n_components)
    
    channels = ['Pz', 'Cz', 'O1', 'P3', 'Oz', 'O2', 'P4', 'Fz', 'FCz']
    # 筛选每个通道的psd特征
    channel_loadings = []
    channel_labels = []

    for ch in channels:
        # 匹配特征：如 Pz_alpha_psd
        mask = [ch in feat and 'psd' in feat for feat in original_feature_names]
        if np.any(mask):
            # 该通道在所有PC上的载荷（取绝对值表示贡献）
            ch_loading = np.abs(loadings[mask])
            ch_loading = ch_loading.sum(axis=0)
            channel_loadings.append(ch_loading)
            channel_labels.append(ch)

    # 转成矩阵 (9个通道, 45个PC)
    channel_loadings = np.array(channel_loadings)
    n_channels = len(channel_labels)
    n_components = channel_loadings.shape[1]

    # 归一化：每个PC内总和=100%
    pc_sums = channel_loadings.sum(axis=0)
    contrib_matrix = channel_loadings / pc_sums  # 占比

    # ===================== PCA通道贡献热力图=====================
    top_pc_num = 45
    contrib_matrix_slim = contrib_matrix[:, :top_pc_num]
    pc_ticks = [f'PC{i+1}' for i in range(top_pc_num)]
    
    import matplotlib.colors as mcolors
    colors = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]  # 红绿蓝三原色
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # 绘图配置
    fig, ax = plt.subplots(figsize=(16, 4))
    mask = np.abs(contrib_matrix_slim) <= 0.15
    heatmap = sns.heatmap(
        data=contrib_matrix_slim,
        cmap=cmap,               
        xticklabels=pc_ticks,
        yticklabels=channel_labels,
        cbar=True,
        annot=False,
        linewidth=0.4,            
        linecolor='black',
        fmt='.2f',
        mask=mask
    )

    # 坐标轴标签
    ax.set_xlabel('Principal components', fontsize=16, fontweight='bold', labelpad=12)
    ax.set_ylabel('EEG electrode channels', fontsize=16, fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', labelsize=16, length=2)
    
    # 显示外部框线
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines[:].set_linewidth(1)
    
    # ax.tick_params(axis='', length=0)  
    ax.set_xticks(np.arange(len(pc_ticks)) + 0.5)
    # 初始化空标签
    xtick_labels = [''] * len(pc_ticks)
    # 隔一个填充文字
    for idx in range(0, len(pc_ticks), 2):
        xtick_labels[idx] = pc_ticks[idx]
    ax.set_xticklabels(xtick_labels, rotation=0, fontsize=13)
    
    # 设置colorbar刻度字体大小
    from matplotlib.ticker import FormatStrFormatter
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=16)        # colorbar刻度字体放大
    cbar.set_label('Proportion', fontsize=16)
    
    # 布局与保存
    plt.tight_layout()
    plt.savefig(
        f'experiment6_res\\pca_channel_heatmap_{index}_{band_name}_{feature_name}.tif',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
