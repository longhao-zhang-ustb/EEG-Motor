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
import mne
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
    original_feature_names = scaler.feature_names_in_
    loadings = pca.components_.T  # shape: (n_features, n_components)
    
    # 1. 电极列表
    ch_names = ['Pz', 'P3', 'P4', 'Cz', 'Fz', 'FCz', 'Oz', 'O1', 'O2']
    n_chan = len(ch_names)
    n_pc = 45

    # 生成所有唯一电极对 i<j
    pairs = []
    for i in range(n_chan):
        for j in range(i+1, n_chan):
            pairs.append((i, j))
    n_edge = len(pairs)

    # 2. 贡献矩阵 替换成你的 (36,45)
    # 遍历电极列表，计算每个电极对的贡献值
    contrib_mat = np.zeros((n_edge, n_pc))
    calc_num = 0
    for i, p in enumerate(ch_names):
        for j, q in enumerate(ch_names):
            if i != j and i < j:
                mask = [p in feat and q in feat for feat in original_feature_names]
                if np.any(mask):
                    # 根据p_q_cols计算贡献值
                    loading_vec = np.abs(loadings[mask])
                    contrib_mat[calc_num] = loading_vec.sum(axis=0)
                    calc_num += 1
                    print(p, q)
                    print(contrib_mat[calc_num-1].tolist())
    
    # ---------------------- 获取电极坐标 ----------------------
    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=["eeg"]*n_chan)
    info.set_montage(montage)

    pos = np.array([montage.get_positions()['ch_pos'][ch] for ch in ch_names])
    pos_2d = pos[:, :2]

    # 固定坐标范围，确保电极完整显示
    x_lim = (-0.13, 0.13)
    y_lim = (-0.13, 0.08)

    # 5种渐变颜色（第1粗最亮最强 → 第5细最弱）
    colors = ['#E53E3E', '#DD6B20', '#D69E2E', '#38A169', '#3182CE']

    # ---------------------- 绘图：5行9列 ----------------------
    fig, axes = plt.subplots(nrows=5, ncols=9, figsize=(26, 10))
    axes = axes.flatten()

    for pc_idx in range(n_pc):
        ax = axes[pc_idx]
        weights = contrib_mat[:, pc_idx]
        # 取贡献度最高的前5条连接
        top5_idx = np.argsort(weights)[::-1][:5]
        top5_weights = weights[top5_idx]
        # 画电极
        ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c='gray', s=72, zorder=5)

        # 画前5条连接：颜色不同、粗细不同
        for i_rank, idx in enumerate(top5_idx):
            i, j = pairs[idx]
            lw = 4.0 - i_rank * 0.6  # 排名越高，线越粗
            ax.plot(
                [pos_2d[i, 0], pos_2d[j, 0]],
                [pos_2d[i, 1], pos_2d[j, 1]],
                color=colors[i_rank],
                lw=lw,
                alpha=0.85
            )

        ax.set_title(f'PC{pc_idx+1}', fontsize=20, fontweight='bold')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.axis('off')

    # 隐藏多余子图
    for ax in axes[n_pc:]:
        ax.axis('off')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.15, hspace=0.25)
    plt.savefig("experiment6_res\\PC1-45_top5_colors.tif", dpi=300, bbox_inches='tight')
    plt.show()
