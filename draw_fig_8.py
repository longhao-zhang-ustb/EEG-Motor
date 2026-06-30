import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
import random
import os
import joblib
import shap
import plotly.graph_objects as go

# 全局字体设置 Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    df = pd.read_csv(r'eeg_features_20260415_80hz_1s.csv')
    # 计算的eeg中psd和正常psd差sfreq倍，需要除掉进行标准化
    sfreq = 250 # 原始数据采样频率
    for col in df.columns:
        if 'psd' in col:
            df[col] /= sfreq
    seeds = [42, 123, 230, 999, 2000]
    seed = seeds[0]
    band_name = 'alpha'
    dim = 30
    set_seed(seed)

    # 剔除标签2
    df = df[df['label'] != 2]
    # 患者维度聚合统计
    df_agg = df.drop(columns=['label']).groupby('patient_id').agg({"mean", "min", "max", "std", "var", "sum", "count"})
    df_agg.columns = [f"{col}_{stat}" for col, stat in df_agg.columns]
    df_agg['label'] = df.groupby('patient_id')['label'].first()
    df_agg = df_agg.reset_index()
    df = df_agg.copy()

    # 合并诊断数据
    df_diagnosis = pd.read_csv(r'dataset\\patient_diagnosis_crs.csv')
    df = pd.merge(df, df_diagnosis, on='patient_id', how='left')

    unique_pids = df['patient_id'].unique()
    feature_cols = [col for col in df.columns if f'{band_name}_psd' in col or f'{band_name}_coherence' in col or 'crs_diff_2' in col]
    feature_name = 'psd_coherence_crs_2'
    X = df[feature_cols]
    y = df['label']

    index = 1
    # 加载预处理模型
    pca = joblib.load(f'experiment7_models\\pca_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')
    scaler = joblib.load(f'experiment7_models\\scaler_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')
    model = joblib.load(f'experiment7_models\\svm_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')

    # 划分数据集
    test_mask = df['patient_id'] == f'P{index:02d}'
    train_mask = ~test_mask
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # 对齐特征 + 标准化 + PCA降维
    X_train = X_train[scaler.feature_names_in_]
    X_test = X_test[scaler.feature_names_in_]
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 原始特征名 & 特征类型划分
    original_feature_names = scaler.feature_names_in_
    feature_types = []
    for feat in original_feature_names:
        if 'psd' in feat or 'power' in feat:
            feature_types.append('PSD')
        elif 'coh' in feat or 'coherence' in feat:
            feature_types.append('Coherence')
        elif 'crs_diff_2' in feat:
            feature_types.append('Motor Function')

    # PCA载荷矩阵
    loadings = pca.components_.T
    n_components = loadings.shape[1]

    # 初始化三大类别贡献
    contrib = {
        'PSD': np.zeros(n_components),
        'Coherence': np.zeros(n_components),
        'Motor Function': np.zeros(n_components)
    }

    # 按类别累加载荷绝对值
    for i, f_type in enumerate(feature_types):
        contrib[f_type] += np.abs(loadings[i])

    # 构建贡献DataFrame
    contrib_df = pd.DataFrame(contrib)
    # 每行(每个PC)归一化总和为1
    contrib_df_norm = contrib_df.div(contrib_df.sum(axis=1), axis=0)

    # 选取前45个主成分
    top_pc_num = 45
    radar_data = contrib_df_norm.iloc[:top_pc_num, :].copy()
    pc_labels = [f'PC{i+1}' for i in range(top_pc_num)]

    # 角度均分 45个维度
    angles = np.linspace(0, 2 * np.pi, top_pc_num, endpoint=False).tolist()
    angles += angles[:1]

    # 三类特征数据并闭合
    psd_vals = radar_data['PSD'].values.tolist()
    coh_vals = radar_data['Coherence'].values.tolist()
    motor_vals = radar_data['Motor Function'].values.tolist()

    psd_vals += psd_vals[:1]
    coh_vals += coh_vals[:1]
    motor_vals += motor_vals[:1]

    # 配色
    color_psd = "#E87A85"
    color_coh = "#7E83A8"
    color_motor = '#5BAA82'

    # 创建画布：左总图 + 右侧放大图
    fig = plt.figure(figsize=(16, 10))

    # 左侧主雷达图
    ax_main = fig.add_subplot(1, 2, 1, polar=True)
    l1, = ax_main.plot(angles, psd_vals, color=color_psd, linewidth=2.5, zorder=3)
    ax_main.fill(angles, psd_vals, color=color_psd, alpha=0.25)
    l2, = ax_main.plot(angles, coh_vals, color=color_coh, linewidth=2.5, zorder=3)
    ax_main.fill(angles, coh_vals, color=color_coh, alpha=0.25)
    l3, = ax_main.plot(angles, motor_vals, color=color_motor, linewidth=2.5, zorder=3)
    ax_main.fill(angles, motor_vals, color=color_motor, alpha=0.25)

    # 原标签行替换
    all_pc_labels = [f'PC{i+1}' for i in range(top_pc_num)]
    # 隔1个显示1个，其余置空
    show_interval = 2
    new_pc_labels = [lab if idx % show_interval == 0 else "" for idx, lab in enumerate(all_pc_labels)]
    ax_main.set_xticks(angles[:-1])
    ax_main.set_xticklabels(new_pc_labels, fontsize=16)
    
    for tick, txt in zip(ax_main.get_xticks(), ax_main.get_xticklabels()):
        if txt.get_text() == 'PC1':
            txt.set_fontweight('bold')
        else:
            txt.set_fontweight("medium")
    
    ax_main.tick_params(axis='x', pad=12)
    ax_main.set_ylim(0, 1.0)
    ax_main.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_main.set_yticklabels(['0.20','0.40','0.60','0.80','1.00'], fontsize=16)
    # ax_main.axvline(x=0, color='gray', linestyle='-.', linewidth=1, zorder=20)
    ax_main.set_rlabel_position(30)
    for tick in ax_main.yaxis.get_majorticklabels():
        tick.set_zorder(10)
    ax_main.grid(True, linestyle='--', alpha=0.6)
    ax_main.spines['polar'].set_linewidth(1.2)
    # ax_main.set_title('Feature Category Contribution across Principal Components', fontsize=14, pad=25, fontweight='bold')

    # 右侧Motor Function放大子图 范围0~0.1
    ax_zoom = fig.add_subplot(1, 2, 2, polar=True)
    ax_zoom.plot(angles, motor_vals, color=color_motor, linewidth=3.2, zorder=3)
    ax_zoom.fill(angles, motor_vals, color=color_motor, alpha=0.45)

    ax_zoom.set_xticks(angles[:-1])
    ax_zoom.set_xticklabels(new_pc_labels, fontsize=16)
    for tick, txt in zip(ax_zoom.get_xticks(), ax_zoom.get_xticklabels()):
        if txt.get_text() == 'PC1':
            txt.set_fontweight('bold')
        else:
            txt.set_fontweight("medium")
    
    ax_zoom.tick_params(axis='x', pad=12)
    ax_zoom.set_ylim(0, 0.06)
    ax_zoom.set_yticks([0.02, 0.04, 0.06])
    ax_zoom.set_yticklabels(['0.02','0.04','0.06'], fontsize=16)
    # ax_zoom.axvline(x=0, color='gray', linestyle='-.', linewidth=1, zorder=20)
    ax_zoom.set_rlabel_position(30)
    for tick in ax_main.yaxis.get_majorticklabels():
        tick.set_zorder(10)
    ax_zoom.grid(True, linestyle='--', alpha=0.6)
    ax_zoom.spines['polar'].set_linewidth(1.2)
    # ax_zoom.set_title('Motor Function Contribution', fontsize=14, pad=25, fontweight='bold')

    # 图例放在两个子图正中间
    fig.legend([l1,l2,l3], ["PSDs","CVs","Motor function"], bbox_to_anchor=(0.5, 0.1),
               loc="lower center", ncol=3, fontsize=16, frameon=False)

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    plt.savefig(
        f'experiment6_res\\pca_contrib_merge_motor_zoom_{index}_{band_name}_{feature_name}.tif',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
