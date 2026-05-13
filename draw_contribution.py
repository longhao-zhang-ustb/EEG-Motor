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
    original_feature_names = scaler.feature_names_in_
    loadings = pca.components_.T  # shape: (n_features, n_components)
    """
    original_feature_names: 每个特征在不同主成分上的贡献
    [[ 0.02868996  0.00137267  0.13946732 ...  0.01463515  0.02513028
    -0.01271464]
    [ 0.01948346  0.0277987   0.08282035 ...  0.04593586  0.02357106
    -0.04518808]
    [ 0.02189478  0.02313153  0.09464288 ...  0.01618588  0.02603154
    -0.03626961]
    ...
    [ 0.01997621  0.027543   -0.00097007 ...  0.08014852 -0.02642977
    -0.07392662]
    [ 0.04560458  0.11971994  0.01238048 ...  0.01135455 -0.00588552
    -0.00344265]
    [ 0.00705215  0.0425527   0.00961893 ... -0.1489618  -0.07955893
    -0.18735235]]
    feature_types:
   ['PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 
    'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 
    'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 
    'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 
    'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'PSD', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 'Coherence', 
    'Motor']
    """
    feature_types = []
    for feat in original_feature_names:
        if 'psd' in feat.lower() or 'power' in feat.lower():
            feature_types.append('PSD')
        elif 'coh' in feat.lower() or 'coherence' in feat.lower():
            feature_types.append('CV')
        elif 'crs_diff_2' in feat.lower():
            feature_types.append('Motor function')

    # 转为DataFrame方便处理
    df_types = pd.DataFrame({
        'feature': original_feature_names,
        'type': feature_types
    })
    # n_components: 主成分数量
    n_components = loadings.shape[1]
    # contrib保存每个特征在不同主成分上的贡献
    contrib = {
        'PSD': np.zeros(n_components),
        'CV': np.zeros(n_components),
        'Motor function': np.zeros(n_components)
    }

    # 遍历每个原始特征，把它的载荷绝对值加到对应类别
    for i, feat_type in enumerate(feature_types):
        # 第i个特征在每个主成分上的贡献
        loading_abs = np.abs(loadings[i])  # 该特征在所有PC上的载荷绝对值
        contrib[feat_type] += loading_abs

    # 转为DataFrame
    contrib_df = pd.DataFrame(contrib)

    # 归一化：每个PC的总和=1（百分比堆叠）
    contrib_sum = contrib_df.sum(axis=1)
    contrib_df_norm = contrib_df.div(contrib_sum, axis=0)
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(14, 3))
    colors = {
        'PSD': '#2878B5',
        'CV': '#E24A33',
        'Motor function': '#32B868'
    }
    bottom = np.zeros(n_components)

    # 画堆叠柱状图
    for col in ['PSD', 'CV', 'Motor function']:
        values = contrib_df_norm[col].values
        ax.bar(
            range(1, n_components+1),
            values,
            bottom=bottom,
            color=colors[col],
            edgecolor='black',
            linewidth=0.5,
            label=col
        )
        bottom += values

    # 图表美化
    # xlabel下移10个单位
    ax.set_xlabel('Principal components', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_ylabel('Proportion of contribution', fontsize=14, y=0.48, fontweight='bold')
    # 坐标刻度字体大小
    ax.tick_params(axis='both', labelsize=14)
    
    ax.set_xticks(range(1, n_components+1, 2))  # 每2个显示一个
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.07), 
        ncol=3, 
        frameon=False, 
        fontsize=12
    )
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    # plt.show()
    # exit()
    #保存为300dpi的tif图片
    plt.savefig(f'experiment6_res\\pca_contribution_patient_{index}_{band_name}_{feature_name}.tif', dpi=300, bbox_inches='tight')
