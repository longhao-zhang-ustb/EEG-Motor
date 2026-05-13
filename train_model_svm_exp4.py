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
import os
plt.rcParams['font.sans-serif'] = ['Times New Roman']

if __name__ == '__main__':
    df = pd.read_csv(r'eeg_features_20260415_80hz_1s.csv')
    # 需要调整的变量
    ###################################
    seed = 42
    band_name = 'alpha'
    dim = 30
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
    df_diagnosis = pd.read_csv(r'dataset\patient_diagnosis_crs.csv')
    df = pd.merge(df, df_diagnosis, on='patient_id', how='left')
    # 选择其中1个患者的数据做测试集，其余患者做训练集
    unique_pids = df['patient_id'].unique()
    feature_cols = [col for col in df.columns if f'{band_name}_psd' in col or f'{band_name}_coherence' in col or 'crs_diff_2' in col]
    feature_name = 'psd_coherence_crs_2'
    X = df[feature_cols]
    y = df['label']
    scaler = StandardScaler()
    y_pred = []
    y_true = []
    y_pred_proba = []
    # 遍历不同维度，记录每个维度的结果
    # 核心：逐个患者留一！！！
    for test_pid in unique_pids:
        # 测试集 = 这个患者的所有数据
        test_mask = df['patient_id'] == test_pid
        # 训练集 = 除了这个患者以外所有人
        train_mask = ~test_mask
        X_train = X[train_mask]
        X_test  = X[test_mask]
        y_train = y[train_mask]
        y_test  = y[test_mask]
        # 标准化
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
        # 使用PCA降维--SVM：30维度效果最好
        pca = PCA(n_components=dim)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        # 使用svm模型
        model = SVC(kernel='linear', probability=True, random_state=seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # 预测概率
        y_pred_proba.extend(model.predict_proba(X_test)[:, 1])
        y_pred.extend(pred)
        y_true.extend(y_test)
        print(f'预测完患者: {test_pid}，数据条数：{len(X_test)}')
    # 绘制校准曲线
    # 计算校准曲线
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_proba, n_bins=10, strategy='quantile'
    )
    brier = brier_score_loss(y_true, y_pred_proba)
    # 加载lr模型的prob_true, prob_pred, brier
    prob_true_lr = np.load(f'experiment4_res\\prob_true_{band_name}.npy')
    prob_pred_lr = np.load(f'experiment4_res\\prob_pred_{band_name}.npy')
    brier_lr = np.load(f'experiment4_res\\brier_{band_name}.npy')
    # 绘制
    plt.figure(figsize=(16, 6))
    # 完美校准对角线
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibrated', linewidth=1.5)
    # SVM校准曲线
    plt.plot(
        prob_pred, prob_true, 
        "o-", color="#A83CE7", linewidth=2.5, markersize=6,
        label=f"SVM (Brier={brier*100:.2f}%)"
    )
    # LR校准曲线
    plt.plot(
        prob_pred_lr, prob_true_lr, 
        "o-", color="#33BB26", linewidth=2.5, markersize=6,
        label=f"LR (Brier={brier_lr*100:.2f}%)"
    )
    plt.xlabel("Forecast probability", fontsize=18, fontweight='bold')
    plt.ylabel("Observed relative frequency", fontsize=18, fontweight='bold')
    # 设置坐标刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # 设置图例字体大小
    plt.legend(loc="upper left", fontsize=18)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    # plt.show()
    # 保存图片
    plt.savefig(f'experiment4_res\\svm_calibration_curve_{band_name}.tif', dpi=300, bbox_inches='tight')
