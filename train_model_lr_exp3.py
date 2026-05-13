import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
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
     # 结果保存到txt文件
    try:
        # 删除之前的结果文件
        os.remove(f'experiment3_res\\lr_{band_name}_{feature_name}__results.txt')
    except:
        pass
    # 遍历不同维度，记录每个维度的结果
    # 核心：逐个患者留一！！！
    for dim in range(1, 78):
        y_pred = []
        y_true = []
        y_pred_proba = []
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
            # 使用PCA：30维度效果最好
            pca = PCA(n_components=dim)
            X_train = pca.fit_transform(X_train)
            X_test  = pca.transform(X_test)
            # 使用LR模型
            model = LogisticRegression(random_state=seed)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            # 预测概率
            y_pred_proba.extend(model.predict_proba(X_test)[:, 1])
            y_pred.extend(pred)
            y_true.extend(y_test)
            print(f'预测完患者: {test_pid}，维度：{dim}，数据条数：{len(X_test)}')
        with open(f'experiment3_res\\lr_{band_name}_{feature_name}__results.txt', 'a+') as f:
            # 预测标签，预测概率，真实标签写入文件
            f.write(f"F1-score: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
            f.write(f"MatthewCC: {matthews_corrcoef(y_true, y_pred):.4f}\n")
