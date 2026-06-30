import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# 导入随机森林模型
from sklearn.ensemble import RandomForestClassifier
# 导入XGBoost
from xgboost import XGBClassifier as xgb
# 导入GBDT模型
from sklearn.ensemble import GradientBoostingClassifier as gbdt
# 导入LightGBM模型
from lightgbm import LGBMClassifier as lgbm
# 导入网格搜索
from sklearn.model_selection import GridSearchCV
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
plt.rcParams['font.sans-serif'] = ['Times New Roman']

if __name__ == '__main__':
    df = pd.read_csv(r'eeg_features_20260415_80hz_1s.csv')
    sfreq = 250 # 原始数据采样频率
    for col in df.columns:
        if 'psd' in col:
            df[col] /= sfreq
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
    # print(df.columns)
    # feature_cols = [col for col in df.columns if 'crs_diff_2' in col]
    X = df[feature_cols]
    y = df['label']
    scaler = StandardScaler()
    y_pred = []
    y_true = []
    y_pred_proba = []
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
        # 使用PCA降维--LR：45维度效果最好
        pca = PCA(n_components=45)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        # 使用lr模型
        # model = LogisticRegression(random_state=seed)
        # model = RandomForestClassifier(
        #     n_estimators=30,
        #     random_state=42
        # )
        # model = xgb(
        #     random_state=seed,
        #     n_estimators=30,       # 树数量
        #     max_depth=3,             # 树深度
        #     learning_rate=0.1,       # 学习率
        #     subsample=0.8,           # 行采样
        #     colsample_bytree=0.8,    # 列采样
        #     gamma=0,                 # 剪枝阈值
        #     reg_alpha=0,             # L1正则
        #     reg_lambda=1,            # L2正则
        #     objective="binary:logistic", # 目标函数
        #     eval_metric="auc",       # 评估指标
        #     use_label_encoder=False,
        #     verbosity=0
        # )
        model = gbdt(
            random_state=seed,
            n_estimators=30,      # 迭代树数
            max_depth=2,           # 单树深度
            learning_rate=0.1,     # 学习率
            subsample=0.8,         # 子采样
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # 预测概率
        y_pred_proba.extend(model.predict_proba(X_test)[:, 1])
        y_pred.extend(pred)
        y_true.extend(y_test)
        print(f'预测完患者: {test_pid}，数据条数：{len(X_test)}')
    print('预测完成')
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print("=" * 50)
    print(" LR模型留一法交叉验证结果 ")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MatthewCC: {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"Cohen Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
