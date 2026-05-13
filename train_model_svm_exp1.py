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
        # 使用PCA降维--SVM：30维度效果最好
        pca = PCA(n_components=30)
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
    print('预测完成')
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print("=" * 50)
    print(" SVM模型留一法交叉验证结果 ")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MatthewCC: {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"Cohen Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    # 计算ROC曲线相关参数
    fpr_svm, tpr_svm, thresholds_roc_svm = roc_curve(y_true, y_pred_proba)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    # 计算PR曲线相关参数
    precision_svm, recall_svm, thresholds_pr_svm = precision_recall_curve(y_true, y_pred_proba)
    ap_svm = average_precision_score(y_true, y_pred_proba)
    # 结果保存到txt文件
    with open(f'experiment1_res\\svm_{band_name}_results.txt', 'w') as f:
        # 预测标签，预测概率，真实标签写入文件
        f.write(f"Predicted labels: {y_pred}\n")
        f.write(f"Predicted probabilities: {y_pred_proba}\n")
        f.write(f"True labels: {y_true}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {pre:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"MatthewCC: {matthews_corrcoef(y_true, y_pred):.4f}\n")
        f.write(f"Cohen Kappa: {cohen_kappa_score(y_true, y_pred):.4f}\n")
        f.write(f"AUC: {roc_auc_score(y_true, y_pred_proba):.4f}\n")
        f.write(f'ROC fpr: {fpr_svm.tolist()}\n')
        f.write(f'ROC tpr: {tpr_svm.tolist()}\n')
        f.write(f'ROC thresholds: {thresholds_roc_svm.tolist()}\n')
        f.write(f"ROC AUC: {roc_auc_svm}\n")
        f.write(f'PR precision: {precision_svm.tolist()}\n')
        f.write(f'PR recall: {recall_svm.tolist()}\n')
        f.write(f'PR thresholds: {thresholds_pr_svm.tolist()}\n')
        f.write(f"PR AP: {ap_svm}\n")
    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.set_style("whitegrid", {'font.family': 'Times New Roman'})
    ax = sns.heatmap(
        cm,
        annot=True,        # 显示数字
        fmt="d",           # 整数格式
        cmap="Reds",      
        cbar=False,        # 不显示颜色条
        annot_kws={"size": 18, "fontweight": "bold"},
        linewidths=0.1    # 边框线
    )
    # 设置横纵坐标标签
    ax.set_xticklabels(['No recovery', 'Recovery'])
    ax.set_yticklabels(['No recovery', 'Recovery'])
    plt.xlabel("Model assessment status", fontsize=18, fontweight='bold')
    plt.ylabel("Observed status", fontsize=18, fontweight='bold')
    # 坐标刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    # plt.show()
    # 保存为dpi300的tif
    plt.savefig(f'experiment1_res\\svm_{band_name}_confusion_matrix.tif', dpi=300)
    exit()
    # 绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', linewidth=2,  label=f'ROC curve (AUC = {roc_auc * 100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
    # 曲线下填充
    plt.fill_between(fpr, tpr, color='blue', alpha=0.2)
    # 曲线上方填充
    plt.fill_between(fpr, tpr, color='gray', alpha=0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('1 - Specificity', fontsize=18, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=18, fontweight='bold')
    # 坐标刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, loc="upper left")
    plt.tight_layout()
    # 保存为dpi300的tif
    plt.savefig(f'experiment1_res\\svm_{band_name}_ROC_curve.tif', dpi=300)
    # 绘制PR曲线
    plt.figure(figsize=(10, 6))
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # 计算AP值
    ap = average_precision_score(y_true, y_pred_proba)
    # 绘制PR曲线
    plt.plot(recall, precision, color='green', linewidth=2, label=f'PR curve (AP = {ap * 100:.2f}%)')
    # 填充曲线下面积
    plt.fill_between(recall, precision, color='green', alpha=0.2)
    plt.xlabel('Recall', fontsize=18, fontweight='bold')
    plt.ylabel('Precision', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, loc="upper right")
    plt.tight_layout()
    # 保存为dpi300的tif
    plt.savefig(f'experiment1_res\\svm_{band_name}_PR_curve.tif', dpi=300)
    
    
