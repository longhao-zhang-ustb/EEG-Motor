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
plt.rcParams['font.sans-serif'] = ['Times New Roman']

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    df = pd.read_csv(r'eeg_features_20260415_80hz_1s.csv')
    # 需要调整的变量
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
    # #########################################################
    # scaler = StandardScaler()
    # y_pred = []
    # y_true = []
    # y_pred_proba = []
    # # 核心：逐个患者留一！！！
    # for index, test_pid in enumerate(unique_pids):
    #     # 测试集 = 这个患者的所有数据
    #     test_mask = df['patient_id'] == test_pid
    #     # 训练集 = 除了这个患者以外所有人
    #     train_mask = ~test_mask
    #     X_train = X[train_mask]
    #     X_test  = X[test_mask]
    #     y_train = y[train_mask]
    #     y_test  = y[test_mask]
    #     # 标准化
    #     X_train = scaler.fit_transform(X_train)
    #     # 保存标准化器
    #     joblib.dump(scaler, f'experiment7_models\\scaler_patient_{index}_{seed}_{band_name}_{feature_name}.pkl')
    #     X_test  = scaler.transform(X_test)
    #     # 使用PCA降维--lr：30维度效果最好
    #     pca = PCA(n_components=dim)
    #     X_train = pca.fit_transform(X_train)
    #     # 保存PCA模型
    #     joblib.dump(pca, f'experiment7_models\\pca_patient_{index}_{seed}_{band_name}_{feature_name}.pkl')
    #     X_test  = pca.transform(X_test)
    #     # 使用lr模型
    #     model = LogisticRegression(random_state=seed)
    #     model.fit(X_train, y_train)
    #     pred = model.predict(X_test)
    #     # 预测概率
    #     y_pred_proba.extend(model.predict_proba(X_test)[:, 1])
    #     y_pred.extend(pred)
    #     y_true.extend(y_test)
    #     # 如果y_test和pred不一致的话，打印test_id
    #     if np.any(y_test != pred):
    #         print(f"患者 {test_pid} 预测错误")
    #     else:
    #         print(f"患者 {test_pid} 预测正确")
    #     # 保存模型
    #     joblib.dump(model, f'experiment7_models\\lr_patient_{index}_{seed}_{band_name}_{feature_name}.pkl')
    #     print(f'预测完患者: {test_pid}，数据条数：{len(X_test)}')
    # exit()
    ###################################################################################################################
    # shap分析结果:https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html#SHAP-Decision-Plots
    # 加载PCA模型
    index = 7
    pca = joblib.load(f'experiment7_models\\pca_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')
    # 加载标准化器
    scaler = joblib.load(f'experiment7_models\\scaler_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')
    # 加载模型
    model = joblib.load(f'experiment7_models\\lr_patient_{index-1}_{seed}_{band_name}_{feature_name}.pkl')
    # 获取训练数据和测试数据P07
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
    # 使用PCA降维--lr：30维度效果最好
    X_train = pca.transform(X_train)
    X_test  = pca.transform(X_test)
    # 保留小数点后2位
    X_test = np.round(X_test, 2)
    # 添加pca降维后的特征名
    X_train = pd.DataFrame(
        X_train, 
        columns=[f'PC{i+1}' for i in range(X_train.shape[1])]
    )
    # -------------------------- 系数计算与排序 --------------------------
    coef = np.round(model.coef_[0], 4)
    coef_df = pd.DataFrame(coef, index=X_train.columns, columns=['coef'])
    coef_df = coef_df.sort_values(by='coef', ascending=False)
    # -------------------------- 绘图 --------------------------
    fig, ax = plt.subplots(figsize=(8, 10))
    # 正负系数区分颜色
    colors = ["#EB8322" if c > 0 else "#1877AD" for c in coef_df['coef']]
    # 水平柱状图
    bars = ax.barh(coef_df.index, coef_df['coef'], color=colors, height=0.65, edgecolor='black')
    # 0轴竖线
    ax.axvline(x=0, color='black', linewidth=1.0, linestyle='-')
    # 标题与标签
    ax.set_xlabel('Coefficients', fontsize=13, labelpad=10)
    ax.set_ylabel('Features', fontsize=13, labelpad=10)
    # 网格
    ax.grid(axis='x', alpha=0.2, color='gray', linewidth=0.5)
    ax.set_axisbelow(True)
    # 倒置 y 轴：最大放最上面
    ax.invert_yaxis()
    # 紧凑布局
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'experiment6_res\\logistic_regression_coefficients_{index}.tif', dpi=300)
    # 解释器
    explainer = shap.Explainer(model, X_train)
    # 计算SHAP值
    expected_value = explainer.expected_value
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots(figsize=(8, 10))
    # 设置图片大小
    shap.decision_plot(
        expected_value,
        shap_values,
        feature_names=X_train.columns.tolist(),
        link="logit",
        plot_color='Paired',
        show=False,
        feature_display_range=slice(-1, -21, -1)
    )
    print(f'true_label, {y_test.values}')
    plt.tight_layout()
    plt.savefig(f'experiment6_res\\logistic_regression_shap_{index}.tif', dpi=300)
    ##################################################################################################################
    
