import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score
from scipy import stats
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from scipy import stats
from rocauc_comparison import delong_roc_test

if __name__ == '__main__':
    # 读取experiment1_res/svm_alpha_results.txt
    svm_true_label = 0
    svm_predicted_label = 0
    with open(r'experiment1_res\\svm_alpha_results.txt', encoding='utf-8') as file:
        for line in file:
            if line.startswith("True labels"):
                svm_true_label = eval(line.split(": ")[1])
            if line.startswith("Predicted probabilities"):
                svm_predicted_label = eval(line.split(": ")[1])
    # 使用bootstrap计算95%置信区间
    # 转为数组
    svm_y_true = np.array(svm_true_label)
    svm_y_pred = np.array(svm_predicted_label)
    
    # 读取experiment1_res/lr_alpha_results.txt
    lr_true_label = 0
    lr_predicted_label = 0
    with open(r'experiment1_res\\lr_alpha_results.txt', encoding='utf-8') as file:
        for line in file:
            if line.startswith("True labels"):
                lr_true_label = eval(line.split(": ")[1])
            if line.startswith("Predicted probabilities"):
                lr_predicted_label = eval(line.split(": ")[1])
    # 使用bootstrap计算95%置信区间
    # 转为数组
    lr_y_true = np.array(lr_true_label)
    lr_y_pred = np.array(lr_predicted_label)
    
    # 计算p-value
    p_value = delong_roc_test(lr_y_true, svm_y_pred, lr_y_pred)
    print(p_value)
