import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
# 设置字体Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman']

if __name__ == '__main__':
    band_name = 'alpha'
    # 读取SVM和LR的ROC曲线参数
    svm_file = f'experiment1_res\\svm_{band_name}_results.txt'
    with open(svm_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'ROC fpr' in line:
                fpr_svm = np.array(eval(line.split(':')[1]))
            if 'ROC tpr' in line:
                tpr_svm = np.array(eval(line.split(':')[1]))
            if 'ROC AUC' in line:
                roc_auc_svm = float(eval(line.split(':')[1]))
            if 'PR precision' in line:
                precision_svm = np.array(eval(line.split(':')[1]))
            if 'PR recall' in line:
                recall_svm = np.array(eval(line.split(':')[1]))
            if 'AP' in line:
                ap_svm = float(eval(line.split(':')[1]))
    # 读取LR的ROC曲线参数
    lr_file = f'experiment1_res\\lr_{band_name}_results.txt'
    with open(lr_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'ROC fpr' in line:
                fpr_lr = np.array(eval(line.split(':')[1]))
            if 'ROC tpr' in line:
                tpr_lr = np.array(eval(line.split(':')[1]))
            if 'ROC AUC' in line:
                roc_auc_lr = float(eval(line.split(':')[1]))
            if 'PR precision' in line:
                precision_lr = np.array(eval(line.split(':')[1]))
            if 'PR recall' in line:
                recall_lr = np.array(eval(line.split(':')[1]))
            if 'AP' in line:
                ap_lr = float(eval(line.split(':')[1]))
    # 绘制ROC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_svm, tpr_svm, color='blue', linewidth=1,  label=f'SVM ROC curve (AUC = {roc_auc_svm * 100:.2f}%)')
    plt.plot(fpr_lr, tpr_lr, color='green', linewidth=1,  label=f'LR ROC curve (AUC = {roc_auc_lr * 100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
    # 曲线下填充
    plt.fill_between(fpr_svm, tpr_svm, color='blue', alpha=0.2)
    plt.fill_between(fpr_lr, tpr_lr, color='green', alpha=0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('1 - Specificity', fontsize=18, fontweight='bold')
    plt.ylabel('Sensitivity', fontsize=18, fontweight='bold')
    # 坐标刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # 底色透明
    plt.legend(fontsize=16, loc="lower right", frameon=False)
    plt.tight_layout()
    # 保存为dpi300的tif
    plt.savefig(f'experiment1_res\\svm_lr_{band_name}_ROC_curve.tif', dpi=300)
    plt.close()
    # 绘制PR曲线
    plt.figure(figsize=(10, 6))
    # 绘制PR曲线
    plt.plot(recall_lr, precision_lr, color='green', linewidth=1, label=f'LR PR curve (AP = {ap_lr * 100:.2f}%)')
    plt.plot(recall_svm, precision_svm, color='blue', linewidth=1, label=f'SVM PR curve (AP = {ap_svm * 100:.2f}%)')
    # 填充曲线下面积
    plt.fill_between(recall_lr, precision_lr, color='green', alpha=0.2)
    plt.fill_between(recall_svm, precision_svm, color='blue', alpha=0.2)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('Recall', fontsize=18, fontweight='bold')
    plt.ylabel('Precision', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, loc="upper right", frameon=False)
    plt.tight_layout()
    # plt.show()
    # 保存为dpi300的tif
    plt.savefig(f'experiment1_res\\svm_lr_{band_name}_PR_curve.tif', dpi=300)
