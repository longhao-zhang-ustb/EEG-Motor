import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
# 设置字体为Times New Roman
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# ===================== 核心：平滑曲线处理 =====================
def smooth_curve(x, y):
    """
    对曲线进行平滑插值处理
    x: 原始x坐标
    y: 原始y坐标（你的F1/MCC结果）
    return: 平滑后的新x、新y
    """
    # 生成更多的插值点（500个点让曲线足够顺滑）
    x_new = np.linspace(min(x), max(x), 500)  
    # 三次样条插值（最常用、效果最好的平滑方式）
    f = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")
    y_new = f(x_new)
    return x_new, y_new

if __name__ == '__main__':
    lr_file_path = r'experiment3_res\\lr_alpha_psd_coherence_crs_2__results.txt'
    svm_file_path  = r'experiment3_res\\svm_alpha_psd_coherence_crs_2__results.txt'
    # 读取svm的结果txt文件
    svm_f1_res = []
    svm_mcc_res = []
    lr_f1_res = []
    lr_mcc_res = []
    with open(lr_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'F1-score' in line:
                lr_f1_res.append(float(line.split(': ')[1].strip()))
            elif 'MatthewCC' in line:
                lr_mcc_res.append(float(line.split(': ')[1].strip()))
    with open(svm_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'F1-score' in line:
                svm_f1_res.append(float(line.split(': ')[1].strip()))
            elif 'MatthewCC' in line:
                svm_mcc_res.append(float(line.split(': ')[1].strip()))
    # # 根据wilcoxon检验判断LR和SVM的F1-score是否有显著差异
    # from scipy.stats import wilcoxon    
    # t_stat, p_val = wilcoxon(lr_f1_res, svm_f1_res)
    # print(f'F1-score t-statistic: {t_stat:.2f}, p-value: {p_val:.4f}')  
    # # 根据wilcoxon检验判断LR和SVM的MatthewCC是否有显著差异
    # t_stat, p_val = wilcoxon(lr_mcc_res, svm_mcc_res)
    # print(f'MCC: t-statistic: {t_stat:.2f}, p-value: {p_val:.4f}')
    # exit()
    
    # 原始x轴（维度 0,1,2,3...）
    x_ori = np.arange(1, len(lr_f1_res)+1)
    # 对LR和SVM分别做平滑
    x_smooth, lr_smooth = smooth_curve(x_ori, lr_f1_res)
    x_smooth, svm_smooth = smooth_curve(x_ori, svm_f1_res)
    # ===================== 找最高点位置 =====================
    # LR 最高点
    lr_max_val = max(lr_f1_res)
    lr_max_idx = lr_f1_res.index(lr_max_val) + 1

    # SVM 最高点
    svm_max_val = max(svm_f1_res)
    svm_max_idx = svm_f1_res.index(svm_max_val) + 1
    
    plt.figure(figsize=(10, 6))
    # 根据lr_f1_res和svm_f1_res绘制曲线图
    plt.plot(x_smooth, lr_smooth, label='LR', linewidth=3, color='#1f77b4')
    plt.plot(x_smooth, svm_smooth, label='SVM', linewidth=3, color='#ff7f0e')
    
    # 标记 LR 最高点
    plt.scatter(lr_max_idx, lr_max_val, color='red', s=120, marker='*', zorder=10, label='LR Max')
    # 从上到下画竖线
    plt.axvline(x=lr_max_idx, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    plt.text(lr_max_idx+0.2, lr_max_val, f'{lr_max_val * 100:.2f}%', fontsize=14, fontweight='bold', color='black')

    # 标记 SVM 最高点
    plt.scatter(svm_max_idx, svm_max_val, color='lightgreen', s=120, marker='*', zorder=10, label='SVM Max')
    plt.axvline(x=svm_max_idx, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    plt.text(svm_max_idx+0.2, svm_max_val, f'{svm_max_val * 100:.2f}%', fontsize=14, fontweight='bold', color='black')
    
    plt.xlabel('Dimension', fontsize=16, fontweight='bold')
    plt.ylabel('Ma-F', fontsize=16, fontweight='bold')
    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=16)
    # plt.show()
    # 保存为dpi=300的tif
    plt.savefig(r'experiment3_res\\lr_svm_f1.tif', dpi=300)
    # exit()
    ###############################################################################
    # 原始x轴（维度 0,1,2,3...）
    x_mcc_ori = np.arange(1, len(lr_mcc_res)+1)
    # 对LR和SVM分别做平滑
    x_smooth, lr_smooth = smooth_curve(x_mcc_ori, lr_mcc_res)
    x_smooth, svm_smooth = smooth_curve(x_mcc_ori, svm_mcc_res)
    # ===================== 找最高点位置 =====================
    # LR 最高点
    lr_max_val = max(lr_mcc_res)
    lr_max_idx = lr_mcc_res.index(lr_max_val) + 1

    # SVM 最高点
    svm_max_val = max(svm_mcc_res)
    svm_max_idx = svm_mcc_res.index(svm_max_val) + 1
    
    plt.figure(figsize=(10, 6))
    # 根据lr_f1_res和svm_f1_res绘制曲线图
    plt.plot(x_smooth, lr_smooth, label='LR', linewidth=3, color='#1f77b4')
    plt.plot(x_smooth, svm_smooth, label='SVM', linewidth=3, color='#ff7f0e')
    
    # 标记 LR 最高点
    plt.scatter(lr_max_idx, lr_max_val, color='red', s=120, marker='*', zorder=10, label='LR Max')
    # 从上到下画竖线
    plt.axvline(x=lr_max_idx, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    plt.text(lr_max_idx+0.2, lr_max_val, f'{lr_max_val * 100:.2f}%', fontsize=14, fontweight='bold', color='black')

    # 标记 SVM 最高点
    plt.scatter(svm_max_idx, svm_max_val, color='lightgreen', s=120, marker='*', zorder=10, label='SVM Max')
    plt.axvline(x=svm_max_idx, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    plt.text(svm_max_idx+0.2, svm_max_val, f'{svm_max_val * 100:.2f}%', fontsize=14, fontweight='bold', color='black')
    
    plt.xlabel('Dimension', fontsize=16, fontweight='bold')
    plt.ylabel('MCC', fontsize=16, fontweight='bold')
    # 设置坐标刻度字体大小
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=16)
    # plt.show()
    plt.savefig(r'experiment3_res\\lr_svm_mcc.tif', dpi=300)
    
