from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

if __name__ == '__main__':
    band_name = 'alpha'
    feature_names = ['psd_', 'coherence_', 'crs_2_', 'psd_coherence_', 'psd_crs_2_', 'coherence_crs_2_']
    feature_name = feature_names[5]
    ###################################################################################
    # # 读取svm模型的预测结果和真实样本标签
    # file_path = 'experiment1_res\\svm_' + band_name + '_results.txt'
    # with open(file_path, 'r') as f:
    #     lines = f.readlines()
    #     predict_svm = eval(lines[0].split(': ')[1].strip())
    #     true_svm = eval(lines[2].split(': ')[1].strip())
    #     # 如果predict_svm的元素一致，保存为1，否则保存为0
    #     if_svm_predict_correct = [1 if i == j else 0 for i, j in zip(predict_svm, true_svm)]
    # # 读取lr模型的预测结果和真实样本标签
    # file_path = 'experiment1_res\\lr_' + band_name + '_results.txt'
    # with open(file_path, 'r') as f:
    #     lines = f.readlines()
    #     predict_lr = eval(lines[0].split(': ')[1].strip())
    #     true_lr = eval(lines[2].split(': ')[1].strip())
    #     # 如果predict_lr的元素一致，保存为1，否则保存为0
    #     if_lr_predict_correct = [1 if i == j else 0 for i, j in zip(predict_lr, true_lr)]
    # exit()
    ######################################################################################
    # 计算实验2的p-value
    file_path = 'experiment2_res\\svm_' + band_name + '_' + feature_name + '_results.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        predict_svm = eval(lines[0].split(': ')[1].strip())
        true_svm = eval(lines[2].split(': ')[1].strip())
        # 如果predict_svm的元素一致，保存为1，否则保存为0
        if_svm_predict_correct = [1 if i == j else 0 for i, j in zip(predict_svm, true_svm)]
    # 读取lr模型的预测结果和真实样本标签
    file_path = 'experiment2_res\\lr_' + band_name + '_' + feature_name + '_results.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        predict_lr = eval(lines[0].split(': ')[1].strip())
        true_lr = eval(lines[2].split(': ')[1].strip())
        # 如果predict_lr的元素一致，保存为1，否则保存为0
        if_lr_predict_correct = [1 if i == j else 0 for i, j in zip(predict_lr, true_lr)]
    ######################################################################################
    # 进行McNemar检验
    svm_res = np.array(if_svm_predict_correct)
    lr_res = np.array(if_lr_predict_correct)
    
    # 构建2×2列联表并计算
    table = np.zeros((2, 2))
    for i in range(len(svm_res)):
        table[svm_res[i], lr_res[i]] += 1
    
    # 进行McNemar检验
    result = mcnemar(table, exact=True)
    print('p-value:', round(result.pvalue, 4))
