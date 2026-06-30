from rocauc_comparison import delong_roc_test
import numpy as np

if __name__ == '__main__':
    file_list = [
        r'experiment1_res\\svm_delta_results.txt',
        r'experiment1_res\\svm_theta_results.txt',
        r'experiment1_res\\svm_alpha_results.txt',
        r'experiment1_res\\svm_beta_results.txt',
        r'experiment1_res\\svm_gamma_results.txt'
    ]

    all_pred_label = []
    all_true_label = []
    for fp in file_list:
        pp = None
        with open(fp, encoding='utf-8') as f:
            for line in f:
                if line.startswith("True labels"):
                    all_true_label.append(np.array(eval(line.split(": ")[1])))
                if line.startswith("Predicted probabilities"):
                    pp = np.array(eval(line.split(": ")[1]))
        # 二分类阈值划分
        pred_label = (pp >= 0.5).astype(int)
        all_pred_label.append(pp)
    
    delta_pred  = all_pred_label[0]
    theta_pred  = all_pred_label[1]
    alpha_pred  = all_pred_label[2]
    beta_pred   = all_pred_label[3]
    gamma_pred  = all_pred_label[4]

    print(delong_roc_test(all_true_label[0], delta_pred, alpha_pred))
    print(delong_roc_test(all_true_label[0], theta_pred, alpha_pred))
    print(delong_roc_test(all_true_label[0], alpha_pred, alpha_pred))
    print(delong_roc_test(all_true_label[0], beta_pred, alpha_pred))
    print(delong_roc_test(all_true_label[0], gamma_pred, alpha_pred))
