import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from scipy.stats import bootstrap

if __name__ == '__main__':
    # 读取experiment1_res/lr_alpha_results.txt
    true_label = 0
    predicted_label = 0
    proba_label = 0
    with open(r'experiment1_res\\lr_alpha_results.txt', encoding='utf-8') as file:
        for line in file:
            if line.startswith("True labels"):
                true_label = eval(line.split(": ")[1])
            if line.startswith("Predicted labels"):
                predicted_label = eval(line.split(": ")[1])
            if line.startswith("Predicted probabilities"):
                proba_label = eval(line.split(": ")[1])
    # 使用bootstrap计算95%置信区间
    # 转为数组
    y_true = np.array(true_label)
    y_pred = np.array(predicted_label)
    y_score = np.array(proba_label)  # 有概率再启用

    # ===================== Bootstrap 计算95%CI =====================
    def bootstrap_metric_ci(y_true_arr, y_pred_arr, y_score_arr, n_boot=1000, random_seed=42):
        np.random.seed(random_seed)
        n_sample = len(y_true_arr)
        
        acc_list = []
        mcc_list = []
        p_list = []
        r_list = []
        f1_list = []
        auc_list = []
        kappa_list = []

        for _ in range(n_boot):
            # 有放回抽样，每次抽和原数据一样多的样本
            idx = np.random.choice(n_sample, size=n_sample, replace=True)
            yt = y_true_arr[idx]
            yp = y_pred_arr[idx]
            ys = y_score_arr[idx]

            acc_list.append(accuracy_score(yt, yp))
            mcc_list.append(matthews_corrcoef(yt, yp))
            p_list.append(precision_score(yt, yp, average='macro'))
            r_list.append(recall_score(yt, yp, average='macro'))
            f1_list.append(f1_score(yt, yp, average='macro'))
            auc_list.append(roc_auc_score(yt, ys))
            kappa_list.append(cohen_kappa_score(yt, yp))

        # 计算95%置信区间 2.5% 和 97.5%分位数
        def get_ci(data, conf_level=0.95):
            # return np.percentile(data, [2.5, 97.5])
            """
            BCa校正Bootstrap计算95%置信区间
            :param data_arr: 每次重抽样得到的指标一维列表/数组
            :param conf_level: 置信水平 默认0.95
            :return: [下界, 上界]
            """
            data = np.asarray(data)
            return np.percentile(data, [2.5, 97.5])
        
        acc_ci = get_ci(acc_list)
        mcc_ci = get_ci(mcc_list)
        p_ci = get_ci(p_list)
        r_ci = get_ci(r_list)
        f1_ci = get_ci(f1_list)
        auc_ci = get_ci(auc_list)
        kappa_ci = get_ci(kappa_list)

        return {
            "acc_mean": np.mean(acc_list),
            "acc_ci": acc_ci,
            "mcc_mean": np.mean(mcc_list),
            "mcc_ci": mcc_ci,
            "p_mean": np.mean(p_list),
            "p_ci": p_ci,
            "r_mean": np.mean(r_list),
            "r_ci": r_ci,
            "f1_mean": np.mean(f1_list),
            "f1_ci": f1_ci,
            "auc_mean": np.mean(auc_list),
            "auc_ci": auc_ci,
            "kappa_mean": np.mean(kappa_list),
            "kappa_ci": kappa_ci
        }

    # 执行计算
    res = bootstrap_metric_ci(y_true, y_pred, y_score, n_boot=1000)

    # 打印结果
    print("===== Bootstrap 95% 置信区间结果 =====")
    print(f"准确率: {res['acc_mean']:.4f} (95%CI: {res['acc_ci'][0]:.4f} ~ {res['acc_ci'][1]:.4f})")
    print(f"Precision: {res['p_mean']:.4f} (95%CI: {res['p_ci'][0]:.4f} ~ {res['p_ci'][1]:.4f})")
    print(f"Recall: {res['r_mean']:.4f} (95%CI: {res['r_ci'][0]:.4f} ~ {res['r_ci'][1]:.4f})")
    print(f"F1分数: {res['f1_mean']:.4f} (95%CI: {res['f1_ci'][0]:.4f} ~ {res['f1_ci'][1]:.4f})")
    print(f"AUC: {res['auc_mean']:.4f} (95%CI: {res['auc_ci'][0]:.4f} ~ {res['auc_ci'][1]:.4f})")
    print(f"MCC系数: {res['mcc_mean']:.4f} (95%CI: {res['mcc_ci'][0]:.4f} ~ {res['mcc_ci'][1]:.4f})")
    print(f"Cohen's Kappa: {res['kappa_mean']:.4f} (95%CI: {res['kappa_ci'][0]:.4f} ~ {res['kappa_ci'][1]:.4f})")
