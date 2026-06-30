import pandas as pd

if __name__ == '__main__':
    # 读取experiment1_res/lr_alpha_results.txt
    true_label = 0
    predicted_label = 0
    with open(r'experiment1_res\\svm_alpha_results.txt', encoding='utf-8') as file:
        for line in file:
            if line.startswith("True labels"):
                true_label = eval(line.split(": ")[1])
            if line.startswith("Predicted labels"):
                predicted_label = eval(line.split(": ")[1])
    # 读取诊断结果
    df = pd.read_csv(r'dataset\\patient_diagnosis.csv')
    # 病因组成
    abi_true = []
    abi_predict = []
    tbi_true = []
    tbi_predict = []
    cvd_true = []
    cvd_predict = []
    # BCI范式
    photogrph_true = []
    photograph_predict = []
    number_true = []
    number_predict = []
    audio_true = []
    audio_predict = []
    # 遍历列表
    for index, row in df.iterrows():
        if row['etiology'] == 'ABI':
            abi_true.append(true_label[index])
            abi_predict.append(predicted_label[index])
        elif row['etiology'] == 'TBI':
            tbi_true.append(true_label[index])
            tbi_predict.append(predicted_label[index])
        else:
            cvd_true.append(true_label[index])
            cvd_predict.append(predicted_label[index])
        if row['BCI'] == 'Photograph':
            photogrph_true.append(true_label[index])
            photograph_predict.append(predicted_label[index])
        elif row['BCI'] == 'Number':
            number_true.append(true_label[index])
            number_predict.append(predicted_label[index])
        else:
            audio_true.append(true_label[index])
            audio_predict.append(predicted_label[index])
            
    # 打印下数组的长度
    print(len(abi_true))
    print(len(tbi_true))
    print(len(cvd_true))
    print(len(photogrph_true))
    print(len(number_true))
    print(len(audio_true))
    
    print('*'*50)
    # 打印classification_report
    from sklearn.metrics import classification_report
    print(classification_report(abi_true, abi_predict, digits=4))
    print(classification_report(tbi_true, tbi_predict, digits=4))
    print(classification_report(cvd_true, cvd_predict, digits=4))
    print(classification_report(photogrph_true, photograph_predict, digits=4))
    print(classification_report(number_true, number_predict, digits=4))
    print(classification_report(audio_true, audio_predict, digits=4))
    
    from sklearn.metrics import matthews_corrcoef
    print(round(matthews_corrcoef(abi_true, abi_predict), 4))
    print(round(matthews_corrcoef(tbi_true, tbi_predict), 4))
    print(round(matthews_corrcoef(cvd_true, cvd_predict), 4))
    print(round(matthews_corrcoef(photogrph_true, photograph_predict), 4))
    print(round(matthews_corrcoef(number_true, number_predict), 4))
    print(round(matthews_corrcoef(audio_true, audio_predict), 4))
