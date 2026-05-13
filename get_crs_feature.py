import pandas as pd

if __name__ == '__main__':
    # 读取患者诊断数据
    df = pd.read_csv(r'patient_diagnosis.csv')
    crs_former = df['crs_former'].copy()
    crs_latter = df['crs_latter'].copy()
    # 遍历crs_former和crs_latter
    crs_diff = []
    for i in range(len(crs_former)):
        print(f'患者: {df.iloc[i]["patient_id"]}，crs_former = {crs_former[i]}，crs_latter = {crs_latter[i]}')
        crs_former[i] =crs_former[i].split('(')[1].replace(')', '')
        crs_latter[i] =crs_latter[i].split('(')[1].replace(')', '')
        crs_diff.append([0 if crs_former[i].split('-')[0] == crs_latter[i].split('-')[0] else 1 if crs_former[i].split('-')[0] <= crs_latter[i].split('-')[0] else 2,
                         0 if crs_former[i].split('-')[1] == crs_latter[i].split('-')[1] else 1 if crs_former[i].split('-')[1] <= crs_latter[i].split('-')[1] else 2,
                         0 if crs_former[i].split('-')[2] == crs_latter[i].split('-')[2] else 1 if crs_former[i].split('-')[2] <= crs_latter[i].split('-')[2] else 2,
                         0 if crs_former[i].split('-')[3] == crs_latter[i].split('-')[3] else 1 if crs_former[i].split('-')[3] <= crs_latter[i].split('-')[3] else 2,
                         0 if crs_former[i].split('-')[4] == crs_latter[i].split('-')[4] else 1 if crs_former[i].split('-')[4] <= crs_latter[i].split('-')[4] else 2,
                         0 if crs_former[i].split('-')[5] == crs_latter[i].split('-')[5] else 1 if crs_former[i].split('-')[5] <= crs_latter[i].split('-')[5] else 2])
    # 把crs_diff添加到df中
    col_names = ['crs_diff_0', 'crs_diff_1', 'crs_diff_2', 'crs_diff_3', 'crs_diff_4', 'crs_diff_5']
    df[col_names] = crs_diff
    # 保留patient_id和crs_diff_0到crs_diff_5
    df = df[['patient_id'] + col_names]
    # 保存到csv文件
    df.to_csv(r'patient_diagnosis_crs.csv', index=False)
   
