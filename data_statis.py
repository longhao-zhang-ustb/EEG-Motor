import pandas as pd
from scipy.stats import ttest_ind

if __name__ == '__main__':
    df = pd.read_csv(r'dataset\\patient_diagnosis.csv')
    print(df.head())
    print(f'年龄分布情况为：{round(df["age"].mean(), 2)} ± {round(df["age"].std(), 2)}')
    # 计算time since injury的最大和最小值
    print(f'time since injury的最大值为：{df["time_since_injury"].max()}')
    print(f'time since injury的最小值为：{df["time_since_injury"].min()}')
    # 根据label区分意识恢复和意识未恢复
    df_recover = df[df['label'] == 1]
    df_not_recover = df[df['label'] == 0]
    # 统计意识恢复和意识未恢复的样本数量
    print(f'意识恢复的样本数量为：{len(df_recover)}')
    print(f'意识未恢复的样本数量为：{len(df_not_recover)}')
    exit()
    # 统计意识恢复的age分布情况
    print(f'意识恢复的年龄分布情况为：{round(df_recover["age"].mean())} ± {round(df_recover["age"].std())}')
    # 统计意识未恢复的age分布情况
    print(f'意识未恢复的年龄分布情况为：{round(df_not_recover["age"].mean())} ± {round(df_not_recover["age"].std())}')
    # 进行t检验
    t, p = ttest_ind(df_recover["age"], df_not_recover["age"], equal_var=False)
    print(f't检验统计量为：{round(t, 4)}')
    print(f'p值为：{round(p, 4)}')
    # 统计意识恢复患者的性别分布情况
    print(f'意识恢复患者的性别分布情况为：{df_recover["gender"].value_counts()}')
    # 统计意识未恢复患者的性别分布情况
    print(f'意识未恢复患者的性别分布情况为：{df_not_recover["gender"].value_counts()}')
    # 统计意识恢复患者的病因分布情况
    print(f'意识恢复患者的病因分布情况为：{df_recover["etiology"].value_counts()}')
    # 统计意识未恢复患者的病因分布情况
    print(f'意识未恢复患者的病因分布情况为：{df_not_recover["etiology"].value_counts()}')
    # 统计意识恢复患者的time since injury的分布情况：均值和标准差
    print(f'意识恢复患者的time since injury的分布情况为：{round(df_recover["time_since_injury"].mean(), 2)} ± {round(df_recover["time_since_injury"].std(), 2)}')
    print(f'意识未恢复患者的time since injury的分布情况为：{round(df_not_recover["time_since_injury"].mean(), 2)} ± {round(df_not_recover["time_since_injury"].std(), 2)}')
    # 进行t检验：time since injury的分布情况
    t, p = ttest_ind(df_recover["time_since_injury"], df_not_recover["time_since_injury"], equal_var=False)
    print(f't检验统计量为：{round(t, 4)}')
    print(f'p值为：{round(p, 4)}')
    # 统计crs_diff_2的分布情况数量
    print(f'意识恢复患者crs_diff_2的分布情况数量为：{df_recover["crs_diff_2"].value_counts()}')
    # 统计cr未恢复患者crs_diff_2的分布情况数量
    print(f'意识未恢复患者crs_diff_2的分布情况数量为：{df_not_recover["crs_diff_2"].value_counts()}')
    # 统计BCI范式的分布情况
    print(f'意识恢复患者的BCI范式的分布情况为：{df_recover["BCI"].value_counts()}')
    print(f'意识未恢复患者的BCI范式的分布情况为：{df_not_recover["BCI"].value_counts()}')
    # 实验前的意识情况统计
    print(f'意识恢复患者的实验前意识情况为：{df_recover["before_exp"].value_counts()}')
    print(f'意识未恢复患者的实验前意识情况为：{df_not_recover["before_exp"].value_counts()}')
    # 实验后的情况统计
    print(f'意识恢复患者的实验后情况为：{df_recover["after_exp"].value_counts()}')
    print(f'意识未恢复患者的实验后情况为：{df_not_recover["after_exp"].value_counts()}')
    
    
