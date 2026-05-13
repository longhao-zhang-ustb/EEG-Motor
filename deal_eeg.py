import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import numpy as np
from mne.time_frequency import tfr_morlet
from mne.time_frequency import psd_array_multitaper
from mne.filter import filter_data
from scipy.signal import coherence
import os
import pandas as pd
import EntropyHub as EH
import pywt
from mne_features.univariate import compute_wavelet_coef_energy, compute_mean, compute_std, compute_skewness, compute_kurtosis

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 设置绘图字体为新罗马
plt.rcParams['font.sans-serif'] = ['Times New Roman']

if __name__ == '__main__':
    patient_id = ['P01_0', 'P02_0', 'P03_0', 'P04_0', 'P05_0', 'P06_0', 'P07_0', 'P08_0', 'P09_0',
                  'P10_0', 'P11_0', 'P12_0', 'P13_0', 'P14_0', 'P15_0', 'P16_0', 'P17_0', 'P18_0',
                  'P19_0', 'P20_0', 'P21_0', 'P22_0', 'P23_0', 'P24_0', 'P25_0', 'P26_0', 'P27_0',
                  'P28_0', 'P29_0', 'P30_0', 'P31_0', 'P32_0', 'P33_0', 'P34_0', 'P35_0', 'P36_0',
                  'P37_0', 'P38_0', 'P39_0', 'P40_0', 'P41_0', 'P42_0', 'P43_0', 'P44_0', 'P45_0',
                  'P46_1', 'P47_1', 'P48_1', 'P49_1', 'P50_1', 'P51_1', 'P52_1', 'P53_1', 'P54_1', 
                  'P55_1', 'P56_1', 'P57_1', 'P58_1', 'P59_1', 'P60_1', 'P61_1', 'P62_1', 'P63_1',
                  'P64_1', 'P65_1', 'P66_1', 'P67_1', 'P68_1', 'P69_1', 'P70_1', 'P71_1', 'P72_1',
                  'P73_1', 'P74_1', 'P75_1', 'P76_1', 'P77_1', 'P78_1']
    sd_event_id = '1' # 这里必须是字符型
    have_header = False
    # 判断csv文件是否存在，存在则删除并重新创建
    if os.path.exists('eeg_features.csv'):
        os.remove('eeg_features.csv')
    for id, patient in enumerate(patient_id):
        eeg_folder = rf'dataset\\eeg\\original\\AD_EEG_' + patient.split('_')[0]
        # 列出eeg_folder下的所有文件
        eeg_files = os.listdir(eeg_folder)
        for eeg_file in eeg_files:
            eeg_features = {}
            eeg_features['patient_id'] = patient.split('_')[0]
            eeg_features['label'] = patient.split('_')[1]
            eeg_file_path = os.path.join(eeg_folder, eeg_file)
            try:
                raw = mne.io.read_raw_cnt(
                    eeg_file_path, 
                    preload=True,
                    eog=['HEOG', 'VEOG'],
                    data_format='auto'
                )
            except:
                raw = mne.io.read_raw_cnt(
                    eeg_file_path, 
                    preload=True,
                    eog=['HEOL', 'HEOR', 'VEOU', 'VEOL'],
                    data_format='auto'
                )
            raw._data = np.asarray(raw._data, dtype=np.float64)
            # 设置电极位置
            montage = mne.channels.make_standard_montage('standard_1020')
            channel_rename = {
                'FP1':'Fp1', 'FP2':'Fp2',
                'FZ':'Fz', 'FCZ':'FCz',
                'CZ':'Cz', 'CPZ':'CPz',
                'PZ':'Pz', 'OZ':'Oz'
            }
            raw.rename_channels(channel_rename)
            raw.set_montage(montage, on_missing='ignore')
            # 陷波滤波器去除50Hz工频干扰
            raw = raw.notch_filter(freqs=50)
            # 滤波，保留所需频段
            raw = raw.filter(l_freq=0.5, h_freq=60)
            # 重参考与滤波
            raw.set_eeg_reference('average')
            # 去除眼动伪迹
            ica = ICA(n_components=20, method='fastica', random_state=42)
            ica.fit(raw)
            # 自动检测眼电成分
            # eog_epochs = mne.preprocessing.create_eog_epochs(raw)
            try:
                eog_indices, _ = ica.find_bads_eog(raw, ch_name=['HEOG', 'VEOG'])
            except:
                eog_indices, _ = ica.find_bads_eog(raw, ch_name=['HEOL', 'HEOR', 'VEOU', 'VEOL'])
            ica.exclude = eog_indices
            # 应用ICA
            raw_clean = ica.apply(raw.copy())
            # 定义事件
            events, event_id = mne.events_from_annotations(raw)
            if len(events) == 0:
                continue
            # 创建epochs
            epochs = mne.Epochs(
                raw_clean, 
                events, 
                event_id=event_id, 
                tmin=-0.2, # 刺激前200ms 0.2
                tmax=0.8, # 刺激后800ms 0.8
                baseline=(-0.2, 0), # 基线矫正 -0.2
                preload=True
            )
            try:
                print('这段代码执行了', epochs[sd_event_id])
            except:
                continue
            # ERP分析（事件相关电位分析）
            channels_array = ['Pz', 'P3', 'P4', 'Cz', 'Fz', 'FCz', 'Oz', 'O1', 'O2']
            # 计算功率谱密度
            for ci, channel in enumerate(channels_array):
                # 计算当前通道的ERP响应
                sig = epochs[sd_event_id].copy().pick([channel]).get_data(copy=False)
                sig_long = sig.reshape(-1)
                fmin_array = np.array([0.5, 4, 8, 12, 30])
                fmax_array = np.array([4, 8, 12, 30, 60])
                band_name_array = np.array(['delta', 'theta', 'alpha', 'beta', 'gamma'])
                for index, item in enumerate(fmin_array):
                    # 提取不同波段, Pz-delta
                    psd, freqs = psd_array_multitaper(
                        sig_long,
                        sfreq=raw.info['sfreq'],
                        fmin=fmin_array[index],
                        fmax=fmax_array[index],
                        adaptive=True,
                        verbose=False
                    )
                    eeg_features[f'{channel}_{band_name_array[index]}_psd'] = psd.mean()
                             
            # 计算两个通道之间的同一波段的相干值
            for ci, channel in enumerate(channels_array):
                for cj, channel_j in enumerate(channels_array):
                    if ci == cj or ci > cj:
                        continue
                    fs = raw.info['sfreq']
                    fmin_array = np.array([0.5, 4, 8, 12, 30])
                    fmax_array = np.array([4, 8, 12, 30, 60])
                    band_name_array = np.array(['delta', 'theta', 'alpha', 'beta', 'gamma'])
                    for index, item in enumerate(fmin_array):
                        # 提取不同波段
                        fmin = fmin_array[index]
                        fmax = fmax_array[index]
                        exp_channel_evoked = epochs[sd_event_id].pick([channel])
                        sig1 = exp_channel_evoked.get_data(copy=False)
                        exp_channel_j_evoked = epochs[sd_event_id].pick([channel_j])
                        sig2 = exp_channel_j_evoked.get_data(copy=False)
                        sig1_long = sig1.reshape(-1)
                        sig1_long = filter_data(sig1_long, sfreq=fs, filter_length='auto', l_freq=fmin, h_freq=fmax, verbose=False)
                        sig2_long = sig2.reshape(-1)
                        sig2_long = filter_data(sig2_long, sfreq=fs, filter_length='auto', l_freq=fmin, h_freq=fmax, verbose=False)
                        fxy, Cxy = coherence(
                            sig1_long,
                            sig2_long,
                            fs=fs,
                            nperseg=256
                        )
                        eeg_features[f'{channel}_{channel_j}_{band_name_array[index]}_coherence'] = Cxy.mean()
            print(eeg_file_path, '文件处理完成')
            
            # 将结果追加到csv文件
            # 根据字典的键创建表头
            if not have_header:
                headers = list(eeg_features.keys())
                df = pd.DataFrame([headers])
                df.to_csv('eeg_features.csv', mode='a', header=False, index=False)
                have_header = True
            df = pd.DataFrame([eeg_features])
            df.to_csv('eeg_features.csv', mode='a', header=False, index=False)
