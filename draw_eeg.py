import mne
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import ICA
from mne.filter import filter_data
import numpy as np

# 全局设置 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

if __name__ == '__main__':
    # 读取cnt文件，使用mne库 dataset\eeg\original\AD_EEG_P01\AD_EEG_P01_20130322_1.cnt
    # file_path = r'dataset\eeg\original\AD_EEG_P01\AD_EEG_P01_20130320_1.cnt'
    file_path = r'dataset\eeg\original\AD_EEG_P01\AD_EEG_P01_20130322_1.cnt'
    # file_path = r'dataset\eeg\original\AD_EEG_P05\AD_EEG_P05_20191219_2.cnt'
    
    # 选择需要的通道
    channels_array = ['Pz', 'P3', 'P4', 'Cz', 'Fz', 'FCz', 'Oz', 'O1', 'O2']
    
    # 使用mne库读取cnt文件
    raw = mne.io.read_raw_cnt(
        file_path, 
        preload=True,
        eog=['HEOG', 'VEOG'],
        data_format='auto'
    )
    
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
    raw = raw.filter(l_freq=0.5, h_freq=80)
    
    # # 重参考
    raw.set_eeg_reference('average')
    
    # 去除眼动伪迹
    ica = ICA(n_components=20, method='fastica', random_state=42)
    ica.fit(raw)
    eog_indices, _ = ica.find_bads_eog(raw, ch_name=['HEOG', 'VEOG'])
    ica.exclude = eog_indices
    # 应用ICA
    raw_clean = ica.apply(raw.copy())
    
    # 定义事件
    events, event_id = mne.events_from_annotations(raw)
    # 创建epochs
    epochs = mne.Epochs(
        raw_clean, 
        events, 
        event_id=event_id, 
        tmin=-0.2, # 刺激前200ms 0.2
        tmax=0.8, # 刺激后800ms 0.8
        baseline=(-0.2, 0), # 基线矫正 -0.2
        preload=True,
        picks=channels_array
    )
    # sig = epochs['1'].copy().get_data(copy=False)
    # sig = sig.reshape(-1)
    sig = epochs['1'].copy().get_data(copy=False)
    sig = sig.reshape(len(channels_array), -1)

    # 获取通道的索引
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80)
    }
    # https://bionichaos.com/EEG1020/
    chs = {
        'Pz': [0, -0.038],
        'P3': [-0.04, -0.041],
        'P4': [0.04, -0.041],
        'Cz': [0.0, 0.0],
        'Fz': [0, 0.038],
        'FCz': [0, -0.0092],
        'Oz': [0.0, -0.08],
        'O1': [-0.03, -0.08],
        'O2': [0.03, -0.08]
    }
    indices = [raw.ch_names.index(ch) for ch in channels_array]
    # 计算每个通道的功率谱密度
    psds, freqs = mne.time_frequency.psd_array_welch(
        sig,
        sfreq = raw.info['sfreq'],
        fmin=0.5,
        fmax=80
    )
    band_data = []
    band_names = []
    for band, (fmin, fmax) in BANDS.items():
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        band_power = np.mean(psds[:, idx], axis=1)
        band_data.append(band_power)
        band_names.append(band)
    
    print(band_data)
    print(band_names)
    
    pos_array = []
    for ch in channels_array:
        pos_array.append(chs[ch])
    pos_array = np.array(pos_array)
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    
    for ax, data, name in zip(axes, band_data, band_names):
        im, _ = mne.viz.plot_topomap(
            data,
            pos=pos_array,
            # names=name,
            axes=ax,
            cmap='coolwarm',
            show=False
        )
        # 设置标题
        ax.set_title(name, fontsize=14)
    
    # 统一色条
    # 这个参数专门给色条预留空间，彻底解决显示不全！
    plt.subplots_adjust(right=0.88)
    # 设置色条位置在最后一个子图
    cax = fig.add_axes([0.9, 0.35, 0.005, 0.2])  # [左, 下, 宽, 高]
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Power (µV²/Hz)', fontsize=14)
    # plt.show()
    # 保存图片，dpi=300
    plt.savefig(r'paper_figures\\eeg_signal_power_spectrum_label0.tif', dpi=300)
    
    exit()
    times = np.arange(len(sig_long)) / raw.info['sfreq']
    
    # 过滤脑电信号的多个波段
    fmin_array = np.array([0.5, 4, 8, 12, 30])
    fmax_array = np.array([4, 8, 12, 30, 80])
    # raw_delta = filter_data(sig_long, sfreq=raw.info['sfreq'], l_freq=fmin_array[0], h_freq=fmax_array[0], verbose=False)
    # raw_theta = filter_data(sig_long, sfreq=raw.info['sfreq'], l_freq=fmin_array[1], h_freq=fmax_array[1], verbose=False)
    # raw_alpha = filter_data(sig_long, sfreq=raw.info['sfreq'], l_freq=fmin_array[2], h_freq=fmax_array[2], verbose=False)
    # raw_beta = filter_data(sig_long, sfreq=raw.info['sfreq'], l_freq=fmin_array[3], h_freq=fmax_array[3], verbose=False)
    raw_gamma = filter_data(sig_long, sfreq=raw.info['sfreq'], l_freq=fmin_array[4], h_freq=fmax_array[4], verbose=False)
    
    # 绘制eeg信号
    indices = [raw.ch_names.index(ch) for ch in channels_array]
    
    # 设置图片大小
    plt.figure(figsize=(18, 3))
    
    # 使用seaborn库绘制eeg信号
    sns.set_style(
        'whitegrid',
        rc={
            'font.family': 'Times New Roman'
        }
    )
    sns.lineplot(
        x=times,
        y=raw_gamma * 1e6
        # y=raw_delta.get_data()[indices][0] * 1e6
    )
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Amplitude (uV)', fontsize=14)
    # 设置刻度字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    # 保存图片，dpi=600
    # plt.savefig(r'paper_figures\\original_eeg_signal.tif', dpi=300)
    # plt.savefig(r'paper_figures\\original_eeg_signal_notch.tif', dpi=300)
    # plt.savefig(r'paper_figures\\eeg_signal_filter_0.5_80.tif', dpi=300)
    # plt.savefig(r'paper_figures\\eeg_signal_set_reference.tif', dpi=300)
    # plt.savefig(r'paper_figures\\eeg_signal_drop_eog.tif', dpi=300)
    # 保存不同波段的信号
    # plt.savefig(r'paper_figures\\eeg_signal_delta.tif', dpi=300)
    # plt.savefig(r'paper_figures\\eeg_signal_theta.tif', dpi=300)
    # plt.savefig(r'paper_figures\\eeg_signal_alpha.tif', dpi=300)
    # plt.savefig(r'paper_figures\\eeg_signal_beta.tif', dpi=300)
    plt.savefig(r'paper_figures\\eeg_signal_gamma.tif', dpi=300)
    
    # plt.show()
