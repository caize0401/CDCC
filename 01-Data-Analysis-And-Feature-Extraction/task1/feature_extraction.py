import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_time_domain_features(signal):
    """提取时域特征"""
    features = {}
    
    # 基本统计特征
    features['mean'] = np.mean(signal)
    features['maximum'] = np.max(signal)
    features['minimum'] = np.min(signal)
    features['variance'] = np.var(signal)
    features['std'] = np.std(signal)
    features['skewness'] = ((signal - np.mean(signal))**3).mean() / (np.std(signal)**3)
    features['kurtosis'] = ((signal - np.mean(signal))**4).mean() / (np.std(signal)**4)
    
    # 顺序统计特征
    features['median'] = np.median(signal)
    features['q25'] = np.percentile(signal, 25)
    features['q75'] = np.percentile(signal, 75)
    
    # 线性拟合特征
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, 1)
    features['linear_slope'] = coeffs[0]
    features['linear_intercept'] = coeffs[1]
    y_pred = coeffs[0] * x + coeffs[1]
    features['linear_rss'] = np.sum((signal - y_pred)**2)
    
    # 多项式拟合特征
    coeffs3 = np.polyfit(x, signal, 3)
    coeffs4 = np.polyfit(x, signal, 4)
    features['c3_0'] = coeffs3[0]
    features['c3_1'] = coeffs3[1]
    features['c3_2'] = coeffs3[2]
    features['c3_3'] = coeffs3[3]
    features['c4_0'] = coeffs4[0]
    features['c4_1'] = coeffs4[1]
    features['c4_2'] = coeffs4[2]
    features['c4_3'] = coeffs4[3]
    features['c4_4'] = coeffs4[4]
    
    # 变化量特征
    features['absolute_sum_of_changes'] = np.sum(np.abs(np.diff(signal)))
    features['mean_abs_change'] = np.mean(np.abs(np.diff(signal)))
    
    # 峰值相关特征
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(signal, height=np.mean(signal))
    features['number_peaks'] = len(peaks)
    
    # 质量中心
    features['index_mass_quantile'] = np.sum(np.arange(len(signal)) * signal) / np.sum(signal)
    
    return features

def extract_frequency_domain_features(signal):
    """提取频域特征"""
    # FFT变换
    fft = np.fft.fft(signal)
    magnitude_spectrum = np.abs(fft)
    
    # 只取前半部分（对称性）
    magnitude_spectrum = magnitude_spectrum[:len(magnitude_spectrum)//2]
    freqs = np.fft.fftfreq(len(signal), d=1)[:len(signal)//2]
    
    features = {}
    
    # 频谱重心
    features['spectral_centroid'] = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
    
    # 频谱带宽
    features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * magnitude_spectrum) / np.sum(magnitude_spectrum))
    
    # 频谱滚降点（85%能量点）
    cumsum = np.cumsum(magnitude_spectrum)
    total_energy = cumsum[-1]
    rolloff_idx = np.where(cumsum >= 0.85 * total_energy)[0]
    if len(rolloff_idx) > 0:
        features['spectral_rolloff'] = freqs[rolloff_idx[0]]
    else:
        features['spectral_rolloff'] = freqs[-1]
    
    # 频谱熵
    # 归一化频谱
    normalized_spectrum = magnitude_spectrum / np.sum(magnitude_spectrum)
    # 避免log(0)
    normalized_spectrum = normalized_spectrum + 1e-10
    features['spectral_entropy'] = -np.sum(normalized_spectrum * np.log2(normalized_spectrum))
    
    return features

def extract_time_frequency_features(signal):
    """提取时频域特征（小波能量特征）"""
    import pywt
    
    # 4层离散小波分解，使用db4小波
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    
    features = {}
    
    # 计算每个子带的能量
    for i, coeff in enumerate(coeffs):
        energy = np.sum(coeff**2)
        if i == 0:
            features[f'wavelet_energy_A4'] = energy  # 近似系数
        else:
            features[f'wavelet_energy_D{5-i}'] = energy  # 细节系数
    
    return features

def extract_all_features(signal):
    """提取所有特征"""
    features = {}
    
    # 时域特征
    time_features = extract_time_domain_features(signal)
    features.update(time_features)
    
    # 频域特征
    freq_features = extract_frequency_domain_features(signal)
    features.update(freq_features)
    
    # 时频域特征
    time_freq_features = extract_time_frequency_features(signal)
    features.update(time_freq_features)
    
    return features

def process_dataset(df, dataset_name):
    """处理单个数据集，提取特征"""
    print(f"正在处理{dataset_name}数据集...")
    print(f"数据集形状: {df.shape}")
    
    # 提取特征
    all_features = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(df)}")
        
        signal = row['Force_curve_RoI']
        features = extract_all_features(signal)
        features['CrimpID'] = row['CrimpID']
        features['Wire_cross-section_conductor'] = row['Wire_cross-section_conductor']
        features['Main_label_string'] = row['Main_label_string']
        features['Sub_label_string'] = row['Sub_label_string']
        features['Main-label_encoded'] = row['Main-label_encoded']
        features['Sub_label_encoded'] = row['Sub_label_encoded']
        features['Binary_label_encoded'] = row['Binary_label_encoded']
        features['CFM_label_encoded'] = row['CFM_label_encoded']
        
        all_features.append(features)
    
    # 转换为DataFrame
    features_df = pd.DataFrame(all_features)
    
    # 保存特征数据
    output_path = f"task1/features_{dataset_name}.pkl"
    features_df.to_pickle(output_path)
    print(f"特征数据已保存到: {output_path}")
    
    return features_df

def main():
    """主函数"""
    print("开始特征提取任务...")
    
    # 读取分割后的数据集
    print("读取分割后的数据集...")
    df_035 = pd.read_pickle("datasets/crimp_force_curves_dataset_035.pkl")
    df_05 = pd.read_pickle("datasets/crimp_force_curves_dataset_05.pkl")
    
    # 处理0.35数据集
    features_035 = process_dataset(df_035, "035")
    
    # 处理0.5数据集
    features_05 = process_dataset(df_05, "05")
    
    print("特征提取完成！")
    print(f"0.35数据集特征形状: {features_035.shape}")
    print(f"0.5数据集特征形状: {features_05.shape}")
    
    return features_035, features_05

if __name__ == "__main__":
    features_035, features_05 = main()

