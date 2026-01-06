"""
Task4_1_1 双路径输入数据处理工具函数
基于task4_2_1的common_utils.py，适配task4_1_1的实验需求
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def load_datasets():
    """加载双路径数据集"""
    print("加载双路径数据集...")
    
    # 获取数据集目录
    datasets_dir = Path(__file__).parent / 'datasets'
    
    # 加载原始曲线数据 (data1)
    print("加载原始曲线数据 (data1)...")
    data1_035_path = datasets_dir / 'data1' / 'crimp_force_curves_dataset_035.pkl'
    data1_05_path = datasets_dir / 'data1' / 'crimp_force_curves_dataset_05.pkl'
    
    with open(data1_035_path, 'rb') as f:
        data1_035 = pickle.load(f)
    with open(data1_05_path, 'rb') as f:
        data1_05 = pickle.load(f)
    
    # 加载筛选特征数据 (data3)
    print("加载筛选特征数据 (data3)...")
    data3_035_path = datasets_dir / 'data3' / 'features_035_selected.pkl'
    data3_05_path = datasets_dir / 'data3' / 'features_05_selected.pkl'
    
    with open(data3_035_path, 'rb') as f:
        data3_035 = pickle.load(f)
    with open(data3_05_path, 'rb') as f:
        data3_05 = pickle.load(f)
    
    datasets = {
        'data1': {
            '035': data1_035,
            '05': data1_05
        },
        'data3': {
            '035': data3_035,
            '05': data3_05
        }
    }
    
    print("数据集加载完成")
    return datasets


def align_and_prepare_dual_path(data1, data3):
    """对齐和准备双路径数据"""
    print("对齐双路径数据...")
    
    # 提取原始曲线数据 - Force_curve_RoI是numpy数组的列表
    raw_curves_list = data1['Force_curve_RoI'].values
    raw_curves = np.array([curve for curve in raw_curves_list])
    raw_labels = data1['Sub_label_encoded'].values
    
    # 提取特征数据
    feature_cols = [col for col in data3.columns if col not in ['CrimpID', 'Sub_label_encoded', 'Wire_cross-section_conductor', 'Main_label_string', 'Sub_label_string', 'Main-label_encoded', 'Binary_label_encoded', 'CFM_label_encoded']]
    features = data3[feature_cols].values
    feature_labels = data3['Sub_label_encoded'].values
    
    print(f"数据对齐完成: {len(raw_curves)} 样本, 原始曲线维度: {raw_curves.shape[1]}, 特征维度: {features.shape[1]}")
    
    return raw_curves, features, raw_labels


def preprocess_dual_path_data(raw_data, feat_data, labels, all_labels=None):
    """预处理双路径数据"""
    print("预处理双路径数据...")
    
    # 标准化原始曲线数据
    raw_scaler = StandardScaler()
    raw_scaled = raw_scaler.fit_transform(raw_data)
    
    # 标准化特征数据
    feat_scaler = StandardScaler()
    feat_scaled = feat_scaler.fit_transform(feat_data)
    
    # 合并双路径数据
    X = np.column_stack([raw_scaled, feat_scaled])
    
    # 标签编码 - 使用训练集和测试集的并集来构建encoder（参考task4_2）
    label_encoder = LabelEncoder()
    if all_labels is not None:
        # 使用所有标签来fit encoder，确保支持完整的类别
        label_encoder.fit(all_labels)
        y = label_encoder.transform(labels)
    else:
        y = label_encoder.fit_transform(labels)
    
    print(f"预处理完成: 合并特征维度 {X.shape[1]}, 类别数 {len(label_encoder.classes_)}")
    
    return X, y, label_encoder, raw_scaler, feat_scaler


def split_train_test(X, y, test_size=0.2, random_state=42):
    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true, y_pred):
    """评估预测结果"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_results_to_csv(results, model_name, dataset_type, size_type, output_dir):
    """保存结果到CSV文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 创建结果DataFrame
    result_data = {
        'model': [model_name],
        'dataset_type': [dataset_type],
        'size_type': [size_type],
        'accuracy': [results['accuracy']],
        'precision': [results['precision']],
        'recall': [results['recall']],
        'f1_score': [results['f1_score']]
    }
    
    df = pd.DataFrame(result_data)
    
    # 保存到CSV
    filename = f"{model_name}_{dataset_type}_{size_type}.csv"
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    
    print(f"结果已保存到: {filepath}")


def create_summary_csv(output_dir):
    """创建汇总CSV文件"""
    output_dir = Path(output_dir)
    
    # 收集所有CSV文件
    csv_files = list(output_dir.glob("*.csv"))
    
    if not csv_files:
        print("没有找到结果文件")
        return
    
    # 读取所有CSV文件并合并
    all_results = []
    for csv_file in csv_files:
        if csv_file.name != 'summary.csv':  # 排除汇总文件本身
            df = pd.read_csv(csv_file)
            all_results.append(df)
    
    if all_results:
        # 合并所有结果
        summary_df = pd.concat(all_results, ignore_index=True)
        
        # 保存汇总文件
        summary_path = output_dir / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"汇总结果已保存到: {summary_path}")
    else:
        print("没有找到有效的结果文件")
