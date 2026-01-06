"""
Task2_3 双路径输入对比实验工具函数
支持原始曲线(data1) + 完整特征(data2)的双路径输入
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_datasets():
    """加载双路径数据集"""
    print("🔄 加载双路径数据集...")
    
    base_dir = Path(__file__).parent
    datasets = {}
    
    # 加载原始曲线数据 (data1)
    print("📊 加载原始曲线数据 (data1)...")
    with open(base_dir / 'datasets/data1/crimp_force_curves_dataset_035.pkl', 'rb') as f:
        data1_035 = pickle.load(f)
    with open(base_dir / 'datasets/data1/crimp_force_curves_dataset_05.pkl', 'rb') as f:
        data1_05 = pickle.load(f)
    
    # 加载完整特征数据 (data2)
    print("🔧 加载完整特征数据 (data2)...")
    with open(base_dir / 'datasets/data2/features_035.pkl', 'rb') as f:
        data2_035 = pickle.load(f)
    with open(base_dir / 'datasets/data2/features_05.pkl', 'rb') as f:
        data2_05 = pickle.load(f)
    
    datasets['data1'] = {'035': data1_035, '05': data1_05}
    datasets['data2'] = {'035': data2_035, '05': data2_05}
    
    print("✅ 数据集加载完成")
    return datasets


def align_and_prepare_dual_path(data1, data2):
    """对齐和准备双路径数据"""
    print("🔗 对齐双路径数据...")
    
    # 使用CrimpID进行合并
    merged_data = pd.merge(data1, data2, on='CrimpID', how='inner', suffixes=('_data1', '_data2'))
    
    # 提取原始曲线数据
    raw_data = np.stack(merged_data['Force_curve_RoI'].values)
    
    # 提取特征数据（排除标识列、标签列和原始曲线列）
    exclude_cols = [
        'CrimpID', 'Wire_cross-section_conductor_data1', 'Wire_cross-section_conductor_data2',
        'Force_curve_raw', 'Force_curve_baseline', 'Force_curve_RoI',
        'Main_label_string_data1', 'Main_label_string_data2', 
        'Sub_label_string_data1', 'Sub_label_string_data2',
        'Main-label_encoded_data1', 'Main-label_encoded_data2',
        'Sub_label_encoded_data1', 'Sub_label_encoded_data2', 
        'Binary_label_encoded_data1', 'Binary_label_encoded_data2',
        'CFM_label_encoded_data1', 'CFM_label_encoded_data2'
    ]
    
    feature_cols = [col for col in merged_data.columns if col not in exclude_cols]
    feat_data = merged_data[feature_cols].values
    
    # 使用Sub_label_encoded作为5类故障标签
    labels = merged_data['Sub_label_encoded_data1'].values
    
    print(f"✅ 数据对齐完成: {len(raw_data)} 样本, 原始曲线维度: {raw_data.shape[1]}, 特征维度: {feat_data.shape[1]}")
    
    return raw_data, feat_data, labels


def preprocess_dual_path_data(raw_data, feat_data, labels):
    """预处理双路径数据"""
    print("🔧 预处理双路径数据...")
    
    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # 数据归一化
    raw_scaler = StandardScaler()
    feat_scaler = StandardScaler()
    
    raw_scaled = raw_scaler.fit_transform(raw_data)
    feat_scaled = feat_scaler.fit_transform(feat_data)
    
    # 合并双路径数据
    X_combined = np.concatenate([raw_scaled, feat_scaled], axis=1)
    
    print(f"✅ 预处理完成: 合并特征维度 {X_combined.shape[1]}, 类别数 {len(label_encoder.classes_)}")
    
    return X_combined, y_encoded, label_encoder, raw_scaler, feat_scaler


def split_train_test(X, y, test_size=0.2, random_state=42):
    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true, y_pred):
    """评估预测结果"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_results_to_csv(results, model_name, dataset_type, size_type, output_dir):
    """保存结果到CSV文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建结果DataFrame
    df_results = pd.DataFrame([{
        'model': model_name,
        'dataset_type': dataset_type,
        'size_type': size_type,
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score']
    }])
    
    # 保存到CSV
    filename = f"{model_name}_{dataset_type}_{size_type}.csv"
    filepath = output_dir / filename
    df_results.to_csv(filepath, index=False)
    
    print(f"💾 结果已保存到: {filepath}")
    return filepath


def create_summary_csv(output_dir):
    """创建汇总CSV文件"""
    output_dir = Path(output_dir)
    csv_files = list(output_dir.glob("*.csv"))
    
    if not csv_files:
        print("⚠️ 没有找到CSV文件")
        return
    
    # 合并所有结果
    all_results = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_results.append(df)
    
    summary_df = pd.concat(all_results, ignore_index=True)
    summary_file = output_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"📊 汇总结果已保存到: {summary_file}")
    return summary_file
