"""
Task4_2_1 MLP双路径输入对比实验
训练集: 0.35, 测试集: 0.5
"""
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from common_utils import (
    load_datasets, align_and_prepare_dual_path, preprocess_dual_path_data,
    split_train_test, evaluate_predictions, save_results_to_csv, create_summary_csv
)


def create_mlp_model():
    """创建MLP模型"""
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    return model


def train_and_evaluate_mlp(X_train, X_test, y_train, y_test, dataset_type, size_type, output_dir):
    """训练和评估MLP模型"""
    print(f"开始训练MLP模型 - 双路径输入 - {size_type}")
    
    # 创建模型
    print("创建MLP模型...")
    model = create_mlp_model()
    
    # 训练模型
    print("开始训练MLP模型...")
    model.fit(X_train, y_train)
    print("MLP模型训练完成")
    
    # 预测
    print("进行MLP预测...")
    y_pred = model.predict(X_test)
    print("预测完成")
    
    # 评估
    print("评估MLP模型...")
    results = evaluate_predictions(y_test, y_pred)
    
    print(f"MLP模型性能 - {size_type}:")
    print(f"   准确率: {results['accuracy']:.4f}")
    print(f"   精确率: {results['precision']:.4f}")
    print(f"   召回率: {results['recall']:.4f}")
    print(f"   F1分数: {results['f1_score']:.4f}")
    
    # 保存结果
    save_results_to_csv(results, 'MLP', 'dual_path', size_type, output_dir)
    
    return results


def main():
    """主函数"""
    print("开始Task4_2_1 MLP双路径输入对比实验")
    print("训练集: 0.35, 测试集: 0.5")
    
    # 设置输出目录
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    datasets = load_datasets()
    
    # 获取0.35数据集作为训练集
    print(f"\n{'='*60}")
    print(f"处理训练集: 0.35")
    print(f"{'='*60}")
    
    data1_train = datasets['data1']['035']
    data3_train = datasets['data3']['035']
    
    # 对齐和准备双路径数据
    raw_data_train, feat_data_train, labels_train = align_and_prepare_dual_path(data1_train, data3_train)
    
    # 获取测试集数据用于构建完整的label_encoder
    data1_test = datasets['data1']['05']
    data3_test = datasets['data3']['05']
    raw_data_test, feat_data_test, labels_test = align_and_prepare_dual_path(data1_test, data3_test)
    
    # 获取所有可能的标签（用于构建完整的label_encoder）
    all_labels = np.concatenate([labels_train, labels_test])
    
    # 预处理数据
    X_train, y_train, label_encoder, raw_scaler, feat_scaler = preprocess_dual_path_data(
        raw_data_train, feat_data_train, labels_train, all_labels
    )
    
    print(f"训练集统计:")
    print(f"   样本数: {X_train.shape[0]}")
    print(f"   特征维度: {X_train.shape[1]}")
    print(f"   类别数: {len(label_encoder.classes_)}")
    
    # 处理测试集数据
    print(f"\n{'='*60}")
    print(f"处理测试集: 0.5")
    print(f"{'='*60}")
    
    # 使用训练集的scaler对测试集进行标准化
    raw_scaled_test = raw_scaler.transform(raw_data_test)
    feat_scaled_test = feat_scaler.transform(feat_data_test)
    X_test = np.column_stack([raw_scaled_test, feat_scaled_test])
    
    # 对测试集标签进行编码（使用训练集的label_encoder）
    y_test = label_encoder.transform(labels_test)
    
    print(f"测试集统计:")
    print(f"   样本数: {X_test.shape[0]}")
    print(f"   特征维度: {X_test.shape[1]}")
    print(f"   类别数: {len(label_encoder.classes_)}")
    
    # 训练和评估MLP
    results = train_and_evaluate_mlp(X_train, X_test, y_train, y_test, 'dual_path', '035_to_05', output_dir)
    
    # 创建汇总CSV
    print(f"\n创建汇总结果...")
    create_summary_csv(output_dir)
    
    print(f"\nTask4_2_1 MLP双路径输入对比实验完成!")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
