"""
Random Forest双路径输入对比实验 - 简化版
"""
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from common_utils import (
    load_datasets, align_and_prepare_dual_path, preprocess_dual_path_data,
    split_train_test, evaluate_predictions, save_results_to_csv, create_summary_csv
)


def create_random_forest_model():
    """创建Random Forest模型"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    return model


def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, dataset_type, size_type, output_dir):
    """训练和评估Random Forest模型"""
    print(f"开始训练Random Forest模型 - 双路径输入 - {size_type}")
    
    # 创建模型
    print("创建Random Forest模型...")
    model = create_random_forest_model()
    
    # 训练模型
    print("开始训练Random Forest模型...")
    model.fit(X_train, y_train)
    print("Random Forest模型训练完成")
    
    # 预测
    print("进行Random Forest预测...")
    y_pred = model.predict(X_test)
    print("预测完成")
    
    # 评估
    print("评估Random Forest模型...")
    results = evaluate_predictions(y_test, y_pred)
    
    print(f"Random Forest模型性能 - {size_type}:")
    print(f"   准确率: {results['accuracy']:.4f}")
    print(f"   精确率: {results['precision']:.4f}")
    print(f"   召回率: {results['recall']:.4f}")
    print(f"   F1分数: {results['f1_score']:.4f}")
    
    # 保存结果
    save_results_to_csv(results, 'RandomForest', 'dual_path', size_type, output_dir)
    
    return results


def main():
    """主函数"""
    print("开始Random Forest双路径输入对比实验")
    
    # 设置输出目录
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    datasets = load_datasets()
    
    all_results = []
    
    # 对每个数据集大小进行实验
    for size_type in ['035', '05']:
        print(f"\n{'='*60}")
        print(f"处理数据集大小: {size_type}")
        print(f"{'='*60}")
        
        # 获取数据
        data1 = datasets['data1'][size_type]
        data2 = datasets['data2'][size_type]
        
        # 对齐和准备双路径数据
        raw_data, feat_data, labels = align_and_prepare_dual_path(data1, data2)
        
        # 预处理数据
        X, y, label_encoder, raw_scaler, feat_scaler = preprocess_dual_path_data(raw_data, feat_data, labels)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
        
        print(f"数据统计:")
        print(f"   训练集: {X_train.shape[0]} 样本")
        print(f"   测试集: {X_test.shape[0]} 样本")
        print(f"   特征维度: {X_train.shape[1]}")
        print(f"   类别数: {len(label_encoder.classes_)}")
        
        # 训练和评估Random Forest
        results = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, 'dual_path', size_type, output_dir)
        all_results.append(results)
    
    # 创建汇总CSV
    print(f"\n创建汇总结果...")
    create_summary_csv(output_dir)
    
    print(f"\nRandom Forest双路径输入对比实验完成!")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
