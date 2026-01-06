"""
H2O AutoML双路径输入对比实验
"""
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from common_utils import (
    load_datasets, align_and_prepare_dual_path, preprocess_dual_path_data,
    split_train_test, evaluate_predictions, save_results_to_csv, create_summary_csv
)


def create_h2o_automl_model():
    """创建H2O AutoML模型"""
    model = H2OAutoML(
        max_models=20,
        seed=42,
        max_runtime_secs=300,  # 5分钟运行时间
        stopping_metric='logloss',
        stopping_tolerance=0.01,
        stopping_rounds=3
    )
    return model


def train_and_evaluate_h2o_automl(X_train, X_test, y_train, y_test, dataset_type, size_type, output_dir):
    """训练和评估H2O AutoML模型"""
    print(f"🔄 开始训练H2O AutoML模型 - 双路径输入 - {size_type}")
    
    # 初始化H2O
    print("🏗️ 初始化H2O...")
    h2o.init(nthreads=-1, max_mem_size='4G')
    
    try:
        # 创建模型
        print("🏗️ 创建H2O AutoML模型...")
        model = create_h2o_automl_model()
        
        # 准备H2O数据格式
        print("📊 准备H2O数据格式...")
        
        # 合并训练数据
        train_data = np.column_stack([X_train, y_train])
        test_data = np.column_stack([X_test, y_test])
        
        # 创建特征列名
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        column_names = feature_names + ['target']
        
        # 转换为H2O Frame
        train_h2o = h2o.H2OFrame(train_data, column_names=column_names)
        test_h2o = h2o.H2OFrame(test_data, column_names=column_names)
        
        # 设置目标列为分类
        train_h2o['target'] = train_h2o['target'].asfactor()
        test_h2o['target'] = test_h2o['target'].asfactor()
        
        # 训练模型
        print("🚀 开始训练H2O AutoML模型...")
        model.train(x=feature_names, y='target', training_frame=train_h2o)
        print("✅ H2O AutoML模型训练完成")
        
        # 预测
        print("🔮 进行H2O AutoML预测...")
        predictions = model.predict(test_h2o)
        y_pred = predictions['predict'].as_data_frame().values.flatten()
        
        # 转换预测结果为整数
        y_pred = [int(x) for x in y_pred]
        print("✅ 预测完成")
        
        # 评估
        print("📊 评估H2O AutoML模型...")
        results = evaluate_predictions(y_test, y_pred)
        
        print(f"📈 H2O AutoML模型性能 - {size_type}:")
        print(f"   准确率: {results['accuracy']:.4f}")
        print(f"   精确率: {results['precision']:.4f}")
        print(f"   召回率: {results['recall']:.4f}")
        print(f"   F1分数: {results['f1_score']:.4f}")
        
        # 保存结果
        save_results_to_csv(results, 'H2O_AutoML', 'dual_path', size_type, output_dir)
        
        return results
        
    finally:
        # 关闭H2O
        print("🔚 关闭H2O...")
        h2o.cluster().shutdown()


def main():
    """主函数"""
    print("🚀 开始H2O AutoML双路径输入对比实验")
    
    # 设置输出目录
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    datasets = load_datasets()
    
    all_results = []
    
    # 对每个数据集大小进行实验
    for size_type in ['035', '05']:
        print(f"\n{'='*60}")
        print(f"🎯 处理数据集大小: {size_type}")
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
        
        print(f"📊 数据统计:")
        print(f"   训练集: {X_train.shape[0]} 样本")
        print(f"   测试集: {X_test.shape[0]} 样本")
        print(f"   特征维度: {X_train.shape[1]}")
        print(f"   类别数: {len(label_encoder.classes_)}")
        
        # 训练和评估H2O AutoML
        results = train_and_evaluate_h2o_automl(X_train, X_test, y_train, y_test, 'dual_path', size_type, output_dir)
        all_results.append(results)
    
    # 创建汇总CSV
    print(f"\n📊 创建汇总结果...")
    create_summary_csv(output_dir)
    
    print(f"\n✅ H2O AutoML双路径输入对比实验完成!")
    print(f"📁 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
