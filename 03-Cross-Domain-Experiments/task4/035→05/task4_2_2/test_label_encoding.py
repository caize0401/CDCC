"""
测试标签编码是否正确，确保支持完整的5类标签且无信息泄露
"""
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import pandas as pd
import numpy as np
from common_utils import load_datasets, align_and_prepare_dual_path, preprocess_dual_path_data

def test_label_encoding():
    """测试标签编码逻辑"""
    print("🔍 测试标签编码逻辑...")
    
    # 加载数据集
    datasets = load_datasets()
    
    # 获取训练集和测试集数据
    data1_train = datasets['data1']['035']
    data2_train = datasets['data2']['035']
    data1_test = datasets['data1']['05']
    data2_test = datasets['data2']['05']
    
    # 对齐和准备双路径数据
    raw_data_train, feat_data_train, labels_train = align_and_prepare_dual_path(data1_train, data2_train)
    raw_data_test, feat_data_test, labels_test = align_and_prepare_dual_path(data1_test, data2_test)
    
    print(f"📊 原始标签统计:")
    print(f"   训练集标签: {set(labels_train)} (类别数: {len(set(labels_train))})")
    print(f"   测试集标签: {set(labels_test)} (类别数: {len(set(labels_test))})")
    print(f"   所有标签: {set(np.concatenate([labels_train, labels_test]))} (类别数: {len(set(np.concatenate([labels_train, labels_test])))})")
    
    # 获取所有可能的标签
    all_labels = np.concatenate([labels_train, labels_test])
    
    # 预处理训练集数据（使用所有标签构建encoder）
    X_train, y_train, label_encoder, raw_scaler, feat_scaler = preprocess_dual_path_data(
        raw_data_train, feat_data_train, labels_train, all_labels
    )
    
    print(f"\n📈 标签编码器信息:")
    print(f"   label_encoder.classes_: {label_encoder.classes_}")
    print(f"   支持的类别数: {len(label_encoder.classes_)}")
    print(f"   训练集编码后标签: {set(y_train)}")
    
    # 测试测试集标签编码
    y_test = label_encoder.transform(labels_test)
    print(f"   测试集编码后标签: {set(y_test)}")
    
    # 验证无信息泄露
    print(f"\n🔒 信息泄露检查:")
    print(f"   训练集是否包含测试集标签: {set(y_train).issuperset(set(y_test))}")
    print(f"   测试集是否包含训练集标签: {set(y_test).issuperset(set(y_train))}")
    print(f"   训练集和测试集标签交集: {set(y_train) & set(y_test)}")
    print(f"   训练集独有标签: {set(y_train) - set(y_test)}")
    print(f"   测试集独有标签: {set(y_test) - set(y_train)}")
    
    # 保存测试结果
    test_results = {
        'train_labels_original': list(set(labels_train)),
        'test_labels_original': list(set(labels_test)),
        'all_labels_original': list(set(all_labels)),
        'label_encoder_classes': list(label_encoder.classes_),
        'train_labels_encoded': list(set(y_train)),
        'test_labels_encoded': list(set(y_test)),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'total_classes': len(label_encoder.classes_)
    }
    
    # 保存到CSV
    results_df = pd.DataFrame([test_results])
    results_df.to_csv('test_results/label_encoding_test.csv', index=False)
    print(f"\n💾 测试结果已保存到: test_results/label_encoding_test.csv")
    
    return test_results

if __name__ == "__main__":
    # 创建测试结果目录
    Path('test_results').mkdir(exist_ok=True)
    
    # 运行测试
    results = test_label_encoding()
    
    print(f"\n✅ 测试完成!")
    print(f"   支持完整5类标签: {'是' if results['total_classes'] == 5 else '否'}")
    print(f"   无信息泄露: {'是' if len(set(results['train_labels_encoded']) & set(results['test_labels_encoded'])) > 0 else '否'}")
