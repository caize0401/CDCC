"""
调试标签分布，检查实际的标签情况
"""
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import pandas as pd
import numpy as np
from common_utils import load_datasets, align_and_prepare_dual_path

def debug_labels():
    """调试标签分布"""
    print("🔍 调试标签分布...")
    
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
    
    print(f"📊 详细标签统计:")
    print(f"   训练集(0.35)标签: {sorted(set(labels_train))}")
    print(f"   测试集(0.5)标签: {sorted(set(labels_test))}")
    
    # 统计每个标签的数量
    from collections import Counter
    train_counts = Counter(labels_train)
    test_counts = Counter(labels_test)
    
    print(f"\n📈 标签数量统计:")
    print(f"   训练集标签分布: {dict(train_counts)}")
    print(f"   测试集标签分布: {dict(test_counts)}")
    
    # 检查交集
    train_set = set(labels_train)
    test_set = set(labels_test)
    intersection = train_set & test_set
    train_only = train_set - test_set
    test_only = test_set - train_set
    
    print(f"\n🔍 标签交集分析:")
    print(f"   共同标签: {sorted(intersection)}")
    print(f"   训练集独有: {sorted(train_only)}")
    print(f"   测试集独有: {sorted(test_only)}")
    
    # 检查是否真的需要5类
    all_labels = np.concatenate([labels_train, labels_test])
    print(f"\n🎯 所有标签: {sorted(set(all_labels))}")
    print(f"   总类别数: {len(set(all_labels))}")
    
    return {
        'train_labels': sorted(set(labels_train)),
        'test_labels': sorted(set(labels_test)),
        'all_labels': sorted(set(all_labels)),
        'train_counts': dict(train_counts),
        'test_counts': dict(test_counts)
    }

if __name__ == "__main__":
    results = debug_labels()
