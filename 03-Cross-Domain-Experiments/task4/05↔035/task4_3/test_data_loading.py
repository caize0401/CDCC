#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据加载功能
"""
from pathlib import Path
from common_utils import load_datasets

def test_data_loading():
    print("测试数据集加载...")
    
    try:
        # 加载数据集
        datasets = load_datasets(Path('.'))
        print("✓ 数据集加载成功")
        
        # 检查数据集结构
        for dataset_type in ['data1', 'data2', 'data3']:
            for size in ['035', '05']:
                df = datasets[dataset_type][size]
                print(f"✓ {dataset_type}_{size}: {df.shape}")
        
        print("\n所有数据集加载测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        return False

if __name__ == '__main__':
    test_data_loading()




