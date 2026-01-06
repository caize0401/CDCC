#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据集结构
"""
from pathlib import Path
from common_utils import load_datasets

def inspect_datasets():
    print("检查数据集结构...")
    
    datasets = load_datasets(Path('.'))
    
    for dataset_type in ['data1', 'data2', 'data3']:
        print(f"\n=== {dataset_type} ===")
        for size in ['035', '05']:
            df = datasets[dataset_type][size]
            print(f"\n{dataset_type}_{size}:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample data:")
            print(df.head(2))

if __name__ == '__main__':
    inspect_datasets()




