#!/usr/bin/env python3
"""检查类别分布情况"""
import pickle
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent
features_dir = base_dir / 'features'

print("=" * 60)
print("类别分布检查")
print("=" * 60)

# 检查 0.5 → 0.35 方向
print("\n方向: 0.5 → 0.35")
base_file = features_dir / 'base_mlp_features_05_to_035.pkl'
if base_file.exists():
    data = pickle.load(open(base_file, 'rb'))
    
    print("\n源域(0.5)类别分布:")
    unique, counts = np.unique(data['source_labels'], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  类别 {int(u)}: {c} 个样本")
    
    print("\n目标域(0.35)类别分布:")
    unique, counts = np.unique(data['target_labels'], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  类别 {int(u)}: {c} 个样本")
    
    source_classes = set(np.unique(data['source_labels']))
    target_classes = set(np.unique(data['target_labels']))
    
    print(f"\n缺少的类别: {source_classes - target_classes}")
    print(f"共同存在的类别: {source_classes & target_classes}")
    
# 检查 0.35 → 0.5 方向
print("\n" + "=" * 60)
print("方向: 0.35 → 0.5")
base_file = features_dir / 'base_mlp_features_035_to_05.pkl'
if base_file.exists():
    data = pickle.load(open(base_file, 'rb'))
    
    print("\n源域(0.35)类别分布:")
    unique, counts = np.unique(data['source_labels'], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  类别 {int(u)}: {c} 个样本")
    
    print("\n目标域(0.5)类别分布:")
    unique, counts = np.unique(data['target_labels'], return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  类别 {int(u)}: {c} 个样本")
    
    source_classes = set(np.unique(data['source_labels']))
    target_classes = set(np.unique(data['target_labels']))
    
    print(f"\n缺少的类别: {source_classes - target_classes}")
    print(f"共同存在的类别: {source_classes & target_classes}")

print("\n" + "=" * 60)




