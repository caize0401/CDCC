#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试task4_4实验设置
"""
from pathlib import Path
from common_utils import load_datasets, prepare_transformer_data, align_and_prepare_hybrid_data

def test_data_loading():
    """测试数据加载功能"""
    print("测试数据集加载...")
    
    try:
        # 加载数据集
        datasets = load_datasets(Path('.'))
        print("✓ 数据集加载成功")
        
        # 检查数据集结构
        for dataset_type in ['data1', 'data2']:
            for size in ['035', '05']:
                df = datasets[dataset_type][size]
                print(f"✓ {dataset_type}_{size}: {df.shape}")
        
        print("\n测试Transformer数据准备...")
        # 测试Transformer数据准备
        df_raw = datasets['data1']['035']
        X_raw, y = prepare_transformer_data(df_raw)
        print(f"✓ Transformer数据 - X_raw: {X_raw.shape}, y: {y.shape}")
        
        print("\n测试混合模型数据准备...")
        # 测试混合模型数据准备
        df_raw = datasets['data1']['035']
        df_feat = datasets['data2']['035']
        X_raw, X_feat, y, feat_cols = align_and_prepare_hybrid_data(df_raw, df_feat)
        print(f"✓ 混合模型数据 - X_raw: {X_raw.shape}, X_feat: {X_feat.shape}, y: {y.shape}")
        print(f"✓ 特征列数: {len(feat_cols)}")
        
        print("\n所有数据加载测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_import():
    """测试模型导入功能"""
    print("\n测试模型导入...")
    
    try:
        from common_utils import get_transformer_model, get_hybrid_model
        
        # 测试Transformer模型导入
        model_v4 = get_transformer_model("transformer_v4", input_dim=500, num_classes=4)
        print(f"✓ Transformer v4导入成功: {type(model_v4).__name__}")
        
        # 测试混合模型导入
        model_v5 = get_hybrid_model("hybrid_fusion_v5", feat_in_dim=35, num_classes=4)
        print(f"✓ HybridFusion v5导入成功: {type(model_v5).__name__}")
        
        print("\n所有模型导入测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 模型导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("Task4_4 实验设置测试")
    print("=" * 50)
    
    success = True
    
    # 测试数据加载
    if not test_data_loading():
        success = False
    
    # 测试模型导入
    if not test_model_import():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过！可以开始运行实验。")
    else:
        print("✗ 部分测试失败，请检查配置。")
    print("=" * 50)
