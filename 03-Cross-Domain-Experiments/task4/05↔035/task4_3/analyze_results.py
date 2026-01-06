#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析跨域实验结果
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_cross_domain_results():
    """分析跨域实验结果"""
    
    # 读取两个方向的实验结果
    results_05_to_035 = pd.read_csv('05_to_035/summary.csv')
    results_035_to_05 = pd.read_csv('035_to_05/summary.csv')
    
    # 合并结果
    all_results = pd.concat([results_05_to_035, results_035_to_05], ignore_index=True)
    
    print("=== 跨域实验结果分析 ===")
    print("\n1. 整体结果汇总:")
    print(all_results.to_string(index=False))
    
    print("\n2. 按模型分组的性能对比:")
    model_comparison = all_results.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std']
    }).round(4)
    print(model_comparison)
    
    print("\n3. 按方向分组的性能对比:")
    direction_comparison = all_results.groupby('direction').agg({
        'accuracy': ['mean', 'std'],
        'f1_score': ['mean', 'std']
    }).round(4)
    print(direction_comparison)
    
    print("\n4. 关键发现:")
    
    # 找出最佳性能
    best_acc = all_results.loc[all_results['accuracy'].idxmax()]
    best_f1 = all_results.loc[all_results['f1_score'].idxmax()]
    
    print(f"- 最佳准确率: {best_acc['model']} ({best_acc['direction']}) = {best_acc['accuracy']:.4f}")
    print(f"- 最佳F1分数: {best_f1['model']} ({best_f1['direction']}) = {best_f1['f1_score']:.4f}")
    
    # 分析跨域方向的影响
    print(f"\n- 0.35→0.5方向平均准确率: {results_035_to_05['accuracy'].mean():.4f}")
    print(f"- 0.5→0.35方向平均准确率: {results_05_to_035['accuracy'].mean():.4f}")
    
    if results_035_to_05['accuracy'].mean() > results_05_to_035['accuracy'].mean():
        print("- 0.35→0.5方向的跨域性能更好（小数据集→大数据集）")
    else:
        print("- 0.5→0.35方向的跨域性能更好（大数据集→小数据集）")
    
    # 分析模型差异
    hybrid_fusion_mean = all_results[all_results['model'] == 'HybridFusion']['accuracy'].mean()
    advanced_fusion_mean = all_results[all_results['model'] == 'AdvancedHybridFusion']['accuracy'].mean()
    
    print(f"\n- HybridFusion平均准确率: {hybrid_fusion_mean:.4f}")
    print(f"- AdvancedHybridFusion平均准确率: {advanced_fusion_mean:.4f}")
    
    if advanced_fusion_mean > hybrid_fusion_mean:
        print("- AdvancedHybridFusion在跨域任务中表现更好")
    else:
        print("- HybridFusion在跨域任务中表现更好")
    
    # 保存综合结果
    all_results.to_csv('cross_domain_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n综合结果已保存到: cross_domain_summary.csv")
    
    return all_results

def plot_results(results_df):
    """绘制结果对比图"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 准确率对比
    ax1 = axes[0, 0]
    results_pivot = results_df.pivot(index='direction', columns='model', values='accuracy')
    results_pivot.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('跨域准确率对比')
    ax1.set_ylabel('准确率')
    ax1.legend(title='模型')
    
    # 2. F1分数对比
    ax2 = axes[0, 1]
    f1_pivot = results_df.pivot(index='direction', columns='model', values='f1_score')
    f1_pivot.plot(kind='bar', ax=ax2, rot=45)
    ax2.set_title('跨域F1分数对比')
    ax2.set_ylabel('F1分数')
    ax2.legend(title='模型')
    
    # 3. 模型性能雷达图
    ax3 = axes[1, 0]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_means = results_df.groupby('model')[metrics].mean()
    
    x = range(len(metrics))
    width = 0.35
    
    ax3.bar([i - width/2 for i in x], model_means.loc['HybridFusion'], 
           width, label='HybridFusion', alpha=0.7)
    ax3.bar([i + width/2 for i in x], model_means.loc['AdvancedHybridFusion'], 
           width, label='AdvancedHybridFusion', alpha=0.7)
    
    ax3.set_xlabel('指标')
    ax3.set_ylabel('分数')
    ax3.set_title('模型性能对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=45)
    ax3.legend()
    
    # 4. 跨域方向性能对比
    ax4 = axes[1, 1]
    direction_means = results_df.groupby('direction')[metrics].mean()
    
    ax4.bar([i - width/2 for i in x], direction_means.loc['035_to_05'], 
           width, label='0.35→0.5', alpha=0.7)
    ax4.bar([i + width/2 for i in x], direction_means.loc['05_to_035'], 
           width, label='0.5→0.35', alpha=0.7)
    
    ax4.set_xlabel('指标')
    ax4.set_ylabel('分数')
    ax4.set_title('跨域方向性能对比')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('cross_domain_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("结果对比图已保存到: cross_domain_analysis.png")

if __name__ == '__main__':
    results = analyze_cross_domain_results()
    
    try:
        plot_results(results)
    except Exception as e:
        print(f"绘图失败: {e}")
        print("可能需要安装matplotlib: pip install matplotlib")




