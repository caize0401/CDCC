"""
Task4_2 MLP 独立实验脚本
训练集: 0.35, 测试集: 0.5
支持三种数据集类型: data1, data2, data3
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from common_utils import load_datasets, preprocess, fit_scaler_on_union, transform_split, encode_labels_union, compute_metrics


def create_mlp_model():
    """创建MLP模型"""
    return MLPClassifier(
        hidden_layer_sizes=(256, 128), 
        activation='relu', 
        solver='adam',
        alpha=1e-5, 
        max_iter=500, 
        random_state=42
    )


def train_and_evaluate_mlp(X_train, X_test, y_train, y_test, dataset_type, output_dir):
    """训练和评估MLP模型"""
    print(f"开始训练MLP模型 - {dataset_type}")
    
    # 创建模型
    print("创建MLP模型...")
    mlp = create_mlp_model()
    
    # 训练模型
    print("开始训练MLP模型...")
    mlp.fit(X_train, y_train)
    print("MLP模型训练完成")
    
    # 预测
    print("进行MLP预测...")
    y_pred = mlp.predict(X_test)
    print("预测完成")
    
    # 评估
    print("评估MLP模型...")
    metrics = compute_metrics(y_test, y_pred)
    
    print(f"MLP模型性能 - {dataset_type}:")
    print(f"   准确率: {metrics['accuracy']:.4f}")
    print(f"   精确率: {metrics['precision']:.4f}")
    print(f"   召回率: {metrics['recall']:.4f}")
    print(f"   F1分数: {metrics['f1_score']:.4f}")
    
    # 创建结果目录
    model_dir = output_dir / f'MLP_{dataset_type}'
    model_dir.mkdir(exist_ok=True)
    
    # 保存指标
    pd.DataFrame([metrics]).to_csv(model_dir / 'metrics.csv', index=False, encoding='utf-8-sig')
    print(f"指标已保存到: {model_dir / 'metrics.csv'}")
    
    # 保存混淆矩阵
    labels_full = list(range(len(np.unique(np.concatenate([y_train, y_test])))))
    cm = confusion_matrix(y_test, y_pred, labels=labels_full)
    cm_df = pd.DataFrame(
        cm, 
        index=[f'True_{i}' for i in range(cm.shape[0])], 
        columns=[f'Pred_{i}' for i in range(cm.shape[1])]
    )
    cm_df.to_csv(model_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
    print(f"混淆矩阵已保存到: {model_dir / 'confusion_matrix.csv'}")
    
    return metrics


def run_mlp_experiment():
    """运行MLP实验"""
    print("开始Task4_2 MLP独立实验")
    print("训练集: 0.35, 测试集: 0.5")
    print("="*60)
    
    # 设置输出目录
    output_dir = Path(__file__).parent / 'mlp_results'
    output_dir.mkdir(exist_ok=True)
    print(f"结果将保存到: {output_dir}")
    
    # 加载数据集
    base_dir = Path(__file__).parent
    datasets = load_datasets(base_dir)
    
    # 存储所有结果
    all_results = []
    
    # 对三种数据集类型进行实验
    for dataset_type in ['data1', 'data2', 'data3']:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_type}")
        print(f"{'='*60}")
        
        # 获取训练集和测试集
        df_train = datasets[dataset_type]['035']
        df_test = datasets[dataset_type]['05']
        
        # 预处理数据
        X_tr_raw, y_tr_raw = preprocess(df_train, dataset_type)
        X_te_raw, y_te_raw = preprocess(df_test, dataset_type)
        
        # 标准化
        scaler = fit_scaler_on_union(X_tr_raw, X_te_raw)
        X_train = transform_split(scaler, X_tr_raw)
        X_test = transform_split(scaler, X_te_raw)
        
        # 标签编码
        y_train, y_test, le = encode_labels_union(y_tr_raw, y_te_raw)
        
        print(f"训练集统计:")
        print(f"   样本数: {X_train.shape[0]}")
        print(f"   特征维度: {X_train.shape[1]}")
        print(f"   类别数: {len(le.classes_)}")
        
        print(f"测试集统计:")
        print(f"   样本数: {X_test.shape[0]}")
        print(f"   特征维度: {X_test.shape[1]}")
        
        # 训练和评估MLP
        metrics = train_and_evaluate_mlp(X_train, X_test, y_train, y_test, dataset_type, output_dir)
        
        # 记录结果
        result = {
            'model': 'MLP',
            'dataset': dataset_type,
            **metrics
        }
        all_results.append(result)
    
    # 创建汇总文件
    print(f"\n创建汇总结果...")
    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"汇总结果已保存到: {summary_path}")
    
    # 显示汇总结果
    print(f"\n{'='*60}")
    print("实验结果汇总:")
    print(f"{'='*60}")
    for _, row in summary_df.iterrows():
        print(f"{row['dataset']}: 准确率={row['accuracy']:.4f}, F1={row['f1_score']:.4f}")
    
    print(f"\nTask4_2 MLP独立实验完成!")
    print(f"所有结果保存在: {output_dir}")


if __name__ == "__main__":
    # 确保在正确的目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_mlp_experiment()
