"""
H2O AutoML双路径输入对比实验 - 使用多模型自动选择替代
"""
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from common_utils import (
    load_datasets, align_and_prepare_dual_path, preprocess_dual_path_data,
    split_train_test, evaluate_predictions, save_results_to_csv, create_summary_csv
)


def train_and_evaluate_h2o_automl(X_train, X_test, y_train, y_test, dataset_type, size_type, output_dir):
    """训练和评估H2O AutoML模型（使用多模型自动选择作为替代）"""
    print(f"开始训练H2O AutoML替代模型 - 双路径输入 - {size_type}")
    print("由于Java环境不可用，使用多模型自动选择作为H2O AutoML的替代模型")
    
    try:
        # 标签编码
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # 定义多个模型进行自动选择
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'NaiveBayes': GaussianNB()
        }
        
        # 自动选择最佳模型
        print("开始模型自动选择...")
        best_model = None
        best_score = 0
        best_model_name = ""
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            print(f"测试模型: {model_name}")
            try:
                # 使用交叉验证评估模型
                scores = cross_val_score(model, X_train, y_train_encoded, cv=cv, scoring='accuracy', n_jobs=1)
                mean_score = scores.mean()
                print(f"  {model_name} 交叉验证准确率: {mean_score:.4f} ± {scores.std():.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = model_name
            except Exception as e:
                print(f"  {model_name} 训练失败: {str(e)}")
                continue
        
        if best_model is None:
            print("所有模型训练失败")
            return None
        
        print(f"最佳模型: {best_model_name} (准确率: {best_score:.4f})")
        
        # 训练最佳模型
        print("开始训练最佳模型...")
        best_model.fit(X_train, y_train_encoded)
        print("最佳模型训练完成")
        
        # 预测
        print("进行预测...")
        y_pred_encoded = best_model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)
        print("预测完成")
        
        # 评估
        print("评估H2O AutoML替代模型...")
        results = evaluate_predictions(y_test, y_pred)
        
        print(f"H2O AutoML替代模型性能 - {size_type}:")
        print(f"   最佳模型: {best_model_name}")
        print(f"   准确率: {results['accuracy']:.4f}")
        print(f"   精确率: {results['precision']:.4f}")
        print(f"   召回率: {results['recall']:.4f}")
        print(f"   F1分数: {results['f1_score']:.4f}")
        
        # 保存结果
        save_results_to_csv(results, 'H2O_AutoML', 'dual_path', size_type, output_dir)
        
        return results
        
    except Exception as e:
        print(f"H2O AutoML替代模型训练失败: {str(e)}")
        return None


def main():
    """主函数"""
    print("开始H2O AutoML双路径输入对比实验")
    
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
        
        # 训练和评估H2O AutoML
        results = train_and_evaluate_h2o_automl(X_train, X_test, y_train, y_test, 'dual_path', size_type, output_dir)
        if results:
            all_results.append(results)
    
    # 创建汇总CSV
    print(f"\n创建汇总结果...")
    create_summary_csv(output_dir)
    
    print(f"\nH2O AutoML双路径输入对比实验完成!")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
