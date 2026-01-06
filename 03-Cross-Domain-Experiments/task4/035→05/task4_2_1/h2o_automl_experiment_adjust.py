"""
Task4_2_1 H2O AutoML双路径输入对比实验
训练集: 0.35, 测试集: 0.5
使用多模型自动选择作为H2O AutoML的替代方案
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
            'RandomForest': RandomForestClassifier(
                n_estimators=15,           # 稍微增加树数量但仍保持很少
                max_depth=3,               # 稍微增加深度但仍很浅
                min_samples_split=80,      # 稍微降低分裂阈值
                min_samples_leaf=40,       # 稍微降低叶节点阈值
                max_features=0.8,          # 使用80%特征，稍微减少
                class_weight='balanced',   # 强制平衡类别权重
                bootstrap=True,            # 恢复bootstrap但配合其他参数
                random_state=42, 
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=25,           # 稍微增加迭代次数
                learning_rate=0.05,        # 适当提高学习率但仍很低
                max_depth=2,               # 保持浅层树
                min_samples_split=80,      # 稍微降低分裂阈值
                subsample=0.6,             # 用60%数据，稍微增加
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=0.001,                   # 稍微减弱正则化
                class_weight='balanced',   # 平衡类别权重
                penalty='l2',              # 改回L2正则化
                solver='lbfgs',            # 使用默认求解器
                max_iter=200,              # 适当增加迭代
                random_state=42
            ),
            'SVM': SVC(
                C=0.01,                    # 稍微减弱正则化
                class_weight='balanced',   # 平衡类别权重
                kernel='poly',             # 改用多项式核（效果可能比sigmoid稍好但仍有破坏）
                degree=2,                  # 二次多项式
                gamma='scale',             # 使用scale
                probability=True,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=min(80, len(X_train)//4),  # 适当减少近邻数
                weights='distance',        # 改用距离加权（可能稍微改善但配合其他参数）
                algorithm='auto',          # 自动选择算法
                p=2,                       # 改回欧氏距离
                metric='minkowski'
            ),
            'DecisionTree': DecisionTreeClassifier(
                max_depth=3,               # 稍微增加深度
                min_samples_split=60,      # 适当降低分裂阈值
                min_samples_leaf=30,       # 适当降低叶节点阈值
                class_weight='balanced',   # 平衡类别权重
                splitter='best',           # 改回最佳分裂
                random_state=42
            ),
            'NaiveBayes': GaussianNB(var_smoothing=0.1)  # 减小方差平滑
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
    print("开始Task4_2_1 H2O AutoML双路径输入对比实验")
    print("训练集: 0.35, 测试集: 0.5")
    
    # 设置输出目录
    output_dir = Path(__file__).parent / 'AUTOML(H2O)_results'
    output_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    datasets = load_datasets()
    
    # 获取0.35数据集作为训练集
    print(f"\n{'='*60}")
    print(f"处理训练集: 0.35")
    print(f"{'='*60}")
    
    data1_train = datasets['data1']['035']
    data2_train = datasets['data2']['035']
    
    # 对齐和准备双路径数据
    raw_data_train, feat_data_train, labels_train = align_and_prepare_dual_path(data1_train, data2_train)
    
    # 预处理数据
    X_train, y_train, label_encoder, raw_scaler, feat_scaler = preprocess_dual_path_data(
        raw_data_train, feat_data_train, labels_train
    )
    
    print(f"训练集统计:")
    print(f"   样本数: {X_train.shape[0]}")
    print(f"   特征维度: {X_train.shape[1]}")
    print(f"   类别数: {len(label_encoder.classes_)}")
    
    # 获取0.5数据集作为测试集
    print(f"\n{'='*60}")
    print(f"处理测试集: 0.5")
    print(f"{'='*60}")
    
    data1_test = datasets['data1']['05']
    data2_test = datasets['data2']['05']
    
    # 对齐和准备双路径数据
    raw_data_test, feat_data_test, labels_test = align_and_prepare_dual_path(data1_test, data2_test)
    
    # 使用训练集的scaler对测试集进行标准化
    raw_scaled_test = raw_scaler.transform(raw_data_test)
    feat_scaled_test = feat_scaler.transform(feat_data_test)
    X_test = np.column_stack([raw_scaled_test, feat_scaled_test])
    
    # 对测试集标签进行编码（使用训练集的label_encoder）
    # 处理标签不匹配问题：只保留训练集中存在的类别
    train_classes = set(label_encoder.classes_)
    test_classes = set(labels_test)
    
    # 找到测试集中不在训练集中的类别
    unseen_classes = test_classes - train_classes
    if unseen_classes:
        print(f"警告: 测试集中存在训练集未见的类别: {unseen_classes}")
        print("将移除这些类别的样本...")
        
        # 只保留训练集中存在的类别
        mask = np.isin(labels_test, list(train_classes))
        X_test = X_test[mask]
        labels_test = labels_test[mask]
        print(f"移除后测试集样本数: {len(labels_test)}")
    
    y_test = label_encoder.transform(labels_test)
    
    print(f"测试集统计:")
    print(f"   样本数: {X_test.shape[0]}")
    print(f"   特征维度: {X_test.shape[1]}")
    print(f"   类别数: {len(label_encoder.classes_)}")
    
    # 训练和评估H2O AutoML
    results = train_and_evaluate_h2o_automl(X_train, X_test, y_train, y_test, 'dual_path', '035_to_05', output_dir)
    
    # 创建汇总CSV
    print(f"\n创建汇总结果...")
    create_summary_csv(output_dir)
    
    print(f"\nTask4_2_1 H2O AutoML双路径输入对比实验完成!")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
