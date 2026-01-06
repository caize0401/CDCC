"""
Task4_2_1 双路径输入对比实验主运行脚本
训练集: 0.35, 测试集: 0.5
"""
import os
import sys
import time
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

import subprocess


def run_experiment(script_name, model_name):
    """运行单个实验"""
    print(f"{'='*80}")
    print(f"开始运行 {model_name} 双路径输入对比实验")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # 运行实验脚本
        result = subprocess.run([sys.executable, script_name], 
                               capture_output=True, text=True, cwd=current_dir)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} 实验成功")
            print(f"⏱️ 耗时: {duration:.2f}秒")
        else:
            print(f"❌ {model_name} 实验失败")
            print(f"📊 错误信息:")
            print(result.stderr)
        
        return result.returncode == 0, duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {model_name} 实验失败")
        print(f"📊 错误信息: {str(e)}")
        return False, duration


def main():
    """主函数"""
    print("🎯 Task4_2_1 双路径输入对比实验")
    print("="*80)
    print("📋 实验计划:")
    print("   训练集: 0.35")
    print("   测试集: 0.5")
    print("   输入: 双路径 (原始曲线 + 完整特征)")
    print("   模型: MLP, Random Forest, XGBoost, H2O AutoML")
    print("="*80)
    
    # 定义实验列表
    experiments = [
        ("mlp_experiment.py", "MLP"),
        ("random_forest_experiment.py", "Random Forest"),
        ("xgboost_experiment.py", "XGBoost"),
        ("h2o_automl_experiment.py", "H2O AutoML")
    ]
    
    # 运行所有实验
    results = {}
    total_start_time = time.time()
    
    for script_name, model_name in experiments:
        print(f"\n🚀 开始运行 {model_name} 双路径输入对比实验")
        print(f"{'='*80}")
        
        success, duration = run_experiment(script_name, model_name)
        results[model_name] = {
            'success': success,
            'duration': duration
        }
        
        print(f"\n{'-'*80}")
    
    # 计算总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 输出结果总结
    print(f"\n🎯 所有实验完成 (总耗时: {total_duration:.2f}秒)")
    print("="*80)
    print("📊 实验结果总结:")
    
    success_count = 0
    for model_name, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"   {model_name}: {status}")
        if result['success']:
            success_count += 1
    
    print(f"\n📈 成功率: {success_count}/{len(experiments)} ({success_count/len(experiments)*100:.1f}%)")
    
    # 检查结果文件
    results_dir = Path(__file__).parent / 'results'
    if results_dir.exists():
        csv_files = list(results_dir.glob("*.csv"))
        print(f"\n📁 生成的结果文件: {len(csv_files)} 个")
        for csv_file in csv_files:
            print(f"   - {csv_file.name}")
        
        # 检查汇总文件
        summary_file = results_dir / 'summary.csv'
        if summary_file.exists():
            print(f"\n📊 汇总文件已生成: {summary_file}")
    
    print(f"\n🎉 Task4_2_1 双路径输入对比实验全部完成!")


if __name__ == "__main__":
    main()
