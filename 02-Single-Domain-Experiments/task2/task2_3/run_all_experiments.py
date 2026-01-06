"""
Task2_3 双路径输入对比实验主运行脚本
运行MLP, Random Forest, XGBoost, H2O AutoML四个模型的对比实验
"""
import os
import sys
import subprocess
from pathlib import Path
import time

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))


def run_experiment(script_name, model_name):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"🚀 开始运行 {model_name} 双路径输入对比实验")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # 运行实验脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=current_dir)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {model_name} 实验完成 (耗时: {duration:.2f}秒)")
            print("📊 输出:")
            print(result.stdout)
        else:
            print(f"❌ {model_name} 实验失败")
            print("📊 错误信息:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ {model_name} 实验出错: {e}")
        return False


def main():
    """主函数"""
    print("🎯 Task2_3 双路径输入对比实验")
    print("="*80)
    print("📋 实验计划:")
    print("   1. MLP - 双路径输入")
    print("   2. Random Forest - 双路径输入") 
    print("   3. XGBoost - 双路径输入")
    print("   4. H2O AutoML - 双路径输入")
    print("="*80)
    
    # 实验配置
    experiments = [
        ("mlp_experiment.py", "MLP"),
        ("random_forest_experiment.py", "Random Forest"),
        ("xgboost_experiment.py", "XGBoost"),
        ("h2o_automl_experiment.py", "H2O AutoML")
    ]
    
    # 记录实验结果
    results = {}
    total_start_time = time.time()
    
    # 运行所有实验
    for script_name, model_name in experiments:
        success = run_experiment(script_name, model_name)
        results[model_name] = success
        
        if success:
            print(f"✅ {model_name} 实验成功完成")
        else:
            print(f"❌ {model_name} 实验失败")
        
        print("\n" + "-"*80)
    
    # 总结结果
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n🎯 所有实验完成 (总耗时: {total_duration:.2f}秒)")
    print("="*80)
    print("📊 实验结果总结:")
    
    success_count = 0
    for model_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {model_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n📈 成功率: {success_count}/{len(experiments)} ({success_count/len(experiments)*100:.1f}%)")
    
    # 检查结果文件
    results_dir = current_dir / 'results'
    if results_dir.exists():
        csv_files = list(results_dir.glob("*.csv"))
        print(f"\n📁 生成的结果文件: {len(csv_files)} 个")
        for csv_file in csv_files:
            print(f"   - {csv_file.name}")
        
        # 检查汇总文件
        summary_file = results_dir / 'summary.csv'
        if summary_file.exists():
            print(f"\n📊 汇总文件已生成: {summary_file}")
        else:
            print(f"\n⚠️ 汇总文件未找到")
    else:
        print(f"\n⚠️ 结果目录不存在")
    
    print(f"\n🎉 Task2_3 双路径输入对比实验全部完成!")
    print(f"📁 结果保存在: {results_dir}")


if __name__ == "__main__":
    main()
