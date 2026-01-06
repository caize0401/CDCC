"""
运行所有跨域实验的主脚本
包括：
1. 0.5数据集训练，0.35数据集测试
2. 0.35数据集训练，0.5数据集测试
使用task3的两个混合模型
"""
import os
import sys
from pathlib import Path
import subprocess
import time


def run_experiment(script_name: str, description: str):
    """运行单个实验脚本"""
    print(f"\n{'='*60}")
    print(f"开始运行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        print("标准输出:")
        print(result.stdout)
        
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"实验失败: {e}")
        print("标准输出:")
        print(e.stdout)
        print("标准错误:")
        print(e.stderr)
        return False
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n实验完成，耗时: {duration:.2f}秒")
    
    return True


def main():
    """主函数"""
    print("开始运行task4_3跨域实验")
    print("使用task3的两个混合模型进行跨域测试")
    
    # 确保在正确的目录
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    
    experiments = [
        ("run_05_to_035_experiment.py", "0.5数据集训练 → 0.35数据集测试"),
        ("run_035_to_05_experiment.py", "0.35数据集训练 → 0.5数据集测试"),
    ]
    
    success_count = 0
    total_start_time = time.time()
    
    for script_name, description in experiments:
        if run_experiment(script_name, description):
            success_count += 1
        else:
            print(f"实验 {description} 失败，继续下一个实验...")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("所有实验完成")
    print(f"成功: {success_count}/{len(experiments)}")
    print(f"总耗时: {total_duration:.2f}秒")
    print(f"{'='*60}")
    
    # 汇总结果
    print("\n结果文件位置:")
    for exp_dir in ['05_to_035', '035_to_05']:
        exp_path = script_dir / exp_dir
        if exp_path.exists():
            print(f"- {exp_dir}/: {exp_path}")
            summary_file = exp_path / 'summary.csv'
            if summary_file.exists():
                print(f"  汇总结果: {summary_file}")


if __name__ == '__main__':
    main()




