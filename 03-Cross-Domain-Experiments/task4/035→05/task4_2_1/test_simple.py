"""
简单的测试脚本
"""
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

print("开始测试...")

try:
    # 测试导入
    from common_utils import load_datasets
    print("✅ common_utils导入成功")
    
    # 测试数据加载
    datasets = load_datasets()
    print("✅ 数据加载成功")
    
    # 检查数据结构
    print(f"data1 keys: {list(datasets['data1'].keys())}")
    print(f"data2 keys: {list(datasets['data2'].keys())}")
    
    print("✅ 测试完成")
    
except Exception as e:
    print(f"❌ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
