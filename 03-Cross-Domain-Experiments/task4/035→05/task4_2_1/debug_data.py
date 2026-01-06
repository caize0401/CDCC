"""
调试数据结构
"""
import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from common_utils import load_datasets

# 加载数据
datasets = load_datasets()

# 检查data1结构
print("=== Data1 结构 ===")
data1_035 = datasets['data1']['035']
print(f"data1_035 type: {type(data1_035)}")
if hasattr(data1_035, 'columns'):
    print(f"data1_035 columns: {list(data1_035.columns)}")
    print(f"data1_035 shape: {data1_035.shape}")
else:
    print(f"data1_035 keys: {list(data1_035.keys()) if hasattr(data1_035, 'keys') else 'No keys'}")

# 检查data2结构
print("\n=== Data2 结构 ===")
data2_035 = datasets['data2']['035']
print(f"data2_035 type: {type(data2_035)}")
if hasattr(data2_035, 'columns'):
    print(f"data2_035 columns: {list(data2_035.columns)}")
    print(f"data2_035 shape: {data2_035.shape}")
else:
    print(f"data2_035 keys: {list(data2_035.keys()) if hasattr(data2_035, 'keys') else 'No keys'}")

# 检查Force_curve_RoI
print("\n=== Force_curve_RoI 检查 ===")
if hasattr(data1_035, 'columns') and 'Force_curve_RoI' in data1_035.columns:
    print("Force_curve_RoI 存在")
    print(f"Force_curve_RoI type: {type(data1_035['Force_curve_RoI'].iloc[0])}")
    print(f"Force_curve_RoI shape: {data1_035['Force_curve_RoI'].iloc[0].shape if hasattr(data1_035['Force_curve_RoI'].iloc[0], 'shape') else 'No shape'}")
else:
    print("Force_curve_RoI 不存在")
