# UACAN 配置：源域/目标域及类别选择
# 可通过命令行覆盖

SOURCE_DOMAIN = "05"
TARGET_DOMAIN = "035"

# 类别 1-5: OK, one missing strand, two missing strands, three missing strands, crimped insulation
SOURCE_CLASSES = [1, 2, 3, 5]
TARGET_CLASSES = [1, 3, 5]

# 动态调度与能量
LAMBDA_0 = 1.0          # 对齐损失基准系数
ENERGY_TEMP = 1.0       # 能量温度 T
ENERGY_THRESHOLD = 0.0  # 能量阈值 δ，E(x)>δ 判为未知（可训练中调）
ENERGY_MARGIN = 1.0     # 拒绝约束边界 m（高能量样本希望 E(x)≥m）
LAMBDA_UNK = 0.1        # 未知拒绝损失权重
