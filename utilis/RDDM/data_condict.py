import numpy as np

# 假设原始数据已经加载为original_data
# original_data.shape = (512, 256, 256)
original_data = np.load('label.npy')
# 1. 将512个样本均匀分为8组，每组64个
original_data = original_data[:256]
groups = []
for i in range(4):
    # 从i开始，步长为8，取64个样本
    group = original_data[i::4][:64]  # 确保每组只取64个
    groups.append(group)

# 检查分组是否正确
print(f"Number of groups: {len(groups)}")
print(f"Each group shape: {groups[0].shape}")  # 应该为(64, 256, 256)

# 2. 将8组沿最后一个维度拼接 (64, 256, 2048)
concatenated = np.concatenate(groups, axis=-1)

# 3. 将2048维度移到最前面 (2048, 64, 256)
result = np.moveaxis(concatenated, -1, 0)

# 检查最终形状
print(f"Final shape: {result.shape}")  # 应该为(2048, 64, 256)

# 保存为新的npy文件
np.save('label1.npy', result)