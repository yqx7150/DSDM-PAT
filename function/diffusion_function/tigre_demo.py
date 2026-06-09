import tigre
import numpy as np
import tigre.algorithms as algs
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 初始化输出数组
out_array = np.zeros((560, 512, 512), dtype=np.float32)

# 几何参数配置
geo = tigre.geometry()
geo.mode = "parallel"
geo.DSD = 1085.6  # 源到探测器的距离
geo.DSO = 595     # 源到原点的距离

# 图像参数
geo.nVoxel = np.array([1, 512, 512])  # 体素数量
geo.sVoxel = np.array([1, 340, 340])  # 图像的总大小
geo.dVoxel = geo.sVoxel / geo.nVoxel   # 每个体素的大小

# 探测器参数
geo.nDetector = np.array([1, 512])    # 像素数量
geo.dDetector = np.array([geo.dVoxel[0], geo.dVoxel[1]])  # 每个像素的大小
geo.sDetector = geo.nDetector * geo.dDetector  # 探测器的总大小

# 偏移参数
geo.offOrigin = np.array([0, 0, 0])   # 图像相对于原点的偏移量
geo.offDetector = np.array([0, 0])    # 探测器相对于 X 轴的偏移量

# 加载完整投影数据
output_file = '/home/fangjc/data/CT/AAPM/full_1mm.npy'
sinogram = np.load(output_file)

# 原始角度设置 (720个角度)
angles_full = np.linspace(0, 2 * np.pi, 560, endpoint=False)

# 创建均匀采样掩码 (保留10%角度)
n_keep = int(len(angles_full) * 0.5)  # 保留10%
step = len(angles_full) // n_keep
mask = np.zeros(len(angles_full), dtype=bool)
mask[::step] = True  # 均匀选择10%角度为True

print(f"原始角度数: {len(angles_full)}，保留角度数: {mask.sum()}")

# 创建保存结果的目录
output_dir = os.path.join('temp', 'masked_sparse_10percent')
os.makedirs(output_dir, exist_ok=True)
print(f"所有结果将保存至: {output_dir}")

# 重建循环
for num, proj in enumerate(tqdm(sinogram, desc="重建进度")):
    # 创建掩码版投影数据（保持原始尺寸，90%区域置零）
    proj_maskeds = (proj.copy() * 255)
    proj_maskeds = np.expand_dims(proj_maskeds, axis=0)  # 在第二个维度上增加一个长度为1的轴
    proj_masked = tigre.Ax(proj_maskeds, geo, angles_full)
    proj_masked[~mask] = 0  # 非保留角度置零
    proj_masked = proj_masked[:,0,:]
    # # 使用FDK算法重建（传入完整角度，但大部分投影数据为零）
    # img_recon = algs.fdk(proj_masked, geo, angles_full)
    
    # # 保存重建结果到数组
    # out_array[num] = np.squeeze(img_recon)
    
    # 每10层保存一次投影和重建结果
    if num % 10 == 0:
        # 保存投影图像
        #plt.figure(figsize=(12, 6))
        #plt.subplot(121)
        #plt.imshow(proj, cmap='gray')
        #plt.title(f"Original Projection\nSlice {num}")
        #plt.subplot(122)
        plt.imshow(proj_masked, cmap='gray')
        #plt.title("Masked Projection (10%)")
        plt.savefig(os.path.join(output_dir, f'projection_compare_{num:04d}.png'))
        #plt.close()
        
        # # 保存重建图像
        # plt.figure(figsize=(10, 10))
        # plt.imshow(out_array[num], cmap='gray', vmin=0, vmax=2.0)
        # plt.title(f"Reconstruction from 10% Angles\nSlice {num}")
        # plt.colorbar()
        # plt.savefig(os.path.join(output_dir, f'reconstruction_{num:04d}.png'), bbox_inches='tight', dpi=300)
        # plt.close()

