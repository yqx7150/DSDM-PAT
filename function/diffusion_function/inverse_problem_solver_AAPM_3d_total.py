import torch
from torch._C import device
from utilis.diffusion_utils.losses import get_optimizer
from model.diffusion.diffusion_model.ema import ExponentialMovingAverage

import numpy as np
import function.diffusion_function.controllable_generation_TV as controllable_generation_TV

from utilis.diffusion_utils.utils import restore_checkpoint, clear, batchfy, patient_wise_min_max, img_wise_min_max
from pathlib import Path
from model.diffusion.diffusion_model import utils as mutils
from model.diffusion.diffusion_model import ncsnpp
from utilis.diffusion_utils.sde_lib import VESDE
from utilis.diffusion_utils.sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import dataset.diffusion_dataset.datasets as datasets
import time
# 用于Radon变换的CT物理模型
from function.diffusion_function.ct import CT
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
#from configs.ve import AAPM_256_ncsnpp_continuous as configs
###############################################
# 配置参数
###############################################
problem = 'sparseview_CT_ADMM_TV_total'  # 问题类型：稀疏视图CT重建使用ADMM和TV正则化
config_name = 'AAPM_256_ncsnpp_continuous'  # 配置名称
sde = 'VESDE'  # 使用的随机微分方程类型
num_scales = 2000  # 尺度数量
ckpt_num = 185  # 检查点编号
N = num_scales  # 等同于num_scales

vol_name = 'L067'  # 数据卷名称
root = Path(f'./data/CT/ind/256_sorted/{vol_name}')  # 数据根目录

# 逆问题参数
Nview = 8  # 视图数量
det_spacing = 1.0  # 探测器间距
size = 256  # 图像尺寸
det_count = int((size * (2 * torch.ones(1)).sqrt()).ceil())  # 探测器数量计算
lamb = 0.04  # 正则化参数λ
rho = 10  # ADMM参数ρ
freq = 1  # 频率参数

# 根据SDE类型加载配置
if sde.lower() == 'vesde':
    ckpt_filename = f"exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"  # 检查点文件路径
    config = configs.get_config()  # 获取配置
    config.model.num_scales = N  # 设置模型尺度数
    # 创建VESDE实例
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N  # 设置SDE的尺度数
    sampling_eps = 1e-5  # 采样epsilon

# 设置预测器和校正器
predictor = ReverseDiffusionPredictor  # 反向扩散预测器
corrector = LangevinCorrector  # Langevin校正器
probability_flow = False  # 是否使用概率流
snr = 0.16  # 信噪比
n_steps = 1  # 步数

# 批次设置
batch_size = 12  # 批次大小
config.training.batch_size = batch_size  # 训练批次大小
config.eval.batch_size = batch_size  # 评估批次大小
random_seed = 0  # 随机种子

# 获取sigma序列
sigmas = mutils.get_sigmas(config)
# 数据缩放器
scaler = datasets.get_data_scaler(config)
# 数据逆缩放器
inverse_scaler = datasets.get_data_inverse_scaler(config)
# 创建分数模型
score_model = mutils.create_model(config)

# 优化器设置
optimizer = get_optimizer(config, score_model.parameters())
# 指数移动平均
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
# 状态字典
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

# 恢复检查点
state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
# 将EMA参数复制到模型
ema.copy_to(score_model.parameters())

# 指定保存生成样本的目录
save_root = Path(f'./results/{config_name}/{problem}/m{Nview}/rho{rho}/lambda{lamb}')
save_root.mkdir(parents=True, exist_ok=True)

# 创建各种类型的保存目录
irl_types = ['input', 'recon', 'label', 'BP', 'sinogram']
for t in irl_types:
    if t == 'recon':
        save_root_f = save_root / t / 'progress'  # 重建进度目录
        save_root_f.mkdir(exist_ok=True, parents=True)
    else:
        save_root_f = save_root / t  # 其他类型目录
        save_root_f.mkdir(parents=True, exist_ok=True)

# 读取所有数据文件
fname_list = os.listdir(root)
fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
print(fname_list)
all_img = []

print("Loading all data")
# 加载所有图像数据
for fname in tqdm(fname_list):
    just_name = fname.split('.')[0]  # 获取文件名（不含扩展名）
    img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))  # 加载numpy文件
    h, w = img.shape  # 获取图像尺寸
    img = img.view(1, 1, h, w)  # 重塑为4D张量
    all_img.append(img)  # 添加到列表
    # 保存标签图像
    plt.imsave(os.path.join(save_root, 'label', f'{just_name}.png'), clear(img), cmap='gray')
# 合并所有图像张量
all_img = torch.cat(all_img, dim=0)
print(f"Data loaded shape : {all_img.shape}")

# 全角度设置
angles = np.linspace(0, np.pi, 180, endpoint=False)
# 创建CT Radon变换实例
radon = CT(img_width=h, radon_view=Nview, circle=False, device=config.device)

predicted_sinogram = []  # 预测的sinogram列表
label_sinogram = []  # 标签sinogram列表
img_cache = None  # 图像缓存

# 将图像移动到指定设备
img = all_img.to(config.device)
# 获取带有ADMM和TV正则化的PC Radon采样器
pc_radon = controllable_generation_TV.get_pc_radon_ADMM_TV_vol(sde,
                                                               predictor, corrector,
                                                               inverse_scaler,
                                                               snr=snr,
                                                               n_steps=n_steps,
                                                               probability_flow=probability_flow,
                                                               continuous=config.training.continuous,
                                                               denoise=True,
                                                               radon=radon,
                                                               save_progress=True,
                                                               save_root=save_root,
                                                               final_consistency=True,
                                                               img_shape=img.shape,
                                                               lamb_1=lamb,
                                                               rho=rho)
# 通过Radon变换获取稀疏sinogram
sinogram = radon.A(img)

# 反投影
bp = radon.AT(sinogram)

# 重建图像
x = pc_radon(score_model, scaler(img), measurement=sinogram)
img_cahce = x[-1].unsqueeze(0)  # 缓存最后一幅图像

count = 0
# 保存各种结果图像
for i, recon_img in enumerate(x):
    plt.imsave(save_root / 'BP' / f'{count}.png', clear(bp[i]), cmap='gray')  # 反投影图像
    plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')  # 标签图像
    plt.imsave(save_root / 'recon' / f'{count}.png', clear(recon_img), cmap='gray')  # 重建图像

    count += 1

# 重建并保存sinogram
label_sinogram.append(radon.A_all(img))  # 全角度标签sinogram
predicted_sinogram.append(radon.A_all(x))  # 全角度预测sinogram

# 合并并保存sinogram数据
original_sinogram = torch.cat(label_sinogram, dim=0).detach().cpu().numpy()
recon_sinogram = torch.cat(predicted_sinogram, dim=0).detach().cpu().numpy()

np.save(str(save_root / 'sinogram' / f'original_{count}.npy'), original_sinogram)  # 保存原始sinogram
np.save(str(save_root / 'sinogram' / f'recon_{count}.npy'), recon_sinogram)  # 保存重建sinogram