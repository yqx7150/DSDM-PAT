import os
import os.path as osp
import time
import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

import matplotlib.pyplot as plt
import imageio.v2 as iio

# NeRF模块
from model.Nerf.Nerf_network import get_network
from model.Nerf.Nerf_encoder import get_encoder
from dataset.Nerf_dataset import TIGREDataset as Dataset
from utilis.Nerf.Nerf_render import render, run_network
from utilis.Nerf.Nerf_utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image, get_mse,gen_log, time2file_name

# 扩散模型模块
from model.diffusion.diffusion_model import ddpm, ncsnv2, ncsnpp, unet
from model.diffusion.diffusion_model import utils as mutils
from model.diffusion.diffusion_model.ema import ExponentialMovingAverage
import utilis.diffusion_utils.losses as losses
from utilis.diffusion_utils.sampling import ReverseDiffusionPredictor, LangevinCorrector


# 数据集与工具函数
import dataset.diffusion_dataset.datasets as datasets
from utilis.diffusion_utils.utils import restore_checkpoint, clear, initSDE
from function.diffusion_function.ct import CT
import function.diffusion_function.controllable_generation_TV as controllable_generation_TV
from configloading import load_config




def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    parser.add_argument("--nerf_category", default=f"AAPM", help="category of the tested scene")
    parser.add_argument("--config1", default=f"config/Diffusion_config/AAPM_256_DR_256x2_1000_val_mid.yaml")
    parser.add_argument("--config2", default=f"config/Nerf_config/Lineformer/AAPM_256_D256X2_20view.yaml")
    return parser

parser = config_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id



def eval_nerf(eval_dset, model, model_fine, cfg,root_path):
    resdir = osp.join(root_path, "nerf_save")
    projs = eval_dset.projs                 # [256, 256] -> [50, 256, 256]
    rays = eval_dset.rays.reshape(-1, 8)    # [65536,8]  -> [3276800, 8]

    N, H, W = projs.shape
    projs_pred = []
    n_rays = cfg["train"]["n_rays"]
    netchunk = cfg["render"]["netchunk"]
    print("Start rendering projection")
    proj_start_time = time.time()
    for i in tqdm(range(0, rays.shape[0], n_rays)):     
        projs_pred.append(render(rays[i:i+n_rays], model, model_fine, **cfg["render"])["acc"])
    proj_end_time = time.time()
    print(f"Time of rendering projection: {proj_end_time - proj_start_time} s")

    projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W) 

    image = eval_dset.image
    print("Start reconstructing CT")
    ct_start_time = time.time()
    image_pred = run_network(eval_dset.voxels, model_fine if model_fine is not None else model, netchunk)
    ct_end_time = time.time()
    print(f"Time of reconstructing CT: {ct_end_time - ct_start_time} s")

    image_pred = image_pred.squeeze()
    
    logger.info("Evaluating performance...")
    loss = {
        "proj_psnr": get_psnr(projs_pred, projs),
        "proj_ssim": get_ssim(projs_pred, projs),
        "psnr_3d": get_psnr_3d(image_pred, image),
        "ssim_3d": get_ssim_3d(image_pred, image),
    }
    logger.info(loss)

    # 保存各种视图

    proj_pred_dir = osp.join(resdir, "proj_pred")
    proj_gt_dir = osp.join(resdir, "proj_gt")

    ct_pred_dir_H = osp.join(resdir, "CT", "H", "ct_pred")
    ct_gt_dir_H = osp.join(resdir, "CT", "H", "ct_gt")
    ct_pred_dir_W = osp.join(resdir, "CT", "W", "ct_pred")
    ct_gt_dir_W = osp.join(resdir, "CT", "W", "ct_gt")
    ct_pred_dir_L = osp.join(resdir, "CT", "L", "ct_pred")
    ct_gt_dir_L = osp.join(resdir, "CT", "L", "ct_gt")

    H, W, L = image_pred.shape
    logger.info(image_pred.shape)

    os.makedirs(proj_pred_dir, exist_ok=True)
    os.makedirs(proj_gt_dir, exist_ok=True)
    os.makedirs(ct_pred_dir_H, exist_ok=True)
    os.makedirs(ct_gt_dir_H, exist_ok=True)
    os.makedirs(ct_pred_dir_W, exist_ok=True)
    os.makedirs(ct_gt_dir_W, exist_ok=True)
    os.makedirs(ct_pred_dir_L, exist_ok=True)
    os.makedirs(ct_gt_dir_L, exist_ok=True)

    for i in tqdm(range(N)):
        iio.imwrite(osp.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((1-cast_to_image(projs_pred[i]))*255).astype(np.uint8))
        iio.imwrite(osp.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((1-cast_to_image(projs[i]))*255).astype(np.uint8))
    
    for i in tqdm(range(H)):
        iio.imwrite(osp.join(ct_pred_dir_H, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[i,...])*255).astype(np.uint8))
        iio.imwrite(osp.join(ct_gt_dir_H, f"ct_gt_{str(i)}.png"), (cast_to_image(image[i,...])*255).astype(np.uint8))

    for i in tqdm(range(W)):
        iio.imwrite(osp.join(ct_pred_dir_W, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[:,i,:])*255).astype(np.uint8))
        iio.imwrite(osp.join(ct_gt_dir_W, f"ct_gt_{str(i)}.png"), (cast_to_image(image[:,i,:])*255).astype(np.uint8))

    for i in tqdm(range(L)):
        iio.imwrite(osp.join(ct_pred_dir_L, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[...,i])*255).astype(np.uint8))
        iio.imwrite(osp.join(ct_gt_dir_L, f"ct_gt_{str(i)}.png"), (cast_to_image(image[...,i])*255).astype(np.uint8))

    image_pred = image_pred.permute(2, 0, 1)
    image = image.permute(2, 0, 1)

    np.save(os.path.join(root_path, 'projs_pred.npy'), projs_pred.cpu().numpy())
    np.save(os.path.join(root_path, 'label_pred.npy'), projs.cpu().numpy())

    return image_pred,image,projs_pred,projs

def iteration(config,image_pred,image,geo,angles,save_root):
    # 设置设备为CUDA
    config['device']= torch.device("cuda")

    # 指定保存生成样本的目录
    sample_save_dir = os.path.join(save_root,"eval_samples")
    Path(sample_save_dir).mkdir(parents=True, exist_ok=True)

    lamb = 0.04  # 正则化参数λ
    rho = 10 # ADMM参数ρ

    sde ,sampling_eps = initSDE(config)
    sde.N = config['diffusion_model']['num_scales']
    # 设置预测器和校正器
    predictor = ReverseDiffusionPredictor  # 反向扩散预测器
    corrector = LangevinCorrector  # Langevin校正器
    probability_flow = False  # 是否使用概率流
    snr = 0.16  # 信噪比

    sigmas = mutils.get_sigmas(config)
    # 数据缩放器
    scaler = datasets.get_data_scaler(config)
    # 数据逆缩放器
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    # 创建分数模型
    score_model = mutils.create_model(config)  # 根据配置创建模型
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config['diffusion_model']['ema_rate'])  # 创建指数移动平均对象
    optimizer = losses.get_optimizer(config, score_model.parameters())  # 根据配置创建优化器
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)  # 初始化训练状态字典

    # 加载检查点
    checkpoint_dir = os.path.join("check/diffusion/")
    ckpt_path = os.path.join(checkpoint_dir, config['diffusion_eval']['pre_model'])
    state = restore_checkpoint(ckpt_path, state, config['device'])
    ema.copy_to(score_model.parameters())
    score_model.eval()  # 设置模型为评估模式

    # 创建各种类型的保存目录
    irl_types = ['input', 'recon', 'label', 'BP', 'sinogram',"img"]
    for t in irl_types:
        if t == 'recon':
            save_root_f = os.path.join(save_root,t,'progress' )   # 重建进度目录
        else:
            save_root_f = os.path.join(save_root, t)  # 其他类型目录
        os.makedirs(save_root_f, exist_ok=True)

    # 读取所有数据文件
    all_image = image_pred
    label = image

    radon = CT(img_width=256, radon_view=len(angles), circle=False, device=config['device'])

    predicted_sinogram = []  # 预测的sinogram列表
    label_sinogram = []  # 标签sinogram列表

    # 将图像移动到指定设备
    # img = all_image.to(config['device'])[25:26]
    # label = label.to(config['device'])[25:26]

    img = torch.from_numpy(all_image[25:26]).to(config['device'])
    label = torch.from_numpy(label[25:26]).to(config['device'])

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    label = (label - label.min()) / (label.max() - label.min() + 1e-8)

    img = img.unsqueeze(1)  # 在第1维增加1个维度
    label = label.unsqueeze(1)  # 在第1维增加1个维度
    # 获取带有ADMM和TV正则化的PC Radon采样器
    pc_radon = controllable_generation_TV.get_pc_radon_ADMM_TV_vol(sde,
                                                                predictor, corrector,
                                                                inverse_scaler,
                                                                snr=snr,
                                                                n_steps=config['diffusion_sampling']['n_steps_each'],
                                                                probability_flow=probability_flow,
                                                                continuous=config['diffusion_training']['continuous'],
                                                                denoise=True,
                                                                radon=radon,
                                                                save_progress=True,
                                                                save_root=save_root,
                                                                final_consistency=True,
                                                                img_shape=img.shape,
                                                                lamb_1=lamb,
                                                                rho=rho)
    
    

    #通过Radon变换获取稀疏sinogram
    sinogram = radon.A(img)

    #反投影
    bp = radon.AT(sinogram)

    # 重建图像
    x = pc_radon(score_model, scaler(img),measurement=sinogram)
    # 计算指标（逐样本计算）
    batch_mse = get_mse(x, label)
    batch_psnr = get_psnr_3d(x, label)
    batch_ssim = get_ssim(x, label)
    logger.info(f"x and label : batch_mse: {batch_mse} batch_psnr: {batch_psnr} batch_ssim: {batch_ssim}")


    batch_mse = get_mse(x, img)
    batch_psnr = get_psnr_3d(x, img)
    batch_ssim = get_ssim(x, img)
    logger.info(f"x and img : batch_mse: {batch_mse} batch_psnr: {batch_psnr} batch_ssim: {batch_ssim}")

    batch_mse = get_mse(img, label)
    batch_psnr = get_psnr_3d(img, label)
    batch_ssim = get_ssim(img, label)
    logger.info(f"img and label : batch_mse: {batch_mse} batch_psnr: {batch_psnr} batch_ssim: {batch_ssim}")

    img_cahce = x[-1].unsqueeze(0)  # 缓存最后一幅图像

    count = 0
    # 保存各种结果图像
    for i, recon_img in enumerate(x):
        plt.imsave(os.path.join(save_root, 'BP', f'{count}.png'), clear(bp[i]), cmap='gray')
        plt.imsave(os.path.join(save_root, 'label', f'{count}.png'), clear(label[i]), cmap='gray')
        plt.imsave(os.path.join(save_root, 'img', f'{count}.png'), clear(img[i]), cmap='gray')
        plt.imsave(os.path.join(save_root, 'recon', f'{count}.png'), clear(recon_img), cmap='gray')
        count += 1
    pass



if __name__ == "__main__":
    #初始化日志、路径
    cfg = {**load_config(args.config1),**load_config(args.config2)}
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time) 
    root_path = os.path.join(f'logs/Iteration/', args.nerf_category,date_time)
    Path(root_path).mkdir(parents=True, exist_ok=True)  
    logger = gen_log(root_path)
    logger.info(cfg)
    device = torch.device("cuda")

    # 载入数据、model
    eval_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device) if cfg["log"]["i_eval"] > 0 else None
    voxels = eval_dset.voxels if cfg["log"]["i_eval"] > 0 else None
    geo = eval_dset.geo
    #angles = eval_dset.dataset.angles
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    network = get_network(cfg["network"]["net_type"])
    network_tpye = cfg["network"]["net_type"]
    cfg["network"].pop("net_type", None)
    encoder = get_encoder(**cfg["encoder"])
    model = network(encoder, **cfg["network"]).to(device)
    model_fine = None
    n_fine = cfg["render"]["n_fine"]
    if n_fine > 0:
        model_fine = network(encoder, **cfg["network"]).to(device)
    weights_path = os.path.join('check','Nerf',network_tpye, args.nerf_category,cfg['test']['pre_model'])
    ckpt = torch.load(weights_path, map_location=device)

    model.load_state_dict(ckpt["network"])
    if n_fine > 0:
        model_fine.load_state_dict(ckpt["network_fine"])
    
    model.eval()
    if n_fine > 0:
        model_fine.eval()

    with torch.no_grad():
        image_pred,image,projs_pred,projs = eval_nerf(eval_dset, model, model_fine, cfg,root_path)
    #projs_pred = np.load('projs_pred.npy')
    #projs = np.load('label_pred.npy')

    #iteration(cfg,projs_pred,projs,geo,angles,root_path)

