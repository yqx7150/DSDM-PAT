import os
import torch
import argparse  # 命令行参数解析
from pathlib import Path  # 路径处理模块，用于处理文件路径
import numpy as np  # 数值计算库，用于处理数组和矩阵
from  model.diffusion.diffusion_model import ddpm, ncsnv2, ncsnpp, unet,ncsnpp_condition  # 导入模型定义（注释掉了）
import utilis.diffusion_utils.losses as losses  # 导入损失函数模块
import utilis.diffusion_utils.sampling as sampling  # 导入采样模块
from model.diffusion.diffusion_model import utils as mutils  # 导入模型工具模块
from model.diffusion.diffusion_model.ema import ExponentialMovingAverage  # 导入指数移动平均模块
import dataset.diffusion_dataset.datasets as datasets  # 导入数据集模块
from torchvision.utils import make_grid, save_image  # 图像处理工具，用于生成图像网格和保存图像
from utilis.diffusion_utils.utils import save_checkpoint, restore_checkpoint, initSDE, root_sum_of_squares  # 导入自定义工具函数
from utilis.Nerf.Nerf_utils import get_psnr, get_ssim ,get_mse ,get_psnr_3d
import matplotlib.pyplot as plt
from configloading import load_config  # 配置文件加载器
from utilis.Nerf.Nerf_utils  import gen_log, time2file_name
import datetime
import yaml

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))  #GPU 名称
print(torch.cuda.get_device_capability(0))  # GPU 算力
# 配置参数解析器
def config_parser():
    parser = argparse.ArgumentParser()
    # 添加配置文件路径参数，默认为TensorF的配置文件
    parser.add_argument("--config", default=f"./config/Diffusion_config/AAPM_256_DR_256x2_1000_noBN.yaml")
    # 添加GPU ID参数，指定使用的GPU
    parser.add_argument("--gpu_id", default="1", help="使用的GPU编号")
    return parser
# 解析命令行参数
parser = config_parser()
args = parser.parse_args()
# 设置CUDA环境变量（指定可见的GPU设备）
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'  # 按照PCI总线ID顺序排列GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 只暴露指定的GPU给程序使用


# 加载配置文件
cfg = load_config(args.config)
# 设置设备为CUDA
cfg['device']= torch.device(f"cuda:{args.gpu_id}")



def train(config,phase='train'):
    # 创建实验日志目录 样本保存目录,检查点目录
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time) 
    save_path = os.path.join("logs", f"Diffusion_{phase}",date_time)
    sample_dir = os.path.join(save_path, "samples") 
    Path(sample_dir).mkdir(parents=True, exist_ok=True)  
    checkpoint_dir = os.path.join(save_path, "checkpoints")  # 检查点保存目录
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
    logger = gen_log(save_path)
    formatted_cfg = yaml.dump(config, default_flow_style=False, sort_keys=False)
    logger.info("\n" + formatted_cfg) 

    # 初始化模型
    score_model = mutils.create_model(config)  
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config['diffusion_model']['ema_rate'])  
    optimizer = losses.get_optimizer(config, score_model.parameters())  
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)  

    # 构建PyTorch数据加载器
    train_dl = datasets.create_dataloader(config,config['diffusion_training']['batch_size'],'train')  # 创建训练和评估数据加载器
    eval_dl = datasets.create_dataloader(config,config['diffusion_training']['batch_size'],'test')  # 创建训练和评估数据加载器
    # 创建数据标准化器和其逆操作
    scaler = datasets.get_data_scaler(config)  # 获取数据标准化器
    inverse_scaler = datasets.get_data_inverse_scaler(config)  # 获取数据逆标准化器

    sde ,sampling_eps = initSDE(config)

    # 构建一步训练和评估函数
    optimize_fn = losses.optimization_manager(config)  # 获取优化管理器
    continuous = config['diffusion_training']['continuous']  # 是否使用连续时间训练
    reduce_mean = config['diffusion_training']['reduce_mean']  # 是否对损失求均值
    likelihood_weighting = config['diffusion_training']['likelihood_weighting']  # 是否使用似然加权
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)  

    # 构建采样函数
    if config['diffusion_training']['eval_freq_sampling']!=0:
        sampling_shape = (config['diffusion_training']['batch_size'], config['diffusion_data']['num_channels'],
                          config['diffusion_data']['image_size'], config['diffusion_data']['image_size'])  # 采样形状
        
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)  # 获取采样函数


    # 训练循环
    for epoch in range(1, config['diffusion_training']['epochs']):
        logger.info('=================================================')
        logger.info(f"[EVAL] epoch: {epoch}/{config['diffusion_training']['epochs']}")
        logger.info('=================================================')

        for step, batch in enumerate(train_dl, start=1):
            batch = scaler(batch.to(config['device']))  # 对数据进行标准化并移动到指定设备

            loss = train_step_fn(state, batch)  # 计算损失
            if step % config['diffusion_training']['log_freq'] == 0:
                logger.info(f"Training | Step: {step:6d} | Loss: {loss.item():.7f}")

        # 每个epoch保存一个检查点
        if epoch % config['diffusion_training']['eval_freq'] == 0:
            save_checkpoint(checkpoint_dir, state, name=f'checkpoint_{epoch}.pth')  # 保存检查点

        # 每个epoch生成并保存样本
        if epoch % config['diffusion_training']['eval_freq_sampling'] == 0 and config['diffusion_training']['eval_freq_sampling'] !=0:
                for step_eval, batch_eval in enumerate(eval_dl):
                    if step_eval == 1:
                        x_true = scaler(batch_eval.to(config['device']))
                        logger.info('sampling')
                        ema.store(score_model.parameters())  # 保存EMA参数
                        ema.copy_to(score_model.parameters())  # 将EMA参数复制到模型
                        logger.info("当前生成sample：{}".format(step_eval))
                        x_hat, _ = sampling_fn(score_model,y=x_true)  # 从随机噪声生成

                        batch_size_eval = x_true.size(0)
                        for sample_idx in range(batch_size_eval):
                            this_sample_dir = os.path.join(sample_dir, "epoch_{}".format(epoch))  # 样本保存目录
                            Path(this_sample_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
                            # 获取单个样本
                            real_img = x_true[sample_idx]
                            fake_img = x_hat[sample_idx]
                            # 生成唯一文件名
                            real_path = os.path.join(this_sample_dir,
                                                    f"sample{sample_idx:03d}_real.png")
                            fake_path = os.path.join(this_sample_dir,
                                                    f"sample{sample_idx:03d}_predict.png")
                            # 保存为独立文件
                            plt.imsave(real_path, real_img[0].cpu(), cmap=plt.cm.Greys_r)
                            plt.imsave(fake_path, fake_img[0].cpu(), cmap=plt.cm.Greys_r)

def evaluate(config, phase='test'):
    # 创建实验日志目录 样本保存目录,检查点目录
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time) 
    eval_dir = os.path.join("logs", f"Diffusion_{phase}",date_time)
    os.makedirs(eval_dir, exist_ok=True)
    sample_save_dir = os.path.join(eval_dir,"eval_samples")
    Path(sample_save_dir).mkdir(parents=True, exist_ok=True)
    logger = gen_log(eval_dir)

    # 初始化模型
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

    # 数据加载
    eval_dl = datasets.create_dataloader(config,config['diffusion_eval']['batch_size'],'test')  
    geo = eval_dl.dataset.geo.geo
    angles = eval_dl.dataset.angles
    # 数据标准化和逆标准化函数
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    total_psnr = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    num_samples = 0

    # 设置SDE（与训练时相同）
    sde ,sampling_eps = initSDE(config)
    # 采样函数
    sampling_shape = (config['diffusion_eval']['batch_size'],
                      config['diffusion_data']['num_channels'],
                      config['diffusion_data']['image_size'],
                      config['diffusion_data']['image_size'])
    config['sample_dir'] = sample_save_dir
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler,sampling_eps,geo=geo,angles=angles)

    logger.info(config)
    # 生成样本
    num_samples = len(eval_dl.dataset)
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dl):      # 直接从数据集中获取真实图像
            logger.info("当前生成sample：{}".format(batch_idx))
            x_true = scaler(batch.to(config['device']))

            x_hat, _ = sampling_fn(score_model,y=x_true)  # 从随机噪声生成


            assert x_hat.shape == x_true.shape, "生成样本与真实数据形状不匹配"

            batch_size = x_true.size(0)
            for sample_idx in range(batch_size):
                # 获取单个样本
                real_img = x_true[sample_idx]
                fake_img = x_hat[sample_idx]
                # 生成唯一文件名
                real_path = os.path.join(sample_save_dir,
                                        f"batch{batch_idx:03d}_sample{sample_idx:03d}_real.png")
                fake_path = os.path.join(sample_save_dir,
                                        f"batch{batch_idx:03d}_sample{sample_idx:03d}_predict.png")
                # 保存为独立文件
                plt.imsave(real_path, real_img[0].cpu(), cmap=plt.cm.Greys_r)
                plt.imsave(fake_path, fake_img[0].cpu(), cmap=plt.cm.Greys_r)
            # 计算指标（逐样本计算）
            batch_mse = get_mse(x_hat, x_true)
            batch_psnr = get_psnr_3d(x_hat, x_true)
            batch_ssim = get_ssim(x_hat, x_true)
            logger.info(f"batch: {batch_idx} batch_mse: {batch_mse} batch_psnr: {batch_psnr} batch_ssim: {batch_ssim}")

            # 累加结果
            total_mse += batch_mse.item() * x_true.size(0)
            total_psnr += batch_psnr.item() * x_true.size(0)
            total_ssim += batch_ssim.item() * x_true.size(0)

    # 计算平均指标
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    logger.info(f"num_samples{num_samples} avg_mse: {avg_mse} avg_psnr: {avg_psnr} avg_ssim: {avg_ssim}")



# 当脚本作为主程序运行时，执行main函数
if __name__ == "__main__":
  train(cfg)
  #evaluate(cfg)

