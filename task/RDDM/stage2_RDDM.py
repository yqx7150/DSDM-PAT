import os
import sys
import argparse  # 命令行参数解析
# 导入自定义的扩散模型相关模块
from model.RDDM.denoising_diffusion_pytorch import GaussianDiffusion
from model.RDDM.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)
import torch
from pathlib import Path  # 路径处理模块，用于处理文件路径
import datetime
from utilis.Nerf.Nerf_utils  import gen_log, time2file_name
from configloading import load_config  # 配置文件加载器



# 初始化设置
# 配置参数解析器
def config_parser():
    parser = argparse.ArgumentParser()
    # 添加配置文件路径参数，默认为TensorF的配置文件
    parser.add_argument("--config", default=f"./config/RDDM/AAPM_256.yaml", help="配置文件路径")
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

# 是否使用原始DDIM/DDPM模型# 模型相关参数设置
objective = 'pred_res_noise'  # 目标函数类型
test_res_or_noise = "res_noise"  # 测试时预测残差或噪声

# 训练参数设置

phase = 'Train'
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time) 
save_path = os.path.join("logs", f"DDRM_{phase}",date_time)
sample_dir = os.path.join(save_path, "samples") 
Path(sample_dir).mkdir(parents=True, exist_ok=True)  
checkpoint_dir = os.path.join(save_path, "checkpoints")  # 检查点保存目录
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
logger = gen_log(save_path)
logger.info(cfg)


model = UnetRes(
    dim=64,  # 基础维度  # 各层维度倍数
    channels = cfg['diffusion_data']['num_channels'],
    condition=cfg['diffusion_training']['condition'],  # 是否使用条件输入
    input_condition=cfg['diffusion_training']['input_condition'],  # 是否使用输入条件
    objective=objective,  # 目标函数类型
    test_res_or_noise=test_res_or_noise,  # 测试时预测残差或噪声
    img_to_img_translation=cfg['diffusion_data']['img_to_img_translation']  # 是否为图像到图像转换任务
)
# 残差扩散模型
diffusion = ResidualDiffusion(
    model,
    image_size=cfg['diffusion_data']['image_size'],  # 图像尺寸
    timesteps=cfg['diffusion_model']['num_scales'] ,  # 扩散步数
    sampling_timesteps=cfg['diffusion_model']['num_scales'] ,  # 采样步数
    objective=objective,  # 目标函数类型
    condition=cfg['diffusion_training']['condition'],  # 是否使用条件输入
    sum_scale=1,  # 求和缩放因子
    input_condition=cfg['diffusion_training']['input_condition'],  # 是否使用输入条件
    input_condition_mask=cfg['diffusion_training']['input_condition_mask'],  # 是否使用输入条件掩码
    test_res_or_noise=test_res_or_noise,  # 测试时预测残差或噪声
    img_to_img_translation=cfg['diffusion_data']['img_to_img_translation']  # 是否为图像到图像转换任务
)

folder = ['data/proj_pred','data/proj_gt']
# 创建训练器实例
trainer = Trainer(
    diffusion,  # 扩散模型
    folder = folder, #cfg['diffusion_data']['root'],  # 数据路径
    train_batch_size=cfg['diffusion_training']['batch_size'],  # 训练批次大小

    train_lr=cfg['diffusion_optim']['lr'],  # 学习率
    train_num_steps=cfg['diffusion_training']['epochs'] ,  # 训练总步数
    gradient_accumulate_every=2,  # 梯度累积步数
    ema_decay=cfg['diffusion_model']['ema_rate'],  # 指数移动平均衰减率
    amp=False,  # 是否使用混合精度训练
    convert_image_to="L",  # 图像转换格式
    condition=cfg['diffusion_training']['condition'],  # 是否使用条件输入
    save_and_sample_every=cfg['diffusion_training']['eval_freq_sampling'],  # 保存和采样间隔
    equalizeHist=False,  # 是否均衡化直方图
    crop_patch=False,  # 是否裁剪图像块
    generation=True,  # 是否为生成任务
    results_folder = save_path,
)

# 开始训练
trainer.train()

# test
trainer.load(trainer.train_num_steps//cfg['diffusion_training']['eval_freq_sampling'])
# 设置结果保存路径
trainer.set_results_folder(save_path)
# 进行最终测试
trainer.test(last=True)

#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
# def train(config,phase='train'):
#     # 创建实验日志目录 样本保存目录,检查点目录
#     date_time = str(datetime.datetime.now())
#     date_time = time2file_name(date_time) 
#     save_path = os.path.join("logs", f"Diffusion_{phase}",date_time)
#     sample_dir = os.path.join(save_path, "samples") 
#     Path(sample_dir).mkdir(parents=True, exist_ok=True)  
#     checkpoint_dir = os.path.join(save_path, "checkpoints")  # 检查点保存目录
#     Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
#     logger = gen_log(save_path)
#     logger.info(config)

#     model = UnetRes(
#     dim=64,  # 基础维度  # 各层维度倍数
#     channels = cfg['diffusion_data']['num_channels'],
#     condition=cfg['diffusion_training']['condition'],  # 是否使用条件输入
#     input_condition=cfg['diffusion_training']['input_condition'],  # 是否使用输入条件
#     objective=objective,  # 目标函数类型
#     test_res_or_noise=test_res_or_noise,  # 测试时预测残差或噪声
#     img_to_img_translation=cfg['diffusion_data']['img_to_img_translation']  # 是否为图像到图像转换任务
#     )
#     # 残差扩散模型
#     diffusion = ResidualDiffusion(
#     model,
#     image_size=cfg['diffusion_data']['image_size'],  # 图像尺寸
#     timesteps=cfg['diffusion_model']['num_scales'] ,  # 扩散步数
#     sampling_timesteps=cfg['diffusion_model']['num_scales'] ,  # 采样步数
#     objective=objective,  # 目标函数类型
#     condition=cfg['diffusion_training']['condition'],  # 是否使用条件输入
#     sum_scale=1,  # 求和缩放因子
#     input_condition=cfg['diffusion_training']['input_condition'],  # 是否使用输入条件
#     input_condition_mask=cfg['diffusion_training']['input_condition_mask'],  # 是否使用输入条件掩码
#     test_res_or_noise=test_res_or_noise,  # 测试时预测残差或噪声
#     img_to_img_translation=cfg['diffusion_data']['img_to_img_translation']  # 是否为图像到图像转换任务
#     )

#     # 初始化模型
#     score_model = mutils.create_model(config)  
#     ema = ExponentialMovingAverage(score_model.parameters(), decay=config['diffusion_model']['ema_rate'])  
#     optimizer = losses.get_optimizer(config, score_model.parameters())  
#     state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)  

#     # 构建PyTorch数据加载器
#     train_dl = datasets.create_dataloader(config,config['diffusion_training']['batch_size'],'train')  # 创建训练和评估数据加载器
#     eval_dl = datasets.create_dataloader(config,config['diffusion_training']['batch_size'],'test')  # 创建训练和评估数据加载器
#     # 创建数据标准化器和其逆操作
#     scaler = datasets.get_data_scaler(config)  # 获取数据标准化器
#     inverse_scaler = datasets.get_data_inverse_scaler(config)  # 获取数据逆标准化器

#     sde ,sampling_eps = initSDE(config)

#     # 构建一步训练和评估函数
#     optimize_fn = losses.optimization_manager(config)  # 获取优化管理器
#     continuous = config['diffusion_training']['continuous']  # 是否使用连续时间训练
#     reduce_mean = config['diffusion_training']['reduce_mean']  # 是否对损失求均值
#     likelihood_weighting = config['diffusion_training']['likelihood_weighting']  # 是否使用似然加权
#     train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
#                                       reduce_mean=reduce_mean, continuous=continuous,
#                                       likelihood_weighting=likelihood_weighting)  

#     # 构建采样函数
#     if config['diffusion_training']['eval_freq_sampling']!=0:
#         sampling_shape = (config['diffusion_training']['batch_size'], config['diffusion_data']['num_channels'],
#                           config['diffusion_data']['image_size'], config['diffusion_data']['image_size'])  # 采样形状
        
#         sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)  # 获取采样函数


#     # 训练循环
#     for epoch in range(1, config['diffusion_training']['epochs']):
#         logger.info('=================================================')
#         logger.info(f"[EVAL] epoch: {epoch}/{config['diffusion_training']['epochs']}")
#         logger.info('=================================================')

#         for step, batch in enumerate(train_dl, start=1):
#             batch = scaler(batch.to(config['device']))  # 对数据进行标准化并移动到指定设备

#             loss = train_step_fn(state, batch)  # 计算损失
#             if step % config['diffusion_training']['log_freq'] == 0:
#                 logger.info(f"Training | Step: {step:6d} | Loss: {loss.item():.7f}")

#         # 每个epoch保存一个检查点
#         if epoch % config['diffusion_training']['eval_freq'] == 0:
#             save_checkpoint(checkpoint_dir, state, name=f'checkpoint_{epoch}.pth')  # 保存检查点

#         # 每个epoch生成并保存样本
#         if epoch % config['diffusion_training']['eval_freq_sampling'] == 0 and config['diffusion_training']['eval_freq_sampling'] !=0:
#                 for step_eval, batch_eval in enumerate(eval_dl):
#                     if step_eval == 1:
#                         x_true = scaler(batch_eval.to(config['device']))
#                         logger.info('sampling')
#                         ema.store(score_model.parameters())  # 保存EMA参数
#                         ema.copy_to(score_model.parameters())  # 将EMA参数复制到模型
#                         logger.info("当前生成sample：{}".format(step_eval))
#                         x_hat, _ = sampling_fn(score_model,y=x_true)  # 从随机噪声生成

#                         batch_size_eval = x_true.size(0)
#                         for sample_idx in range(batch_size_eval):
#                             this_sample_dir = os.path.join(sample_dir, "epoch_{}".format(epoch))  # 样本保存目录
#                             Path(this_sample_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
#                             # 获取单个样本
#                             real_img = x_true[sample_idx]
#                             fake_img = x_hat[sample_idx]
#                             # 生成唯一文件名
#                             real_path = os.path.join(this_sample_dir,
#                                                     f"sample{sample_idx:03d}_real.png")
#                             fake_path = os.path.join(this_sample_dir,
#                                                     f"sample{sample_idx:03d}_predict.png")
#                             # 保存为独立文件
#                             plt.imsave(real_path, real_img[0].cpu(), cmap=plt.cm.Greys_r)
#                             plt.imsave(fake_path, fake_img[0].cpu(), cmap=plt.cm.Greys_r)