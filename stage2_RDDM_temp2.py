import argparse  
import os
# 初始化设置
# 配置参数解析器
def config_parser():
    parser = argparse.ArgumentParser()
    # 添加配置文件路径参数，默认为TensorF的配置文件
    parser.add_argument("--config", default=f"./config/RDDM/AAPMDR_512_light_L067.yaml")
    # 添加GPU ID参数，指定使用的GPU
    parser.add_argument("--gpu_id", default="0", help="使用的GPU编号")
    parser.add_argument("--phase", default=f"RDDM_Test")
    return parser

# 解析命令行参数
parser = config_parser()
args = parser.parse_args()

# 设置CUDA环境变量（指定可见的GPU设备）
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'  # 按照PCI总线ID顺序排列GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 只暴露指定的GPU给程序使用



from model.RDDM.residual_denoising_diffusion_pytorch import ResidualDiffusion,Trainer, UnetRes
import torch
from pathlib import Path  # 路径处理模块，用于处理文件路径
import datetime
from utilis.Nerf.Nerf_utils  import gen_log, time2file_name
from configloading import load_config  # 配置文件加载器
import yaml


# 加载配置文件
cfg = load_config(args.config)
# 设置设备为CUDA
cfg['device'] = torch.device("cuda:0")


if __name__ == "__main__":
    # 训练参数设置
    cfg['phase'] = args.phase
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time) 
    save_path = os.path.join("logs", cfg['phase'],date_time)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    cfg['save_path'] = save_path
    logger = gen_log(save_path)
    formatted_cfg = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    logger.info("\n" + formatted_cfg) 



    model = UnetRes(cfg)
    diffusion = ResidualDiffusion(model,cfg)
    trainer = Trainer(diffusion, cfg,logger,save_path)


    if cfg['phase'] == 'RDDM_Train' :
        trainer.train()
    elif cfg['phase'] == 'RDDM_Test' :
        trainer.load()
        trainer.test(last=True)
