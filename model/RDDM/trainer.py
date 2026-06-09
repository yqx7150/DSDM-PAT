import copy
import glob
import math
import os
import random
from collections import namedtuple
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator
from dataset.RDDM.get_dataset import dataset
from ema_pytorch import EMA
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])


def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)



class Trainer(object):
    def __init__(
        self,
        diffusion_model,  # 核心扩散模型（必须传入）
        folder,          # 训练数据路径（字符串或路径列表）
        *,  # 强制后续参数必须用关键字传递
        train_batch_size=16,      # 训练批次大小
        gradient_accumulate_every=1,  # 梯度累积步数（模拟更大batch_size）
        augment_flip=True,        # 是否启用随机水平/垂直翻转增强
        train_lr=1e-4,            # 学习率
        train_num_steps=100000,   # 总训练步数
        ema_update_every=10,      # EMA（指数移动平均）更新频率
        ema_decay=0.995,          # EMA衰减系数
        adam_betas=(0.9, 0.99),   # Adam优化器的beta参数
        save_and_sample_every=1000,  # 每隔多少步保存和生成样本
        num_samples=1,           # 每次生成的样本数量
        results_folder='./results/sample',  # 结果保存路径
        amp=False,                # 是否启用自动混合精度
        fp16=False,               # 是否使用FP16半精度
        split_batches=False,       # 是否在多GPU训练时拆分批次
        convert_image_to=None,    # 图像格式转换（如RGB/L等）
        condition=False,          # 是否使用条件生成模式
        #sub_dir=False,            # 是否在子目录中查找数据
        equalizeHist=False,       # 是否应用直方图均衡化
        crop_patch=False,         # 是否裁剪图像块
        generation=False,         # 是否处于生成模式（影响数据加载）
        num_unet=2                # UNet数量（1或2）
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        #self.sub_dir = sub_dir
        self.crop_patch = crop_patch

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        self.num_unet = num_unet

        if self.condition:
            '''
            单条件输入生成（如超分辨率）
            双条件输入（如带参考图的修复）
            多模态条件生成（如文本+图像+边缘）
            图像修复 
            '''
            if len(folder) == 3:
                self.condition_type = 1
                # test_input
                ds = dataset(folder[-1], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                               pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)))
            elif len(folder) == 4:
                self.condition_type = 2
                # test_gt+test_input
                ds = dataset(folder[2:4], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                               pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)))
            elif len(folder) == 6:
                self.condition_type = 3
                # test_gt+test_input
                ds = dataset(folder[3:6], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:3]

                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                                                                               pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)))
            else:
                self.condition_type = 3
                ds = dataset(folder[0], self.image_size)
                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,pin_memory=True, num_workers=4)))  
                ds = dataset(folder[1], self.image_size)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)))


        else:
            self.condition_type = 0
            trian_folder = folder

            ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                         convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
            self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                            shuffle=True, pin_memory=True, num_workers=4)))


        if self.num_unet == 1:
            self.opt0 = RAdam(diffusion_model.parameters(),
                              lr=train_lr, weight_decay=0.0)
        elif self.num_unet == 2:
            self.opt0 = RAdam(
                diffusion_model.model.unet0.parameters(), lr=train_lr, weight_decay=0.0)
            self.opt1 = RAdam(
                diffusion_model.model.unet1.parameters(), lr=train_lr, weight_decay=0.0)


        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        if self.num_unet == 1:
            self.model, self.opt = self.accelerator.prepare(
                self.model, self.opt0)
        elif self.num_unet == 2:
            self.model, self.opt0, self.opt1 = self.accelerator.prepare(
                self.model, self.opt0, self.opt1)
        device = self.accelerator.device
        self.device = device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        if self.num_unet == 1:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt0': self.opt0.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
        elif self.num_unet == 2:
            data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt0': self.opt0.state_dict(),
                'opt1': self.opt1.state_dict(),
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
            }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')

        if path.exists():
            data = torch.load(
                str(path), map_location=self.device)

            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])

            self.step = data['step']
            if self.num_unet == 1:
                self.opt0.load_state_dict(data['opt0'])
            elif self.num_unet == 2:
                self.opt0.load_state_dict(data['opt0'])
                self.opt1.load_state_dict(data['opt1'])
            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - "+str(path))

        # self.ema.to(self.device)

    def train(self):
        """执行模型训练的主循环"""
        # 获取加速器实例（用于多GPU/TPU训练）
        accelerator = self.accelerator

        # 初始化进度条（仅主进程显示）
        with tqdm(initial=self.step, 
                total=self.train_num_steps,
                disable=not accelerator.is_main_process) as pbar:

            # 训练循环（直到达到总步数）
            while self.step < self.train_num_steps:
                # 初始化损失记录（支持单/双UNet结构）
                if self.num_unet == 1:
                    total_loss = [0]  # [unet0_loss]
                elif self.num_unet == 2:
                    total_loss = [0, 0]  # [unet0_loss, unet1_loss]

                # --- 梯度累积阶段 ---
                #for _ in range(self.gradient_accumulate_every):
                ###############################################
                ###############################################
                ###############################################
                # 数据加载（区分条件/无条件模式）
                if self.condition:
                    data = next(self.dl)  # 获取条件数据批次
                    data = [item.to(self.device) for item in data]  # 转移所有条件数据到设备
                else:
                    data = next(self.dl)  # 获取普通数据批次
                    data = data[0] if isinstance(data, list) else data  # 解包数据
                    data = data.to(self.device)  # 数据转移到设备

                # 混合精度训练上下文
                with self.accelerator.autocast():
                    # 前向计算损失（返回列表，每个UNet对应一个损失）
                    loss = self.model(data)
                    
                    # 梯度累积下的损失归一化
                    for i in range(self.num_unet):
                        loss[i] = loss[i] 
                        total_loss[i] = total_loss[i] + loss[i].item()  # 累加损失

                # 反向传播（每个UNet独立计算梯度）
                for i in range(self.num_unet):
                    self.accelerator.backward(loss[i])

                # --- 梯度更新阶段 ---
                # 梯度裁剪（防止梯度爆炸）
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                # 等待所有进程同步（多GPU/TPU必需）
                accelerator.wait_for_everyone()

                # 参数更新（区分单/双UNet优化器）
                if self.num_unet == 1:
                    self.opt0.step()       # 更新UNet0参数
                    self.opt0.zero_grad()  # 清零梯度
                elif self.num_unet == 2:
                    self.opt0.step()       # 更新UNet0参数
                    self.opt0.zero_grad()  
                    self.opt1.step()       # 更新UNet1参数
                    self.opt1.zero_grad()

                # 再次同步所有进程
                accelerator.wait_for_everyone()

                # --- 训练状态更新 ---
                self.step += 1  # 全局步数+1

                # 主进程专属操作
                if accelerator.is_main_process:
                    # EMA模型更新
                    self.ema.to(self.device)
                    self.ema.update()  # 滑动平均更新参数

                    # 定期采样和保存
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.sample(milestone)  # 生成样本预览

                        # 更低频率的模型保存
                        if self.step % (self.save_and_sample_every * 10) == 0:
                            self.save(milestone)  # 保存完整检查点

                # 更新进度条显示
                if self.num_unet == 1:
                    pbar.set_description(f'loss_unet0: {total_loss[0]:.4f}')
                elif self.num_unet == 2:
                    pbar.set_description(
                        f'loss_unet0: {total_loss[0]:.4f}, loss_unet1: {total_loss[1]:.4f}')
                pbar.update(1)  # 进度条前进一步

        # 训练完成提示（仅主进程打印）
        accelerator.print('training complete')

    def sample(self, milestone, last=True, FID=False):
        self.ema.ema_model.eval()

        with torch.no_grad():
            batches = self.num_samples
            if self.condition_type == 0:
                x_input_sample = [0]
                show_x_input_sample = []
            elif self.condition_type == 1:
                x_input_sample = [next(self.sample_loader).to(self.device)]
                show_x_input_sample = x_input_sample
            elif self.condition_type == 2:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device)
                                  for item in x_input_sample]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]
            elif self.condition_type == 3:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device)
                                  for item in x_input_sample]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]

            all_images_list = show_x_input_sample + \
                list(self.ema.ema_model.sample(
                    x_input_sample, batch_size=batches, last=last))

            all_images = torch.cat(all_images_list, dim=0)

            if last:
                nrow = int(math.sqrt(self.num_samples))
            else:
                nrow = all_images.shape[0]

            if FID:
                for i in range(batches):
                    file_name = f'sample-{milestone}.png'
                    utils.save_image(
                        all_images_list[0][i].unsqueeze(0), os.path.join(self.results_folder, file_name), nrow=1)
                    milestone += 1
                    if milestone >= self.total_n_samples:
                        break
            else:
                file_name = f'sample-{milestone}.png'
                utils.save_image(all_images, str(
                    self.results_folder / file_name), nrow=nrow)
            print("sampe-save "+file_name)
        self.ema.ema_model.train()
        return milestone

    def test(self, sample=False, last=True, FID=False):
        # self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=self.num_samples)
            i = 0
            for items in loader:
                if self.condition:
                    file_name = self.sample_dataset.load_name(
                        i, sub_dir=self.sub_dir)
                    file_name = f'{i}.png' if file_name==None else file_name
                else:
                    file_name = f'{i}.png'
                i += 1

                with torch.no_grad():
                    batches = self.num_samples

                    if self.condition_type == 0:
                        x_input_sample = [0]
                        show_x_input_sample = []
                    elif self.condition_type == 1:
                        x_input_sample = [items.to(self.device)]
                        show_x_input_sample = x_input_sample
                    elif self.condition_type == 2:
                        x_input_sample = [item.to(self.device)
                                          for item in items]
                        show_x_input_sample = x_input_sample
                        x_input_sample = x_input_sample[1:]
                    elif self.condition_type == 3:
                        x_input_sample = [item.to(self.device)
                                          for item in items]
                        show_x_input_sample = x_input_sample
                        x_input_sample = x_input_sample[1:]

                    if sample:
                        all_images_list = show_x_input_sample + \
                            list(self.ema.ema_model.sample(
                                x_input_sample, batch_size=batches, last=last))
                    else:
                        all_images_list = list(self.ema.ema_model.sample(
                            x_input_sample, batch_size=batches, last=last))
                        all_images_list = [all_images_list[-1]]
                        if self.crop_patch:
                            k = 0
                            for img in all_images_list:
                                pad_size = self.sample_dataset.get_pad_size(i)
                                _, _, h, w = img.shape
                                img = img[:, :, 0:h -
                                          pad_size[0], 0:w-pad_size[1]]
                                all_images_list[k] = img
                                k += 1

                all_images = torch.cat(all_images_list, dim=0)

                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]

                utils.save_image(all_images, str(
                    self.results_folder / file_name), nrow=nrow)
                print("test-save "+file_name)
        else:
            if FID:
                self.total_n_samples = 50000
                img_id = len(glob.glob(f"{self.results_folder}/*"))
                n_rounds = (self.total_n_samples -
                            img_id) // self.num_samples+1
            else:
                n_rounds = 100
            for i in range(n_rounds):
                if FID:
                    i = img_id
                img_id = self.sample(i, last=last, FID=FID)
        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)







#############################################################################################################