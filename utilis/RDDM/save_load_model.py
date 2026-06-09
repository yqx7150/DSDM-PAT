from pathlib import Path
import torch

def exists(x):
    return x is not None


def save_RDDM(cfg,ema,opt0,model,accelerator,dir,opt1=None):
    if not accelerator.is_local_main_process:
        return
    if cfg['diffusion_model']['num_unet']  == 1:
        data = {
            'model': accelerator.get_state_dict(model),
            'opt0': opt0.state_dict(),
            'ema': ema.state_dict(),
            'scaler': accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
    elif cfg['diffusion_model']['num_unet']  == 2:
        data = {
            'model': accelerator.get_state_dict(model),
            'opt0': opt0.state_dict(),
            'opt1': opt1.state_dict(),
            'ema': ema.state_dict(),
            'scaler': accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
    torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

def load_RDDM(milestone):
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



def sample(self, milestone, last=True, FID=False):
    """使用EMA模型生成样本并保存结果
    
    Args:
        milestone (int): 当前保存的里程碑步数（用于文件名）
        last (bool): 是否只保存最终生成的样本（否则保存所有中间结果）
        FID (bool): 是否为FID评估生成样本（影响保存格式）
    Returns:
        int: 更新后的milestone值（用于连续生成时追踪进度）
    """
    # 将EMA模型设为评估模式（关闭Dropout/BatchNorm等随机性层）
    self.ema.ema_model.eval()

    # 禁用梯度计算以节省显存和加速
    with torch.no_grad():
        # 确定生成样本的批次大小
        batches = self.num_samples
        
        # --- 条件输入处理 ---
        # 根据条件类型准备输入数据（x_input_sample）和可视化数据（show_x_input_sample）
        if self.condition_type == 0:  # 无条件生成
            x_input_sample = [0]  # 无输入条件
            show_x_input_sample = []  # 无需要展示的输入
        elif self.condition_type == 1:  # 单条件输入（如超分辨率）
            # 从sample_loader加载一个批次的条件数据
            x_input_sample = [next(self.sample_loader).to(self.device)]
            show_x_input_sample = x_input_sample  # 直接展示输入条件
        elif self.condition_type == 2:  # 双条件输入（如带参考图的修复）
            x_input_sample = next(self.sample_loader)
            # 将所有条件数据移动到设备
            x_input_sample = [item.to(self.device) for item in x_input_sample]
            show_x_input_sample = x_input_sample  # 展示完整输入
            x_input_sample = x_input_sample[1:]  # 实际使用除第一个外的条件（如仅用mask）
        elif self.condition_type == 3:  # 多条件输入（如文本+图像+边缘）
            x_input_sample = next(self.sample_loader)
            x_input_sample = [item.to(self.device) for item in x_input_sample]
            show_x_input_sample = x_input_sample  # 展示完整输入
            x_input_sample = x_input_sample[1:]  # 实际使用部分条件

        # --- 样本生成 ---
        # 组合输入条件和生成结果（show_x_input_sample + 生成图像列表）
        all_images_list = show_x_input_sample + \
            list(self.ema.ema_model.sample(
                x_input_sample,         # 实际输入条件
                batch_size=batches,     # 生成数量
                last=last               # 是否返回所有中间结果
            ))

        # 将所有图像拼接为单个张量（用于批量保存）
        all_images = torch.cat(all_images_list, dim=0)

        # --- 图像排列方式 ---
        # 根据是否保存中间结果决定排列行数
        if last:  # 只保存最终结果 => 方形网格排列
            nrow = int(math.sqrt(self.num_samples))
        else:     # 保存所有中间结果 => 单行排列
            nrow = all_images.shape[0]

        # --- 结果保存 ---
        if FID:  # FID评估模式（每个样本单独保存）
            for i in range(batches):
                # 生成唯一文件名（如sample-1000.png）
                file_name = f'sample-{milestone}.png'
                # 保存条件输入的第i个样本（单张图像）
                utils.save_image(
                    all_images_list[0][i].unsqueeze(0),  # 添加批次维度
                    os.path.join(self.results_folder, file_name),
                    nrow=1  # 单行排列
                )
                milestone += 1
                # 达到FID评估所需数量时提前终止
                if milestone >= self.total_n_samples:
                    break
        else:  # 常规保存模式（所有样本保存为一张图）
            file_name = f'sample-{milestone}.png'
            utils.save_image(
                all_images,  # 所有样本拼接结果
                str(self.results_folder / file_name),
                nrow=nrow  # 动态决定排列方式
            )
        print("sample-save " + file_name)  # 日志输出
    
    # 恢复EMA模型为训练模式（不影响后续训练）
    self.ema.ema_model.train()
    return milestone  # 返回更新后的milestone



def sample(self, x_input=0, batch_size=16, last=True):
    """使用扩散模型生成样本（支持DDIM和原始DDPM采样）
    
    Args:
        x_input (int/list/tensor): 条件输入数据，默认为0表示无条件生成
        batch_size (int): 每批生成的样本数量
        last (bool): 是否只返回最终生成结果（否则返回所有中间步骤）
    Returns:
        generator: 生成样本的迭代器（可能包含多个时间步的结果）
    """
    # 获取模型预设的图像尺寸和通道数
    image_size, channels = self.image_size, self.channels

    # 根据是否使用DDIM采样选择对应的采样函数
    # self.is_ddim_sampling是布尔值，表示是否启用DDIM加速采样
    sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

    # --- 条件输入处理 ---
    if self.condition:  # 如果是条件生成模式
        # 处理输入条件的归一化（不同条件类型的处理）
        if self.input_condition and self.input_condition_mask:
            # 双条件输入情况（如图像+掩码），只归一化第一个条件
            x_input[0] = normalize_to_neg_one_to_one(x_input[0])
        else:
            # 单条件输入，直接归一化整个输入
            x_input = normalize_to_neg_one_to_one(x_input)
        
        # 从条件输入中提取实际的批量大小和图像尺寸
        # 条件输入的shape为 [batch_size, channels, height, width]
        batch_size, channels, h, w = x_input[0].shape
        size = (batch_size, channels, h, w)  # 生成尺寸与输入条件保持一致
    else:  # 无条件生成模式
        # 使用模型预设的尺寸生成图像
        size = (batch_size, channels, image_size, image_size)

    # 调用选定的采样函数（p_sample_loop或ddim_sample）生成样本
    # 返回结果可能是：
    # - 当last=True时：只返回最终生成结果的张量
    # - 当last=False时：返回包含所有时间步结果的生成器
    return sample_fn(x_input, size, last=last)