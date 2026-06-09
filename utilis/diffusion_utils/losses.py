import torch
import torch.optim as optim
import numpy as np
from model.diffusion.diffusion_model import utils as mutils
from utilis.diffusion_utils.sde_lib import VESDE, VPSDE
from utilis.diffusion_utils.utils import fft2, ifft2, get_mask


def get_optimizer(config, params):
  # 检查配置中指定的优化器类型
  if config['diffusion_optim']['optimizer'] == 'Adam':
    # 实例化Flax的Adam优化器
    optimizer = optim.Adam(
      # 待优化的模型参数（从模型初始化获得）
      params,
      # 学习率，默认值可参考论文或经验值（如0.001）
      lr=config['diffusion_optim']['lr'],
      # beta1和beta2是Adam的动量参数，beta2固定为0.999
      betas=(config['diffusion_optim']['beta1'], 0.999),
      # 数值稳定性常数，防止除以零（通常保持默认1e-8）
      eps=config['diffusion_optim']['eps'],
      # L2正则化系数，用于权重衰减（默认0.0不启用）
      weight_decay=config['diffusion_optim']['weight_decay']
    )
  else:
    # 抛出未实现错误，提示用户扩展其他优化器
    raise NotImplementedError(
      f'Optimizer not supported yet!'
      ' 可支持的优化器: ["Adam"]'  # 可扩展为["Adam", "SGD", "RMSprop"]
    )

  return optimizer


def optimization_manager(config):
  """根据配置生成并返回一个优化器管理函数 `optimize_fn`

  功能:
      该工厂函数生成一个优化器管理函数，用于在训练过程中动态调整学习率（如预热）、
      执行梯度裁剪等操作。适用于PyTorch框架的优化器管理。

  参数:
      config (dict/namespace):
          配置对象，需包含以下字段：
          - config.optim.lr: 基础学习率（预热结束后使用的学习率）
          - config.optim.warmup: 预热步数（warmup=0表示禁用预热）
          - config.optim.grad_clip: 梯度裁剪阈值（<0表示禁用裁剪）

  返回:
      function: 优化器管理函数 `optimize_fn`，其参数为：
          - optimizer: PyTorch优化器实例
          - params: 模型参数（用于梯度裁剪）
          - step: 当前训练步数（用于学习率预热）
          - lr: 可覆盖配置的基础学习率（可选）
          - warmup: 可覆盖配置的预热步数（可选）
          - grad_clip: 可覆盖配置的梯度裁剪阈值（可选）
  """

  def optimize_fn(optimizer, params, step, lr=config['diffusion_optim']['lr'],
                  warmup=config['diffusion_optim']['warmup'], grad_clip=config['diffusion_optim']['grad_clip']):
    """执行优化步骤，包含学习率预热和梯度裁剪

    参数:
        optimizer (torch.optim.Optimizer):
            PyTorch优化器实例（如Adam、SGD）
        params (iterable):
            模型参数列表/生成器，用于梯度裁剪（如model.parameters()）
        step (int):
            当前训练步数（从0开始计数）
        lr (float):
            基础学习率（预热结束后生效）
        warmup (int):
            学习率预热步数（若>0，学习率将线性增长至`lr`）
        grad_clip (float):
            梯度裁剪阈值（若>=0，裁剪梯度范数；<0时禁用）

    逻辑:
        1. 学习率预热：在 `warmup` 步内，学习率从0线性增长到 `lr`
        2. 梯度裁剪：若 `grad_clip >=0`，裁剪梯度范数不超过此阈值
        3. 执行优化器更新：调用 optimizer.step()
    """

    # 学习率预热 --------------------------------------------------------
    if warmup > 0:
      # 遍历所有参数组（如不同层可设置不同学习率）
      for g in optimizer.param_groups:
        # 当前步的学习率 = 基础学习率 * min(step / warmup, 1.0)
        # 例如：warmup=1000，step=500 → lr=0.5*lr；step>=1000 → lr=lr
        g['lr'] = lr * min(step / warmup, 1.0)

    # 梯度裁剪 ---------------------------------------------------------
    if grad_clip >= 0:
      # 使用PyTorch的梯度裁剪函数，限制所有梯度的范数不超过grad_clip
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

    # 执行优化器参数更新 ------------------------------------------------
    optimizer.step()

  return optimize_fn  # 返回闭包函数


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """创建一个用于训练任意SDE（随机微分方程）的损失函数。

  参数:
    sde: 一个 `sde_lib.SDE` 对象，表示前向SDE。
    train: `True` 表示训练损失，`False` 表示评估损失。
    reduce_mean: 如果为 `True`，则在数据维度上对损失取平均；否则对损失求和。
    continuous: 如果为 `True`，表示模型定义为连续时间步长；否则需要插值来处理连续时间步长。
    likelihood_weighting: 如果为 `True`，则根据 https://arxiv.org/abs/2101.09258 对分数匹配损失进行加权；
      否则使用论文中推荐的加权方式。
    eps: 一个浮点数，表示采样时的最小时间步长。

  返回:
    一个损失函数。
  """
  # 定义损失 reduction 操作：取平均或求和
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """计算损失函数。

    参数:
      model: 一个分数模型。
      batch: 一个训练数据的小批量。

    返回:
      loss: 一个标量，表示小批量上的平均损失值。
    """
    # 从模型中获取分数函数
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)

    # 为每个数据点随机采样时间步长 t
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps

    # 采样与 batch 形状相同的高斯噪声 z
    z = torch.randn_like(batch)

    # 计算时间 t 对应的边际概率分布的均值和标准差
    mean, std = sde.marginal_prob(batch, t)

    # 使用噪声 z 对数据进行扰动，扰动幅度由标准差 std 控制
    perturbed_data = mean + std[:, None, None, None] * z

    # 计算扰动数据在时间 t 处的分数
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      # 不使用似然加权时的损失计算
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      # 使用似然加权时的损失计算
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2  # 计算扩散系数的平方
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    # 计算整个小批量的平均损失
    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn





def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """创建单步训练/评估函数

  功能:
      根据输入参数（SDE类型、训练模式等）生成一个单步训练/评估函数`step_fn`。
      该函数负责：
      1. 计算损失（训练模式或评估模式）
      2. 反向传播梯度（仅训练模式）
      3. 更新模型参数和EMA状态（仅训练模式）

  参数:
      sde (sde_lib.SDE):
          随机微分方程对象，定义前向扩散过程（如VESDE/VPSDE）
      train (bool):
          是否为训练模式。True时执行梯度更新，False时仅计算损失
      optimize_fn (function, optional):
          优化器更新函数，需实现参数更新逻辑（类似PyTorch的optim.step()）
      reduce_mean (bool):
          损失计算方式。True表示平均损失，False表示求和
      continuous (bool):
          是否使用连续时间模型。True对应Score SDE，False对应DDPM/SMLD
      likelihood_weighting (bool):
          是否应用似然加权策略（仅连续时间SDE有效）

  返回:
      function: 单步执行函数`step_fn`，接收`state`和`batch`，返回损失值
  """

  # 阶段1: 根据SDE类型选择损失函数 --------------------------------------------
  if continuous:
    # 连续时间SDE的损失函数（Score SDE）
    loss_fn = get_sde_loss_fn(
      sde, train,
      reduce_mean=reduce_mean,
      continuous=True,
      likelihood_weighting=likelihood_weighting
    )
  else:
    # 离散时间SDE（DDPM/SMLD）的损失函数
    assert not likelihood_weighting, "离散训练不支持似然加权"

    if isinstance(sde, VESDE):
      # SMLD（Variance Exploding SDE）
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      # DDPM（Variance Preserving SDE）
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"不支持的离散SDE类型: {sde.__class__.__name__}")

  # 阶段2: 定义单步执行函数 ------------------------------------------------
  def step_fn(state, batch):
    """单步训练/评估的核心逻辑

    参数:
        state (dict): 包含以下键值：
            - 'model': 当前模型实例
            - 'optimizer': 优化器对象（仅训练模式）
            - 'ema': EMA状态管理器（如ExponentialMovingAverage）
            - 'step': 当前训练步数计数器
        batch (Tensor): 当前批次数据，形状为 [B, C, H, W]

    返回:
        loss (Tensor): 标量损失值（detached from计算图）
    """

    # 从state中提取模型
    model = state['model']

    if train:
      # 训练模式逻辑 ------------------------------------------------
      optimizer = state['optimizer']

      # 1. 梯度清零
      optimizer.zero_grad()

      # 2. 计算损失（前向传播）
      loss = loss_fn(model, batch)

      # 3. 反向传播梯度
      loss.backward()  # 需PyTorch风格自动微分

      # 4. 优化器更新参数
      optimize_fn(optimizer, model.parameters(), step=state['step'])

      # 5. 更新训练步数
      state['step'] += 1

      # 6. 更新EMA参数
      state['ema'].update(model.parameters())  # EMA跟踪模型参数

    else:
      # 评估模式逻辑 ------------------------------------------------
      with torch.no_grad():  # 禁用梯度计算
        ema = state['ema']

        # 1. 保存原始参数
        ema.store(model.parameters())

        # 2. 将EMA参数复制到模型
        ema.copy_to(model.parameters())

        # 3. 计算损失（使用EMA参数）
        loss = loss_fn(model, batch)

        # 4. 恢复原始参数
        ema.restore(model.parameters())

    return loss  # 返回标量损失值

  return step_fn  # 返回闭包函数



def get_step_fn_regression(train, config, mask=None, loss_fn=None, optimize_fn=None):

  def step_fn(state, batch):
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()

      # fft
      kspace = fft2(batch)

      # sample mask

      acc_factor = np.random.choice(config['diffusion_training']['acc_factor'])
      mask = get_mask(batch, config['diffusion_data']['image_size'], config['diffusion_training']['batch_size'],
                      type=config['diffusion_training']['mask_type'],
                      acc_factor=acc_factor,
                      fix=True)

      # undersampling
      under_kspace = kspace * mask
      under_img = torch.abs(ifft2(under_kspace))

      est_img = model(under_img)
      loss = loss_fn(est_img, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
      return loss
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        # fft
        kspace = fft2(batch)

        # sample mask
        mask = get_mask(batch, config['diffusion_data']['image_size'], config['diffusion_training']['batch_size'],
                        type=config['diffusion_training']['mask_type'],
                        acc_factor=config['diffusion_training']['acc_factor'])

        # undersampling
        under_kspace = kspace * mask
        under_img = torch.real(ifft2(under_kspace))

        est_img = model(under_img)
        ema.restore(model.parameters())
        return est_img
  return step_fn
