"""All functions and modules related to model definition.
"""

import torch
import utilis.diffusion_utils.sde_lib as sde_lib
import numpy as np

_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config['diffusion_model']['sigma_max']), np.log(config['diffusion_model']['sigma_min']), config['diffusion_model']['num_scales']))

  return sigmas


def get_ddpm_params(config):
  """Get betas and alphas --- parameters used in the original DDPM paper."""
  num_diffusion_timesteps = 1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas
  alphas_cumprod = np.cumprod(alphas, axis=0)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config):
  """Create the score model."""
  model_name = config['diffusion_model']['name']
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config['device'])
  return score_model


def get_model_fn(model, train=False):
    """创建模型调用函数，统一训练/评估模式下的行为。

    Args:
        model: 分数模型（如UNet），需支持输入(x, labels)。
        train: 若为True，启用训练模式（如Dropout、BatchNorm）；否则为评估模式。
    
    Returns:
        model_fn: 函数，输入(x, labels)，输出模型预测结果。
    """

    def model_fn(x, labels):
        """计算分数模型的输出。

        Args:
            x: 输入数据张量，形状通常为 [B, C, H, W]。
            labels: 时间步标签，可以是连续或离散值，具体解释取决于模型。

        Returns:
            模型的输出（分数估计），无额外修改。
        """
        if not train:
            # 评估模式：关闭Dropout、冻结BatchNorm统计量
            model.eval()
            return model(x, labels)
        else:
            # 训练模式：启用Dropout、更新BatchNorm统计量
            model.train()
            return model(x, labels)

    return model_fn



def get_score_fn(sde, model, train=False, continuous=False):
    # 获取模型的推理函数（根据train决定是否启用训练模式）
    model_fn = get_model_fn(model, train=train)

    # Case 1: 处理VPSDE或subVPSDE（方差保持/子方差保持SDE）
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
        def score_fn(x, t):
            # 连续时间步或subVPSDE的特殊处理
            if continuous or isinstance(sde, sde_lib.subVPSDE):
                # 将时间步t映射到[0, 999]区间（连续时间模型假设最大时间嵌入为999）
                labels = t * 999
                # 调用模型获取原始分数估计
                score = model_fn(x, labels)
                # 计算当前时间步的边际分布标准差 std = sqrt(beta(t))
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # 离散时间步：将t映射到[0, sde.N-1]的整数
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                # 从预计算的表中获取标准差（sqrt(1 - α_cumprod)）
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            # 分数修正：score = -模型输出 / std
            # 因为真实分数是 - (x - mean)/std^2 = - (x / std) / std
            score = -score / std[:, None, None, None]
            return score

    # Case 2: 处理VESDE（方差爆炸SDE）
    elif isinstance(sde, sde_lib.VESDE):
        def score_fn(x, t):
            if continuous:
                # 连续时间步：直接获取当前时间步的噪声标准差
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # 离散时间步：VESDE中t=0对应最大噪声，需反向映射
                labels = sde.T - t  # 反转时间（T是最大时间）
                labels *= sde.N - 1  # 映射到[0, sde.N-1]
                labels = torch.round(labels).long()  # 四舍五入为整数

            # 调用模型获取分数（VESDE下模型直接输出分数，无需缩放）
            score = model_fn(x, labels)
            return score

    # 其他SDE类型不支持
    else:
        raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))