"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np

class SDE(abc.ABC):
  """随机微分方程（SDE）的抽象基类，所有方法设计为支持小批量输入（mini-batch）"""

  def __init__(self, N):
    """初始化 SDE 类

    Args:
        N (int): 离散化时间步数（用于数值方法的时间离散化）
    """
    super().__init__()  # 调用父类 abc.ABC 的初始化
    self.N = N  # 存储时间步数

  # ----------------------------------------------------------------------------
  # 抽象属性和方法（必须由子类实现）
  # ----------------------------------------------------------------------------
  @property
  @abc.abstractmethod
  def T(self):
    """SDE 的终止时间（连续时间范围，例如 T=1）

    例如：物理时间 t ∈ [0, T]
    """
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    """定义 SDE 的漂移（drift）和扩散（diffusion）系数

    Args:
        x (Tensor): 当前状态张量，形状为 (batch_size, *data_shape)
        t (Tensor): 当前时间步张量，形状为 (batch_size,)

    Returns:
        drift (Tensor): 漂移项 f(x, t)，形状与 x 相同
        diffusion (Tensor): 扩散项 g(t)，形状为 (batch_size, 1, 1, 1)（可能需广播）
    """
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """计算 SDE 边际分布的参数（例如均值和标准差）

    通常用于定义前向扩散过程 p_t(x)

    Args:
        x (Tensor): 初始数据
        t (Tensor): 时间步

    Returns:
        mean (Tensor): 均值参数
        std (Tensor): 标准差参数
    """
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """从先验分布 p_T(x) 生成样本（通常为标准高斯噪声）

    Args:
        shape (tuple): 生成样本的形状，例如 (batch_size, channels, height, width)

    Returns:
        Tensor: 采样结果，形状为 `shape`
    """
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """计算先验分布的对数概率密度（用于概率流 ODE 的似然计算）

    Args:
        z (Tensor): 潜在编码

    Returns:
        Tensor: 对数概率密度，形状为 (batch_size,)
    """
    pass

  # ----------------------------------------------------------------------------
  # 具体方法（可直接使用或由子类覆盖）
  # ----------------------------------------------------------------------------
  def discretize(self, x, t):
    """离散化 SDE（默认使用 Euler-Maruyama 方法）

    离散化形式：x_{i+1} = x_i + f_i(x_i) * dt + G_i * z_i * sqrt(dt)

    Args:
        x (Tensor): 当前状态，形状为 (batch_size, *data_shape)
        t (Tensor): 当前时间，形状为 (batch_size,)

    Returns:
        f (Tensor): 离散漂移项 f_i(x_i) = drift * dt
        G (Tensor): 离散扩散项 G_i = diffusion * sqrt(dt)
    """
    dt = 1 / self.N  # 时间步长 Δt = T/N
    drift, diffusion = self.sde(x, t)  # 调用抽象方法获取连续项
    f = drift * dt  # 漂移项离散化
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))  # 扩散项离散化
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """构建逆向时间 SDE/ODE

    Args:
        score_fn (callable): 评分函数，接受 x 和 t 返回 score
        probability_flow (bool): 若为 True，构建用于概率流采样的 ODE

    Returns:
        RSDE: 逆向 SDE/ODE 的实例
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # --------------------------------------------------------------------
    # 逆向 SDE/ODE 的内部类
    # --------------------------------------------------------------------
    class RSDE(self.__class__):
      """逆向时间 SDE/ODE 的实现"""

      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow  # 标记是否为概率流 ODE

      @property
      def T(self):
        return T  # 保持与原 SDE 相同的终止时间

      def sde(self, x, t):
        """逆向 SDE 的漂移和扩散项"""
        drift, diffusion = sde_fn(x, t)  # 原始 SDE 的漂移和扩散
        score = score_fn(x, t)  # 计算得分（来自模型）

        # 逆向漂移修正（根据是否概率流选择系数）
        drift_rev = drift - diffusion[:, None, None, None]  **  2 * score * (0.5 if self.probability_flow else 1.0)

        # 如果是概率流 ODE，扩散项设为 0
        diffusion_rev = 0.0 if self.probability_flow else diffusion
        return drift_rev, diffusion_rev

      def discretize(self, x, t):
        """离散化逆向过程（用于反向扩散采样）"""
        f, G = discretize_fn(x, t)  # 原始离散化项
        score = score_fn(x, t)  # 计算得分

        # 修正漂移项
        rev_f = f - G[:, None, None, None] **  2 * score * (0.5 if self.probability_flow else 1.0)

        # 如果是概率流 ODE，扩散项设为 0
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()  # 返回逆向 SDE 的实例


class VPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  """方差爆炸型随机微分方程（Variance Exploding SDE）实现
  特征：扩散过程的方差随时间指数增长至最大值（sigma_max^2）
  适用场景：基于分数的生成模型（如 NCSN、SMLD）
  """

  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """初始化 VE-SDE
    Args:
        sigma_min (float): 噪声标准差的最小值（起始噪声级别）
        sigma_max (float): 噪声标准差的最大值（终止噪声级别）
        N (int): 时间离散化步数（影响数值稳定性）
    """
    super().__init__(N)  # 调用父类 SDE 的初始化，传入离散化步数 N
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    # 生成离散化的噪声级别序列（对数均匀分布 → 指数增长）
    self.discrete_sigmas = torch.exp(
      torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N)
    )
    self.N = N  # 冗余定义，强调类属性

  @property
  def T(self):
    """SDE 的终止时间（固定为1，时间范围 t ∈ [0,1]）"""
    return 1  # VE-SDE 的标准设计，将物理时间归一化到 [0,1]

  def sde(self, x, t):
    """计算连续形式 SDE 的漂移项 (drift) 和扩散项 (diffusion)

    Args:
        x (Tensor): 当前状态，形状 (batch_size, *data_dims)
        t (Tensor): 当前时间，形状 (batch_size,)

    Returns:
        drift (Tensor): 漂移系数 f(x,t)，本实现中始终为 0
        diffusion (Tensor): 扩散系数 g(t)，形状 (batch_size, 1, 1, 1)
    """
    # 计算随时间指数增长的噪声标准差 σ(t) = σ_min * (σ_max/σ_min)^t
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min)  **  t
    # VE-SDE 的漂移项为 0（纯扩散过程）
    drift = torch.zeros_like(x)  # 形状与 x 相同
    # 扩散系数 g(t) = σ(t) * sqrt(2 * log(σ_max/σ_min))
    diffusion = sigma * torch.sqrt(
      torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device)
    )
    return drift, diffusion[:, None, None, None]  # 扩展维度以匹配数据形状

  def marginal_prob(self, x, t):
    """计算扩散过程在时间 t 的边际分布 p_t(x) 的参数（均值和标准差）

    对于 VE-SDE，边际分布为高斯分布：
        p_t(x) = N(x; x_0, σ(t)^2 I)
    其中 σ(t) = σ_min * (σ_max/σ_min)^t

    Args:
        x (Tensor): 初始数据 x_0
        t (Tensor): 时间步

    Returns:
        mean (Tensor): 均值（此处等于 x_0，因为漂移项为 0）
        std (Tensor): 标准差 σ(t)
    """
    std = self.sigma_min * (self.sigma_max / self.sigma_min)  **  t
    return x, std  # 均值不变，标准差随时间增长

  def prior_sampling(self, shape):
    """从先验分布 p_T(x) 采样（最终时刻的高斯噪声）

    Args:
        shape (tuple): 样本形状，如 (batch_size, C, H, W)

    Returns:
        Tensor: 采样结果 ~ N(0, σ_max^2 I)
    """
    return torch.randn(*shape) * self.sigma_max  # 标准高斯采样后缩放

  def prior_logp(self, z):
    """计算先验分布 p_T(z) 的对数概率密度

    Args:
        z (Tensor): 潜在变量，形状 (batch_size, *data_dims)

    Returns:
        Tensor: 对数概率密度，形状 (batch_size,)
    """
    shape = z.shape
    # 计算数据维度总数（通道×高度×宽度）
    N = np.prod(shape[1:])  # 排除 batch 维度
    # 高斯对数概率公式：log p(z) = -0.5*(N log(2πσ^2) + (z^2)/(σ^2)) 求和
    logp = -N / 2. * np.log(2 * np.pi * self.sigma_max  **  2)  # 归一化项
    logp -= torch.sum(z  **  2, dim = (1, 2, 3)) / (2 * self.sigma_max  **  2)  # 数据项
    return logp

  def discretize(self, x, t):
    """离散化方法（覆盖父类默认的欧拉-丸山方法，改用 SMLD/NCSN 的离散方式）

    离散化形式：
        x_{i+1} = x_i + f_i(x_i) + G_i * z_i
    其中：
        f_i = 0（漂移项为 0）
        G_i = sqrt(σ_{i+1}^2 - σ_i^2)（确保离散过程方差连续增长）

    Args:
        x (Tensor): 当前状态
        t (Tensor): 当前时间（已归一化到 [0,1]）

    Returns:
        f (Tensor): 离散漂移项（全零）
        G (Tensor): 离散扩散系数
    """
    # 将连续时间 t ∈ [0,1] 映射到离散时间步索引 [0, N-1]
    timestep = (t * (self.N - 1) / self.T).long()
    # 获取当前时间步对应的 σ_i
    sigma = self.discrete_sigmas.to(t.device)[timestep]  # 确保设备一致
    # 获取前一时间步的 σ_{i-1}（处理边界 i=0 的情况）
    adjacent_sigma = torch.where(
      timestep == 0,
      torch.zeros_like(t),  # 若 i=0，相邻 σ 设为 0（对应 σ_0 为初始值）
      self.discrete_sigmas.to(t.device)[timestep - 1]
    )
    f = torch.zeros_like(x)  # 漂移项保持为 0
    # 计算离散扩散系数 G_i = sqrt(σ_i^2 - σ_{i-1}^2)
    G = torch.sqrt(sigma  **  2 - adjacent_sigma  **  2)
    return f, G