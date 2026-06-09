import functools
import time
import torch
import numpy as np
import abc
from model.diffusion.diffusion_model.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import utilis.diffusion_utils.sde_lib as sde_lib
from model.diffusion.diffusion_model import utils as mutils
from utilis.diffusion_utils.tirge_lib import tigre_sin_consist
from tqdm import tqdm 
_CORRECTORS = {}
_PREDICTORS = {}

# 注册器模式：用于注册不同的预测器和校正器
def register_predictor(cls=None, *, name=None):
  """类装饰器，用于将预测器（Predictor）子类注册到全局注册表。

  功能说明:
      实现一个灵活的注册器模式，允许通过名称（name）动态管理不同的预测器类。
      注册后的类可通过名称从全局字典 _PREDICTORS 中检索，便于后续实例化。

  参数:
      cls (type, optional): 待注册的预测器类。若为 None，返回装饰器函数等待调用。
      name (str, optional): 注册名称。若未指定，默认使用类的 __name__ 属性。

  返回值:
      若 cls 为 None，返回装饰器函数 _register；否则返回注册后的类。

  使用场景:
      在定义 Predictor 子类时，通过装饰器语法注册，例如：
      @register_predictor(name='euler')
      class EulerPredictor(Predictor):
          ...

      或省略 name 参数，使用类名：
      @register_predictor()
      class MyPredictor(Predictor):
          ...

  设计亮点:
      - 支持两种调用方式：带参数（指定名称）或不带参数（自动获取类名）
      - 防止名称重复注册，避免覆盖
  """

  # 内部装饰器函数，实际执行注册逻辑
  def _register(cls):
    # 确定注册名称：优先使用显式指定的 name，否则用类名
    local_name = name if name is not None else cls.__name__
    # 检查名称是否已被占用
    if local_name in _PREDICTORS:
      raise ValueError(f'名称 {local_name} 已被注册，请更换其他名称')
    # 注册类到全局字典
    _PREDICTORS[local_name] = cls
    return cls  # 返回原类，不影响其定义

  # 处理装饰器的两种调用方式：
  # 1. 带参数调用：@register_predictor(name='xxx')
  if cls is None:
    return _register  # 返回待应用的装饰器函数
  # 2. 无参数调用：@register_predictor
  else:
    return _register(cls)  # 直接应用装饰器


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps,geo=None,angles=None,ct_consist=True):

  """根据配置创建样本生成函数，用于扩散模型的推理阶段。

  功能:
      根据配置文件中的采样方法（ODE 或 PC），生成对应的采样函数。
      该函数用于从训练好的扩散模型中生成样本，支持以下方法：
      - ODE（概率流常微分方程求解器）
      - PC（Predictor-Corrector，预测-校正联合采样）

  参数:
      config (ml_collections.ConfigDict):
          包含所有配置信息的对象，关键字段包括：
          - config.sampling.method: 采样方法名称（'ode' 或 'pc'）
          - config.sampling.noise_removal: 是否在最终步去噪（布尔值）
          - config.sampling.predictor: 预测器类型（当 method='pc' 时生效）
          - config.sampling.corrector: 校正器类型（当 method='pc' 时生效）
          - config.sampling.snr: 信噪比阈值（用于校正器步长调整）
          - config.sampling.n_steps_each: 每个时间步的校正器迭代次数
          - config.sampling.probability_flow: 是否启用概率流（布尔值）
          - config.training.continuous: 模型是否基于连续时间训练
          - config.device: 计算设备（如 'cuda:0'）

      sde (sde_lib.SDE):
          随机微分方程对象，定义前向扩散过程（如 VESDE/VPSDE）。

      shape (tuple):
          单个样本的期望形状，例如图像数据为 (C, H, W)。

      inverse_scaler (function):
          逆数据标准化函数，将模型输出的 [-1, 1] 范围数据还原到原始数据范围（如 [0, 255]）。

      eps (float):
          反向时间 SDE 积分的终止时间（出于数值稳定性考虑，避免 t=0 的奇点）。

  返回:
      function:
          采样函数 `sampling_fn`，其输入为随机状态和训练状态，输出形状为 `shape` 的样本。
          示例用法：
          samples = sampling_fn(rng, state)  # JAX 风格
          或
          samples = sampling_fn(noise, model)  # PyTorch 风格

  异常:
      ValueError: 当配置中指定了未知的采样方法时抛出。
  """

  # 获取配置中指定的采样方法名称（不区分大小写）
  sampler_name = config['diffusion_sampling']['method']

  # 选择 ODE 采样方法 ------------------------------------------------------
  if sampler_name.lower() == 'ode':
      # 使用概率流 ODE 求解器（黑盒 ODE 求解器，如 Euler 或 RK45）
      sampling_fn = get_ode_sampler(
          sde=sde,  # 定义前向/反向 SDE
          shape=shape,  # 样本形状（如 (C, H, W)）
          inverse_scaler=inverse_scaler,  # 逆标准化函数
          denoise=config['diffusion_sampling']['noise_removal'],  # 是否在最后一步去噪
          eps=eps,  # 反向积分的终止时间（避免 t=0）
          device=config['device']  # 计算设备（如 GPU）
      )

  # 选择 Predictor-Corrector (PC) 采样方法 ---------------------------------
  elif sampler_name.lower() == 'pc':
      # 获取预测器和校正器（通过名称动态加载）
      predictor = get_predictor(config['diffusion_sampling']['predictor'].lower())  # 如 'euler'
      corrector = get_corrector(config['diffusion_sampling']['corrector'].lower())  # 如 'langevin'

      # 创建 PC 采样函数
      sampling_fn = get_pc_sampler(
          sde=sde,  # 定义前向/反向 SDE
          shape=shape,  # 样本形状
          predictor=predictor,  # 预测器对象（如 EulerPredictor）
          corrector=corrector,  # 校正器对象（如 LangevinCorrector）
          inverse_scaler=inverse_scaler,  # 逆标准化函数
          snr=config['diffusion_sampling']['snr'],  # 信噪比阈值（控制校正器步长）
          n_steps=config['diffusion_sampling']['n_steps_each'],  # 每个时间步的校正次数
          probability_flow=config['diffusion_sampling']['probability_flow'],  # 是否启用概率流
          continuous=config['diffusion_training']['continuous'],  # 模型是否连续时间训练
          denoise=config['diffusion_sampling']['noise_removal'],  # 是否在最后一步去噪
          eps=eps,  # 终止时间
          device=config['device'],  # 计算设备
          cs=config['diffusion_sampling']['cs'],
          sigma_min=config['diffusion_sampling']['sigma_min'],
          N=config['diffusion_sampling']['N'],
          sigma_max=config['diffusion_sampling']['sigma_max'],
          discrete_sigmas=config['diffusion_sampling']['discrete_sigmas'],
          geo = geo,
          angles=angles,
          ct_consist = ct_consist
      )
  # 处理未知采样方法 -------------------------------------------------------
  else:
    raise ValueError(f"未知的采样器名称 {sampler_name}，支持的选项: ['ode', 'pc']")

  return sampling_fn

#预测器抽象基类，定义所有预测器算法的统一接口
class Predictor(abc.ABC):
  """预测器抽象基类，定义所有预测器算法的统一接口。

  功能说明:
      预测器用于扩散模型的逆向过程（生成过程），负责从当前状态 x_t 和时间 t 计算下一步状态 x_{t-Δt}。
      通过近似求解逆向 SDE/ODE 实现单步更新，不同子类对应不同的数值求解方法（如欧拉-丸山法、祖先采样等）。
      此为抽象类，需继承并实现 `update_fn` 方法后使用。

  子类示例:
      - EulerMaruyamaPredictor: 欧拉-丸山方法求解逆向 SDE
      - ReverseDiffusionPredictor: 通用逆向扩散求解器
      - AncestralSamplingPredictor: 祖先采样（针对特定 SDE 类型）

  关键参数:
      sde (sde_lib.SDE): 随机微分方程配置，描述正向扩散过程及逆过程参数
      score_fn (Callable): 分数函数，计算 ∇_x log p_t(x)（模型预测的梯度）
      probability_flow (bool): 是否启用概率流 ODE 模式（True 时用确定性ODE代替随机SDE）

  核心属性:
      rsde: 逆向 SDE/ODE 对象，通过 sde.reverse() 生成，包含逆向过程的漂移/扩散系数计算方法
  """

  def __init__(self, sde, score_fn, probability_flow=False):
    """初始化预测器公共参数

    Args:
        sde: 随机微分方程配置，需实现 reverse() 方法
        score_fn: 分数函数，签名需为 (x, t) -> Tensor
        probability_flow: 是否将逆向 SDE 转换为 ODE（启用确定性采样）
    """
    super().__init__()
    self.sde = sde
    # 生成逆向 SDE/ODE 对象（根据 probability_flow 切换模式）
    self.rsde = sde.reverse(score_fn, probability_flow)  # 核心：定义逆向过程
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """执行一步预测器更新（抽象方法，子类需实现具体逻辑）

    输入:
        x (Tensor): 当前状态张量，形状为 [B, ...]（如 [B, C, H, W]）
        t (Tensor): 当前时间步张量，形状为 [B]，值范围通常为 [0, T]

    返回:
        x (Tensor): 更新后的状态（含噪声，形状与输入一致）
        x_mean (Tensor): 去噪后的均值（不含噪声，可用于评估生成质量）

    典型实现:
        - 欧拉法：x_{t-Δt} = x + drift*Δt + diffusion*sqrt(Δt)*noise
        - 祖先采样：x_{t-1} = (x + β*score)/sqrt(1-β) + sqrt(β)*noise
        - 逆向扩散：基于 rsde.discretize() 计算漂移和扩散项
    """
    pass

#"校正器抽象基类，定义所有校正器的统一接口。
class Corrector(abc.ABC):
  """校正器抽象基类，定义所有校正器的统一接口。

  功能说明:
      校正器用于扩散模型的逆过程（生成过程），负责对预测器（Predictor）的输出进行细化调整。
      典型的校正策略包括：基于分数梯度多步迭代去噪（如朗之万动力学）、噪声注入调整等。
      此为抽象类，需继承并实现 `update_fn` 方法后使用。

  子类示例:
      - LangevinCorrector: 通过朗之万动力学迭代校正
      - NoneCorrector: 空操作校正器（用于禁用校正步骤）
      - AnnealedLangevinDynamics: 退火朗之万校正

  关键参数:
      sde (sde_lib.SDE): 随机微分方程配置，描述正向扩散过程及逆过程参数
      score_fn (Callable): 分数函数，计算 ∇_x log p_t(x)（模型预测的梯度）
      snr (float): 信噪比参数，控制校正步长（影响梯度与噪声的平衡）
      n_steps (int): 每个时间步内的校正迭代次数（如朗之万步数）

  抽象方法:
      update_fn: 子类必须实现的具体校正逻辑
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    """初始化校正器公共参数

    Args:
        sde: 随机微分方程配置，需实现边际概率等关键方法
        score_fn: 分数函数，签名需为 (x, t) -> Tensor
        snr: 信噪比阈值，用于动态调整步长（值越大，梯度项的权重越高）
        n_steps: 单时间步内的校正迭代次数（如朗之万动力学步数）
    """
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  #def update_fn(self, x, t):
  def update_fn(self, x, t, y=None, discrete_sigmas=None):
    """执行一步校正操作（抽象方法，子类需实现具体逻辑）

    输入:
        x (Tensor): 当前状态张量，形状为 [B, ...]（如 [B, C, H, W]）
        t (Tensor): 当前时间步张量，形状为 [B]，值范围通常为 [0, T]

    返回:
        x (Tensor): 校正后的状态（含噪声）
        x_mean (Tensor): 校正后的去噪均值（不含噪声，用于评估或中间结果）

    典型实现:
        - 基于分数梯度迭代调整状态（如朗之万动力学）
        - 多步噪声注入和去噪操作
    """
    pass


@register_predictor(name='euler_maruyama')  # 注册为欧拉-丸山预测器
class EulerMaruyamaPredictor(Predictor):
  """欧拉-丸山方法预测器，用于求解逆向 SDE 的数值解

  核心原理:
      通过欧拉-丸山离散化方法近似逆向 SDE：
      dx = drift(x,t)dt + diffusion(t)dw
      更新公式：
      x_{t-Δt} = x_t + drift(x,t)·Δt + diffusion(t)·√|Δt|·ε

      特别说明:
      - 时间步进方向为反向（Δt < 0），因此 Δt = -1/N
      - 适用于连续型 SDE（如 VESDE、VPSDE）

  适用场景:
      通用型 SDE 逆向求解，支持概率流（deterministic ODE）和随机采样两种模式
  """

  def __init__(self, sde, score_fn, probability_flow=False):
    """初始化欧拉-丸山预测器

    Args:
        sde (sde_lib.SDE): 需实现 rsde（逆向 SDE）的 sde 方法
        score_fn (Callable): 分数函数 ∇_x log p_t(x)
        probability_flow (bool): 是否启用概率流 ODE 模式（True 时扩散项置零）
    """
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    """执行欧拉-丸山更新步骤

    数学推导:
        逆向 SDE 离散化为：
        Δx = drift(x,t)·Δt + diffusion(t)·√|Δt|·ε
        其中：
        - Δt = -1/N （负时间步长，逆向模拟）
        - ε ~ N(0,I)

    流程:
        1. 计算时间步长 Δt = -1/N
        2. 通过 rsde.sde() 获取当前漂移项 drift 和扩散项 diffusion
        3. 计算确定性更新 x_mean = x + drift·Δt
        4. 添加随机项 diffusion·√|Δt|·ε
        5. 返回更新后的 x 和 x_mean（去噪均值）

    Args:
        x (Tensor): 当前状态张量，形状 [B, C, H, W]
        t (Tensor): 当前时间步（标准化为 [0,1]），形状 [B]

    Returns:
        x (Tensor): 更新后的状态（含噪声）
        x_mean (Tensor): 去噪后的均值（无噪声版本）
    """
    # 计算时间步长（逆向过程，Δt 为负值）
    dt = -1. / self.rsde.N  # Δt = -1/N，其中 N 为总离散步数

    # 生成标准高斯噪声
    z = torch.randn_like(x)

    # 获取逆向 SDE 的漂移项和扩散项（由 rsde.sde 方法计算）
    drift, diffusion = self.rsde.sde(x, t)
    # 当 probability_flow=True 时，diffusion 会被置零，退化为 ODE

    # 计算确定性部分（均值预测）
    x_mean = x + drift * dt  # Δx_deterministic = drift·Δt

    # 计算扩散项系数 √|Δt|（因 dt 为负，取绝对值）
    diffusion_coeff = np.sqrt(-dt)  # 等价于 sqrt(|Δt|)

    # 更新状态（添加随机项）
    x = x_mean + diffusion[:, None, None, None] * diffusion_coeff * z

    return x, x_mean


@register_predictor(name='reverse_diffusion')  # 注册为逆向扩散预测器
class ReverseDiffusionPredictor(Predictor):
  """逆向扩散预测器，基于逆向 SDE 的离散化形式更新样本

  核心原理:
      通过逆向 SDE 的离散化形式计算漂移项 f 和扩散项 G，执行反向扩散步骤：
      x_{t-Δt} = x_t - f + G·ε，其中 ε ~ N(0,I)
      该更新规则适用于通用 SDE 的逆向过程

  适用场景:
      与任意实现 `rsde.discretize()` 方法的 SDE 兼容
      常用于 VP/VE/subVP 等 SDE 类型的反向扩散采样
  """

  def __init__(self, sde, score_fn, probability_flow=False):
    """初始化逆向扩散预测器

    Args:
        sde (sde_lib.SDE): 需实现 rsde（逆向 SDE）的 discretize 方法
        score_fn (Callable): 分数函数 ∇_x log p_t(x)
        probability_flow (bool): 是否启用概率流 ODE 模式（影响 discretize 计算）
    """
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    """执行逆向扩散更新步骤

    数学形式:
        dx = -f(x,t) dt + G(t) dw （逆向 SDE 的离散近似）
        更新公式：
        x_mean = x - f(x,t)  # 去漂移项
        x = x_mean + G(t) * ε  # 添加反向扩散噪声

    流程:
        1. 离散化逆向 SDE 得到漂移项 f 和扩散项 G
        2. 计算无噪声更新 x_mean = x - f
        3. 添加噪声项 G·ε 得到新状态 x

    Args:
        x (Tensor): 当前状态张量，形状 [B, C, H, W]
        t (Tensor): 当前时间步，形状 [B]

    Returns:
        x (Tensor): 更新后的状态
        x_mean (Tensor): 去噪后的均值（无噪声版本）
    """
    # 离散化逆向 SDE 获得漂移项 f 和扩散项 G（形状均为 [B, ...]）
    f, G = self.rsde.discretize(x, t)  # f: 漂移项，G: 扩散项系数

    # 生成标准高斯噪声
    z = torch.randn_like(x)

    # 计算去噪后的均值（不含噪声项）
    x_mean = x - f  # 逆向过程的确定性部分

    # 添加噪声项（注意 G 的维度广播，假设 G 的形状为 [B]）
    x = x_mean + G[:, None, None, None] * z  # 扩散项作用

    return x, x_mean


@register_predictor(name='ancestral_sampling')  # 注册为祖先采样预测器
class AncestralSamplingPredictor(Predictor):
  """祖先采样预测器，适用于 VE/VP 型 SDE

  功能说明:
      实现离散时间步的祖先采样策略，通过逆向过程逐步去噪生成样本。
      核心公式：x_{t-1} = mean_model(x_t, t) + σ * ε，其中：
      - mean_model 由分数函数推导而来
      - σ 由 SDE 的噪声调度参数决定

      支持 SDE 类型：
      - VESDE（方差爆炸型 SDE）
      - VPSDE（方差保持型 SDE）

      限制：
      - 不支持概率流 ODE（probability_flow 必须为 False）
      - 仅适用于离散时间步调度
  """

  def __init__(self, sde, score_fn, probability_flow=False):
    """初始化祖先采样器

    Args:
        sde (sde_lib.VPSDE 或 sde_lib.VESDE): SDE 配置对象
        score_fn (Callable): 分数函数 ∇_x log p_t(x)
        probability_flow (bool): 必须为 False（本类不支持 ODE 求解）
    """
    super().__init__(sde, score_fn, probability_flow)
    # 验证 SDE 类型
    if not isinstance(sde, (sde_lib.VPSDE, sde_lib.VESDE)):
      raise NotImplementedError(f"不支持 {sde.__class__.__name__} 类型的 SDE")
    # 祖先采样仅支持 SDE 模式（不支持概率流 ODE）
    assert not probability_flow, "祖先采样不支持概率流模式"

  def vesde_update_fn(self, x, t):
    """VESDE 的祖先采样更新规则

    数学推导:
        基于离散时间步的方差爆炸 SDE，更新公式：
        x_{t-1} = x_t + (σ_t^2 - σ_{t-1}^2) * score + √[ (σ_{t-1}^2 (σ_t^2 - σ_{t-1}^2)) / σ_t^2 ] * ε
        其中 ε ~ N(0, I)

    流程:
        1. 将连续时间 t 映射到离散时间步索引
        2. 获取当前 σ_t 和前一步的 σ_{t-1}
        3. 计算均值偏移量 (σ_t^2 - σ_{t-1}^2) * score
        4. 计算噪声项的标准差 std
        5. 采样并更新 x
    """
    sde = self.sde
    # 将 t ∈ [0,1] 映射到离散索引 [0, N-1]
    timestep = (t * (sde.N - 1) / sde.T).long()
    # 获取当前时间步的噪声级别 σ_t [B]
    sigma = sde.discrete_sigmas[timestep]
    # 获取前一时间步的噪声级别 σ_{t-1}，若当前为第一步则设为 0
    adjacent_sigma = torch.where(
      timestep == 0,
      torch.zeros_like(t),
      sde.discrete_sigmas.to(t.device)[timestep - 1]
    )
    score = self.score_fn(x, t)  # 计算分数 ∇log p(x) [B, C, H, W]

    # 计算均值偏移量：(σ_t^2 - σ_{t-1}^2) * score
    x_mean = x + (sigma  **  2 - adjacent_sigma  **  2)[:, None, None, None] * score

    # 计算噪声项标准差：√[ (σ_{t-1}^2 (σ_t^2 - σ_{t-1}^2)) / σ_t^2 ]
    std = torch.sqrt((adjacent_sigma  **  2 * (sigma  **  2 - adjacent_sigma  **  2)) / (sigma  **  2))

    # 生成噪声并更新 x
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise  # 广播 std 到 x 的维度

    return x, x_mean

  def vpsde_update_fn(self, x, t):
    """VPSDE 的祖先采样更新规则

    数学推导:
        基于离散时间步的方差保持 SDE，更新公式：
        x_{t-1} = (x_t + β_t * score) / √(1 - β_t) + √β_t * ε
        其中 ε ~ N(0, I)，β_t 为噪声调度参数

    流程:
        1. 将连续时间 t 映射到离散时间步索引
        2. 获取当前时间步的 β_t
        3. 计算均值项 (x_t + β_t * score) / √(1 - β_t)
        4. 添加噪声项 √β_t * ε
    """
    sde = self.sde
    # 将 t ∈ [0,1] 映射到离散索引 [0, N-1]
    timestep = (t * (sde.N - 1) / sde.T).long()
    # 获取当前时间步的 β_t [B]
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)  # 计算分数 ∇log p(x) [B, C, H, W]

    # 计算均值项：(x + β * score) / √(1 - β)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]

    # 生成噪声并更新 x
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise  # 广播 beta 到 x 的维度

    return x, x_mean

  def update_fn(self, x, t):
    """根据 SDE 类型分发更新"""
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')  # 注册为预测器，名称 'none'（表示空操作）
class NonePredictor(Predictor):
  """空预测器，不执行任何更新操作

  功能说明:
      在采样流程中跳过预测器步骤，保持输入状态不变。主要用于以下场景：
      - 纯校正器（Corrector-only）采样流程（如某些 Langevin 动力学变体）
      - 需要禁用预测步骤的实验性配置
      - 作为代码框架中的占位符，保持接口统一性

  设计意义:
      与 `NoneCorrector` 对称，允许通过配置名称灵活组合预测-校正流程
      例如：
      - predictor='none', corrector='langevin' → 纯朗之万采样
      - predictor='euler', corrector='none' → 纯预测器采样（如 DDIM）
  """

  def __init__(self, sde, score_fn, probability_flow=False):
    """初始化空预测器（参数仅用于接口兼容）

    Args:
        sde (sde_lib.SDE): 随机微分方程配置（未使用）
        score_fn (Callable): 分数函数（未使用）
        probability_flow (bool): 是否用概率流 ODE（未使用）
    """
    pass  # 显式声明无初始化操作

  def update_fn(self, x, t):
    """空更新函数：返回原始状态和均值

    Args:
        x (Tensor): 当前状态张量，形状为 [batch_size, ...]
        t (Tensor): 当前时间步张量，形状为 [batch_size]

    Returns:
        x (Tensor): 未经修改的输入状态（与输入相同）
        x (Tensor): 伪均值（直接复制输入，保持接口兼容性）

    注：返回两个相同张量是为了与其他预测器接口对齐（如 EulerMaruyamaPredictor）
    """
    return x, x  # 维持输入不变，模拟“无预测”操作


@register_corrector(name='langevin')  # 注册为校正器，名称 'langevin'
class LangevinCorrector(Corrector):
  """标准朗之万动力学校正器，用于扩散模型采样中的校正步骤

  功能说明:
      通过多步朗之万动力学迭代修正样本，更新规则为：
      x_{k+1} = x_k + ε·∇log p(x_k) + √(2ε)·噪声
      步长 ε 自适应调整以维持目标信噪比（SNR）

  典型应用:
      与预测器（Predictor）结合使用，构成 PC Sampler（Predictor-Corrector）
      适用于 VE、VP、subVP 等 SDE 类型
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    """初始化朗之万校正器

    Args:
        sde (sde_lib.SDE): 必须为 VPSDE/VESDE/subVPSDE 类型（需要实现边际概率方法）
        score_fn (Callable): 分数函数 ∇_x log p_t(x)
        snr (float): 目标信噪比，控制步长自适应调整
        n_steps (int): 每个时间步的朗之万迭代次数
    """
    super().__init__(sde, score_fn, snr, n_steps)
    # 验证 SDE 类型是否支持（需要 marginal_prob 方法）
    if not isinstance(sde, (sde_lib.VPSDE, sde_lib.VESDE, sde_lib.subVPSDE)):
      raise NotImplementedError(f"SDE 类型 {sde.__class__.__name__} 暂不支持")

  def update_fn(self, x, t,y=None, discrete_sigmas=None):
    """执行多步朗之万动力学更新

    流程:
        1. 根据 SDE 类型获取退火因子 alpha（VP/subVP 使用预设调度，VE 为1）
        2. 循环 n_steps 次：
            a. 计算当前梯度 grad = ∇log p(x)
            b. 计算梯度与噪声的范数比，自适应调整步长 step_size
            c. 更新 x_mean = x + step_size * grad
            d. 添加噪声 x = x_mean + sqrt(2*step_size) * noise

    Args:
        x (Tensor): 当前状态张量，形状 [B, C, H, W]
        t (Tensor): 当前时间步（标准化为 [0,1]），形状 [B]

    Returns:
        x (Tensor): 修正后的状态
        x_mean (Tensor): 去噪后的均值（无噪声版本的估计）
    """
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr

    # 获取退火因子 alpha（VP/subVP 使用预设值，VE 为1）
    if isinstance(sde, (sde_lib.VPSDE, sde_lib.subVPSDE)):
      timestep = (t * (sde.N - 1) / sde.T).long()  # 将连续时间映射到离散索引
      alpha = sde.alphas.to(t.device)[timestep]  # [B]
    else:  # VESDE
      alpha = torch.ones_like(t)  # [B]

    # 多步朗之万迭代
    for _ in range(n_steps):
      grad = score_fn(x, t)  # 计算分数 ∇log p(x) [B, C, H, W]
      noise = torch.randn_like(x)  # 生成标准高斯噪声 [B, C, H, W]

      # 计算梯度与噪声的 L2 范数（按样本平均）
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()  # 标量
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()  # 标量

      # 自适应步长计算（基于目标 SNR 的公式）
      step_size = (target_snr * noise_norm / grad_norm)  **  2 * 2 * alpha  # [B]

      # 更新均值（无噪声版本）
      x_mean = x + step_size[:, None, None, None] * grad  # 广播 step_size 到与 x 同维度

      # 添加噪声项
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean

@register_corrector(name='langevincs')
class LangevinCorrectorCS(Corrector):
  """改进的朗之万动力学校正器，用于求解条件分布 p(x|y)（如基于测量的图像重建）

  功能说明:
      在基础朗之万更新基础上，引入测量项 y 的条件似然梯度，执行基于测量的条件采样。
      更新公式：x ← x + ε*(score + grad_likelihood) + √(2ε)*噪声
      其中 grad_likelihood = (x - y)/σ^2，σ 为测量噪声水平

  典型应用:
      压缩感知（CS, Compressive Sensing）重建
      图像修复（Inpainting）、超分辨率（Super-Resolution）等逆问题求解

  设计差异（对比原版 LangevinCorrector）：
      - 添加测量项 y 的似然梯度
      - 自适应步长控制（基于梯度与噪声的范数比例）
      - 使用离散化的噪声水平调度（discrete_sigmas）
  """

  def __init__(self, sde, score_fn, snr, n_steps, sigma_min, sigma_max, nn):
    """初始化条件采样校正器

    Args:
        sde (sde_lib.VESDE): 必须为 VESDE 类型（确保噪声调度兼容）
        score_fn (Callable): 分数函数 ∇_x log p_t(x)（先验项）
        snr (float): 目标信噪比，控制步长大小
        n_steps (int): 每个时间步的朗之万迭代次数
        sigma_min/max (float): 测量噪声水平的最小/最大值（对数均匀采样）
        N (int): 离散噪声级别数（对应时间步数）
    """
    super().__init__(sde, score_fn, snr, n_steps)
    self.N = nn
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    # 在对数空间生成离散的噪声级别（用于不同时间步的测量噪声）
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), nn))
    # 目前仅支持 VESDE（与离散噪声调度兼容性相关）
    if not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE 类型 {sde.__class__.__name__} 暂不支持")

  def update_fn(self, x, t, y, discrete_sigmas):
    """执行条件朗之万更新（结合先验分数与测量似然）

    流程:
        1. 根据时间步 t 获取当前测量噪声水平 sigma
        2. 计算先验梯度 grad = ∇log p(x) 和似然梯度 grad_likelihood = (x - y)/sigma^2
        3. 自适应计算步长 step_size，使得 SNR ≈ target_snr
        4. 更新 x：x = x + step_size*(grad + grad_likelihood) + 噪声

    Args:
        x (Tensor): 当前估计值 x_i，形状 [B, C, H, W]
        t (Tensor): 当前时间步（标准化为 [0,1]），形状 [B]
        y (Tensor): 测量数据（如低分辨率图像），形状与 x 相同
        discrete_sigmas (Tensor): 预计算的离散噪声级别（用于索引）

    Returns:
        x (Tensor): 更新后的状态
        x_mean (Tensor): 去噪后的均值（无噪声版本的估计）
    """
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr

    # 退火因子（VESDE 下为1，保持接口兼容）
    if isinstance(sde, (sde_lib.VPSDE, sde_lib.subVPSDE)):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    # 多步朗之万迭代
    for _ in range(n_steps):
      # 似然梯度：假设测量模型 y = x + N(0, sigma^2I)

      # 根据时间步索引当前测量噪声 sigma
      timestep = (t * (self.N - 1) / 1).long()  # 将 t ∈ [0,1] 映射到 [0, N-1]
      sigma = self.discrete_sigmas.to(t.device)[timestep]  # [B]

      # 先验梯度（来自分数模型）
      #grad_likelihood = (x - y) / (sigma  **  2 + 1e-6)  # (x - y)/sigma^2
      #x = x + grad_likelihood
      grad = score_fn(x, t)  # ∇log p(x) [B, C, H, W]

      # 生成标准高斯噪声
      noise = torch.randn_like(x)

      # 自适应步长计算（基于信噪比）
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()  # 平均梯度范数
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()  # 平均噪声范数
      step_size = (target_snr * noise_norm / grad_norm)  **  2 * 2 * alpha  # [B]
      # 更新均值（先验 + 似然）
      #x_mean = x + step_size[:, None, None, None] * (grad) + grad_likelihood
      x_mean = x + step_size[:, None, None, None] * (grad)
      # 添加噪声项
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise 
    return x, x_mean


@register_corrector(name='ald')  # 注册为校正器，名称 'ald' (Annealed Langevin Dynamics)
class AnnealedLangevinDynamics(Corrector):
  """原始退火朗之万动力学校正器，来自 NCSN/NCSNv2 系列工作

  功能说明:
      通过多步朗之万动力学迭代修正样本，逐步降低噪声强度（退火）。
      每次更新步骤：x ← x + ε·∇log p(x) + √(2ε)·噪声

  典型应用:
      配合噪声条件分数网络（NCSN）使用，用于生成高质量样本
      在连续SDE框架中可作为校正步骤（如 PC Sampling 的 Corrector）

  注意:
      作者提到该校正器未被其论文直接使用，此处仅为完整性保留
  """

  def __init__(self, sde, score_fn, snr, n_steps,sigma_min=0.1, sigma_max=1, nn=1):
    """初始化退火朗之万校正器

    Args:
        sde (sde_lib.SDE): 必须为 VPSDE/VESDE/subVPSDE 类型（因需要特定噪声调度）
        score_fn (Callable): 分数函数 ∇_x log p_t(x)
        snr (float): 信噪比参数，控制更新步长（目标信噪比）
        n_steps (int): 每个时间步的朗之万迭代次数
    """
    super().__init__(sde, score_fn, snr, n_steps)
    # 验证 SDE 类型（需要实现特定方法如 marginal_prob）
    if not isinstance(sde, (sde_lib.VPSDE, sde_lib.VESDE, sde_lib.subVPSDE)):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} 暂不支持")

  def update_fn(self, x, t, y=None, discrete_sigmas=None):
    """执行多步朗之万动力学更新

    流程:
        1. 根据 SDE 类型计算退火参数 alpha
        2. 计算当前噪声尺度 std
        3. 循环 n_steps 次：
            a. 计算分数梯度 grad = ∇log p(x)
            b. 计算更新步长 step_size = (SNR * std)^2 * 2 * alpha
            c. 更新 x_mean = x + step_size * grad
            d. 添加噪声 x = x_mean + sqrt(2*step_size) * noise

    Args:
        x (Tensor): 当前状态张量，形状 [B, C, H, W]
        t (Tensor): 当前时间步，形状 [B]

    Returns:
        x (Tensor): 修正后的状态
        x_mean (Tensor): 去噪后的均值（无噪声版本的估计）
    """
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr  # 目标信噪比（控制步长）

    # 计算退火因子 alpha（不同 SDE 类型处理不同）
    if isinstance(sde, (sde_lib.VPSDE, sde_lib.subVPSDE)):
      # 将连续时间 t 离散化为 [0, N-1] 的整数索引
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]  # 获取预设的 alpha 调度值
    else:  # VESDE 类型不需要 alpha 调度
      alpha = torch.ones_like(t)

    # 计算当前时刻的噪声标准差（边际概率的标准差）
    _, std = sde.marginal_prob(x, t)

    # 多步朗之万迭代
    for _ in range(n_steps):
      grad = score_fn(x, t)  # ∇log p_t(x) [B, C, H, W]
      noise = torch.randn_like(x)  # 标准高斯噪声

      # 计算步长 (基于信噪比公式 SNR = |step * grad| / |noise| )
      step_size = (target_snr * std)  **  2 * 2 * alpha  # [B]

      # 更新均值（无噪声版本）
      x_mean = x + step_size[:, None, None, None] * grad  # 广播维度

      # 添加噪声项（朗之万动力学的随机性）
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')  # 将此类注册为校正器，命名为 'none'（用于配置采样流程）
class NoneCorrector(Corrector):
  """An empty corrector that does nothing.

  一个空的校正器，不执行任何更新操作。通常用于：
  - 纯预测器（Predictor-only）的采样流程（如 DDIM）
  - 需要跳过校正步骤的场景（例如某些快速采样模式）
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    """初始化空校正器（参数仅用于接口兼容，无实际操作）

    Args:
        sde (sde_lib.SDE): 随机微分方程配置（此处未使用）
        score_fn (Callable): 分数函数 ∇_x log p_t(x)（此处未使用）
        snr (float): 信噪比参数（用于调节校正强度，此处未使用）
        n_steps (int): 校正步数（此处未使用）
    """
    pass  # 显式声明无需初始化操作

  def update_fn(self, x, t, y=None, discrete_sigmas=None):
    """空更新函数：返回原始状态和均值（无校正操作）

    Args:
        x (Tensor): 当前状态张量，形状为 [batch_size, ...]
        t (Tensor): 当前时间步张量，形状为 [batch_size]

    Returns:
        x (Tensor): 未经修改的输入状态（与输入相同）
        x (Tensor): 伪均值（此处直接复制输入，无实际意义）

    注：返回两个相同张量是为了与其他校正器接口兼容（如 LangevinCorrector）
    """
    return x, x  # 保持输入不变，模拟“无校正”操作

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors.

  该函数封装了预测器（Predictor）的配置过程，返回其更新函数。主要用于扩散模型的反向过程生成步骤。

  Args:
      x (Tensor): 当前状态张量，形状为 [batch_size, ...]（如图像的 [B, C, H, W]）
      t (Tensor): 当前时间步张量，形状为 [batch_size]
      sde (sde_lib.SDE): 随机微分方程（SDE）的配置对象，定义前向扩散过程
      model (torch.nn.Module): 训练好的分数模型（score model），用于估计 ∇_x log p_t(x)
      predictor (Predictor): 预测器类型（如 EulerMaruyamaPredictor），若为 None 表示仅使用校正器
      probability_flow (bool): 是否使用概率流方法（若为 True，则用确定性ODE近似SDE，减少方差）
      continuous (bool): 是否使用连续时间参数化（影响分数计算方式）

  Returns:
      update_fn (Callable): 预测器的更新函数，用于执行一步状态更新 x_{t-Δt} ← x_t

  流程说明:
      1. 获取分数函数 score_fn: 根据 SDE 和模型计算 ∇_x log p_t(x)
      2. 初始化预测器对象 predictor_obj: 根据参数选择具体预测器
      3. 返回该预测器的 update_fn 方法
  """
  # 获取分数函数：根据 SDE 和模型参数化方式（连续/离散）返回 ∇_x log p_t(x)
  score_fn = mutils.get_score_fn(
    sde,
    model,
    train=False,  # 推理模式（关闭 dropout 等）
    continuous=continuous  # 是否使用连续时间参数化
  )

  # 初始化预测器对象
  if predictor is None:
    # 若 predictor 为 None，使用空预测器（仅校正器时使用）
    predictor_obj = NonePredictor(
      sde,
      score_fn,
      probability_flow  # 是否用概率流方法（ODE/SDE）
    )
  else:
    # 根据给定的预测器类型（如 EulerMaruyamaPredictor）初始化
    predictor_obj = predictor(
      sde,
      score_fn,
      probability_flow
    )

  # 返回预测器的更新函数，该函数接受 (x, t) 并返回下一步状态 x_{t-Δt}
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps,
                               sigma_min=None, sigma_max=None, N=None,  discrete_sigmas=None,cs=False,y=None,):
  """配置并返回校正器（corrector）的更新函数（用于采样过程的迭代更新）

  核心功能：根据输入的校正器类型和参数，生成对应的状态更新函数

  Args:
      x (Tensor): 当前状态张量（例如去噪过程中的图像数据）
      t (Tensor): 当前时间步张量（已归一化到[0,1]）
      sde (SDE): 随机微分方程对象（定义正向/逆向扩散过程）
      model (nn.Module): 预训练的分数模型（score-based model）
      corrector (Class): 校正器类（如LangevinCorrector，None表示仅用预测器）
      continuous (bool): 是否使用连续时间参数化
      snr (float): 信噪比（控制校正器步长）
      n_steps (int): 校正器单次迭代的步数
      cs (bool): 是否使用分类器指导（classifier guidance）或其他条件机制
      sigma_min (float): 噪声标准差下限（仅当cs=True时生效）
      sigma_max (float): 噪声标准差上限（仅当cs=True时生效）
      N (int): 时间离散化步数（仅当cs=True时生效）
      y (Tensor): 条件信息（如分类标签，仅当cs=True时生效）
      discrete_sigmas (Tensor): 预计算的离散噪声级别（仅当cs=True时生效）

  Returns:
      function: 更新函数 fn，执行形式为 x_new = fn(x, t)
  """

  # 获取分数函数（score function）
  # score_fn的调用签名：score_fn(x, t, y=None)
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)

  # 如果没有校正器（仅用预测器）
  if corrector is None:
    # 创建空校正器对象（无实际校正操作）
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    # 获取基础更新函数（可能只是恒等变换或简单噪声添加）
    fn = corrector_obj.update_fn(x, t)

  # 使用校正器的情况
  else:
    # 条件机制分支（如分类器指导）
    if cs:
      # 创建带条件参数的校正器实例
      corrector_obj = corrector(sde, score_fn, snr, n_steps, sigma_min, sigma_max, N)
      # 获取带条件信息的更新函数
      fn = corrector_obj.update_fn(x, t, y, discrete_sigmas)
    # 普通校正器分支
    else:
      # 创建标准校正器实例
      corrector_obj = corrector(sde, score_fn, snr, n_steps)
      # 获取标准更新函数
      fn = corrector_obj.update_fn(x, t)

  return fn  # 返回用于状态更新的函数


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda',cs = False,
                   sigma_min=None,sigma_max=None,N=None,discrete_sigmas=None,ct_consist=True,geo=None,angles=None):
  """创建预测-校正（Predictor-Corrector，PC）采样器

  Args:
      sde (sde_lib.SDE): 定义正向随机微分方程（SDE）的对象
      shape (tuple): 生成样本的形状，如 (batch_size, channels, height, width)
      predictor (sampling.Predictor): 预测器算法对象（如欧拉方法）
      corrector (sampling.Corrector): 校正器算法对象（如Langevin动力学）
      inverse_scaler (function): 数据逆标准化函数，将模型输出还原到原始数据范围
      snr (float): 信噪比，用于校正器步长控制
      n_steps (int): 每个预测步后执行的校正步数，默认为1
      probability_flow (bool): 是否使用概率流ODE求解（替代SDE）
      continuous (bool): 指示是否使用连续时间训练的模型
      denoise (bool): 是否在最后一步执行去噪操作
      eps (float): 数值稳定性阈值，避免积分到t=0时的不稳定性
      device (str): 计算设备（'cuda'或'cpu'）
      cs(bool):是否有条件输入

  Returns:
      function: 返回一个采样函数，该函数返回样本和总函数评估次数
  """

  # 创建预测器和校正器的更新函数（通过部分函数固定共有参数）
  predictor_update_fn = functools.partial(
    shared_predictor_update_fn,
    sde=sde,
    predictor=predictor,
    probability_flow=probability_flow,
    continuous=continuous
  )
  if cs:
    corrector_update_fn = functools.partial(
      shared_corrector_update_fn,
      sde=sde,
      corrector=corrector,
      continuous=continuous,
      snr=snr,
      n_steps=n_steps,
      cs = cs,
      sigma_min = sigma_min,
      sigma_max = sigma_max,
      N = N,
      discrete_sigmas = discrete_sigmas,
    )
  else:
    corrector_update_fn = functools.partial(
      shared_corrector_update_fn,
      sde=sde,
      corrector=corrector,
      continuous=continuous,
      snr=snr,
      n_steps=n_steps,
      cs=cs,
    )


  def pc_sampler(model,y=None):
    """PC采样器核心逻辑

    Args:
        model (torch.nn.Module): 训练好的分数模型

    Returns:
        tuple: (生成的样本, 总函数评估次数)
    """
    with torch.no_grad():  # 禁用梯度计算
      # 初始化样本：从先验分布（高斯噪声）采样
      x = sde.prior_sampling(shape).to(device)
      # 生成时间步序列：从最大时间T到最小时间eps，均匀分成N步
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      # 初始化计时器（用于性能分析）
      time_corrector_tot = 0
      time_predictor_tot = 0
      #x = y
      # 反向扩散过程：从t=T到t=eps
      for i in tqdm(range(sde.N)) :
          t = timesteps[i]
          vec_t = torch.ones(shape[0], device=t.device) * t  # 创建时间向量

          # 校正阶段（Corrector Step）
          tic_corrector = time.time()
          x, x_mean = corrector_update_fn(x, vec_t, model=model,y=y)
          time_corrector_tot += time.time() - tic_corrector

          # 预测阶段（Predictor Step）
          tic_predictor = time.time()
          x, x_mean = predictor_update_fn(x, vec_t, model=model)
          time_predictor_tot += time.time() - tic_predictor


      # 打印各阶段平均耗时（调试用）
      print(f'校正器单步平均耗时: {time_corrector_tot / sde.N:.4f}秒')
      print(f'预测器单步平均耗时: {time_predictor_tot / sde.N:.4f}秒')

      # 返回处理后的样本：若启用去噪则返回x_mean，否则返回x
      # 总评估次数 = 总步数 × (校正步数 + 1预测步)
      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]  # returns only the drift term because diffusion = 0 for probability_flow

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler
