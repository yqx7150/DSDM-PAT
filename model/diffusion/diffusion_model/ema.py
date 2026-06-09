# Modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py

from __future__ import division
from __future__ import unicode_literals

import torch


# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
  """
  维护一组参数的(指数)移动平均值。
  通常用于在模型训练时跟踪参数的平滑版本，以减少训练波动并提升模型鲁棒性。
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    初始化指数移动平均类。

    Args:
        parameters: 可迭代的`torch.nn.Parameter`，通常来自`model.parameters()`
        decay: 指数衰减率，介于0和1之间。值越大，历史值的权重越高，当前更新影响越小
        use_num_updates: 是否根据参数更新次数动态调整衰减率。若为True，则在初始阶段衰减率会较小，逐渐增大到设定值
    """
    # 参数校验
    if decay < 0.0 or decay > 1.0:
      raise ValueError('衰减率必须在[0, 1]区间内')

    self.decay = decay  # 基础衰减率
    self.num_updates = 0 if use_num_updates else None  # 参数更新计数器（仅在use_num_updates=True时启用）

    # 初始化影子参数：存储参数的指数移动平均版本
    # 仅处理需要梯度的参数，使用detach()切断计算图，clone()创建数据副本
    self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]

    # 临时存储容器：用于保存模型原始参数（例如验证前保存，验证后恢复）
    self.collected_params = []

  def update(self, parameters):
    """
    更新影子参数。应在每次参数更新后调用（例如optimizer.step()之后）

    Args:
        parameters: 可迭代的`torch.nn.Parameter`，应与初始化时的参数顺序一致
    """
    decay = self.decay
    # 动态调整衰减率：在初始阶段逐步增加衰减率，使训练更稳定
    if self.num_updates is not None:
      self.num_updates += 1  # 更新计数器
      decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))  # 调整后的实际衰减率

    one_minus_decay = 1.0 - decay  # 当前更新权重

    # 仅处理需要梯度的参数
    parameters = [p for p in parameters if p.requires_grad]
    with torch.no_grad():  # 禁用梯度计算
      for s_param, param in zip(self.shadow_params, parameters):
        # 更新公式：s_param = decay * s_param + (1 - decay) * param
        # 等价于 s_param -= (1 - decay) * (s_param - param)
        s_param.sub_(one_minus_decay * (s_param - param))  # 原地更新影子参数

  def copy_to(self, parameters):
    """
    将影子参数复制到目标参数。用于将EMA参数应用到原模型（例如模型验证时）

    Args:
        parameters: 可迭代的`torch.nn.Parameter`，应与初始化参数顺序一致
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        # 将影子参数的值复制到原参数中
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    临时存储原始参数。通常在调用copy_to之前使用，以保存模型当前状态

    Args:
        parameters: 可迭代的`torch.nn.Parameter`
    """
    # 保存参数的当前状态（深拷贝）
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    恢复之前存储的参数。用于恢复模型到应用EMA之前的状态（例如验证结束后）

    Args:
        parameters: 可迭代的`torch.nn.Parameter`
    """
    # 将存储的原始参数复制回模型
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    """返回当前状态字典，用于模型保存"""
    return {
      'decay': self.decay,
      'num_updates': self.num_updates,
      'shadow_params': self.shadow_params  # 包含当前所有影子参数
    }

  def load_state_dict(self, state_dict):
    """从状态字典加载EMA状态，用于模型恢复"""
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']