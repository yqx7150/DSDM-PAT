# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    """
    初始化 Gaussian Fourier 投影模块。

    参数:
      embedding_size (int): 嵌入的维度大小，默认为 256。
      scale (float): 用于缩放随机权重的因子，默认为 1.0。
    """
    super().__init__()
    # 初始化一个随机权重矩阵 W，形状为 (embedding_size,)
    # W 是从标准正态分布中采样并乘以 scale 的随机值
    # requires_grad=False 表示 W 在训练过程中不会更新
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    """
    前向传播函数，生成 Gaussian Fourier 特征嵌入。

    参数:
      x (torch.Tensor): 输入张量，形状为 (batch_size,)，表示噪声级别或时间步。

    返回:
      torch.Tensor: 生成的 Fourier 特征嵌入，形状为 (batch_size, 2 * embedding_size)。
    """
    # 将输入 x 从形状 (batch_size,) 扩展为 (batch_size, 1)
    # 然后与权重 W 进行逐元素相乘，W 的形状为 (embedding_size,)
    # 结果 x_proj 的形状为 (batch_size, embedding_size)
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi

    # 对 x_proj 分别计算正弦和余弦值
    # torch.sin(x_proj) 和 torch.cos(x_proj) 的形状均为 (batch_size, embedding_size)
    # 最后将两者在最后一个维度上拼接，得到形状为 (batch_size, 2 * embedding_size) 的输出
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    """
    初始化自注意力块。

    参数:
      channels (int): 输入特征的通道数。
      skip_rescale (bool): 是否对跳跃连接的输出进行缩放（除以 sqrt(2)），默认为 False。
      init_scale (float): 初始化缩放因子，用于最后一个 NIN 层的权重初始化，默认为 0。
    """
    super().__init__()
    # GroupNorm 层，用于归一化输入特征
    # num_groups 设置为 min(channels // 4, 32)，确保分组数不超过 32
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)

    # 四个 NIN（Network In Network）层，用于生成查询（q）、键（k）、值（v）和输出
    self.NIN_0 = NIN(channels, channels)  # 生成查询 q
    self.NIN_1 = NIN(channels, channels)  # 生成键 k
    self.NIN_2 = NIN(channels, channels)  # 生成值 v
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)  # 输出层，带初始化缩放

    # 是否对跳跃连接的输出进行缩放
    self.skip_rescale = skip_rescale

  def forward(self, x):
    """
    前向传播函数。

    参数:
      x (torch.Tensor): 输入张量，形状为 (B, C, H, W)，其中
        B 是 batch size，C 是通道数，H 和 W 是高度和宽度。

    返回:
      torch.Tensor: 输出张量，形状与输入相同。
    """
    B, C, H, W = x.shape  # 获取输入张量的形状

    # 1. 归一化输入特征
    h = self.GroupNorm_0(x)

    # 2. 生成查询（q）、键（k）和值（v）
    q = self.NIN_0(h)  # 查询 q，形状为 (B, C, H, W)
    k = self.NIN_1(h)  # 键 k，形状为 (B, C, H, W)
    v = self.NIN_2(h)  # 值 v，形状为 (B, C, H, W)

    # 3. 计算注意力权重
    # 使用 Einstein 求和约定计算 q 和 k 的点积
    # torch.einsum('bchw,bcij->bhwij', q, k) 表示对 q 和 k 进行点积
    # 结果 w 的形状为 (B, H, W, H, W)
    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))  # 缩放因子 1/sqrt(C)

    # 4. 对注意力权重进行 softmax 归一化
    # 先将 w 的形状从 (B, H, W, H, W) 重塑为 (B, H, W, H * W)
    w = torch.reshape(w, (B, H, W, H * W))
    # 对最后一个维度（H * W）进行 softmax 归一化
    w = F.softmax(w, dim=-1)
    # 将 w 的形状恢复为 (B, H, W, H, W)
    w = torch.reshape(w, (B, H, W, H, W))

    # 5. 使用注意力权重对值 v 进行加权求和
    # torch.einsum('bhwij,bcij->bchw', w, v) 表示对 w 和 v 进行加权求和
    # 结果 h 的形状为 (B, C, H, W)
    h = torch.einsum('bhwij,bcij->bchw', w, v)

    # 6. 通过最后一个 NIN 层生成输出
    h = self.NIN_3(h)

    # 7. 跳跃连接
    if not self.skip_rescale:
      return x + h  # 直接相加
    else:
      return (x + h) / np.sqrt(2.)  # 缩放跳跃连接的输出


class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    """
    初始化上采样模块。

    参数:
      in_ch (int): 输入特征的通道数。如果未指定，则默认为 out_ch。
      out_ch (int): 输出特征的通道数。如果未指定，则默认为 in_ch。
      with_conv (bool): 是否在上采样后添加卷积层，默认为 False。
      fir (bool): 是否使用 FIR 滤波器进行上采样，默认为 False。
      fir_kernel (tuple): FIR 滤波器的核大小，默认为 (1, 3, 3, 1)。
    """
    super().__init__()
    # 如果未指定 out_ch，则默认与 in_ch 相同
    out_ch = out_ch if out_ch else in_ch

    # 如果不使用 FIR 滤波器
    if not fir:
      # 如果 with_conv 为 True，则添加一个 3x3 卷积层
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)  # 3x3 卷积层
    else:
      # 如果使用 FIR 滤波器且 with_conv 为 True，则添加一个自定义的卷积层
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, up=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())  # 自定义卷积层

    # 保存参数
    self.fir = fir  # 是否使用 FIR 滤波器
    self.with_conv = with_conv  # 是否添加卷积层
    self.fir_kernel = fir_kernel  # FIR 滤波器核
    self.out_ch = out_ch  # 输出通道数

  def forward(self, x):
    """
    前向传播函数。

    参数:
      x (torch.Tensor): 输入张量，形状为 (B, C, H, W)，其中
        B 是 batch size，C 是通道数，H 和 W 是高度和宽度。

    返回:
      torch.Tensor: 上采样后的输出张量，形状为 (B, out_ch, H * 2, W * 2)。
    """
    B, C, H, W = x.shape  # 获取输入张量的形状

    # 如果不使用 FIR 滤波器
    if not self.fir:
      # 使用最近邻插值进行上采样，将特征图的高和宽放大 2 倍
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      # 如果 with_conv 为 True，则在上采样后添加一个 3x3 卷积层
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      # 如果使用 FIR 滤波器
      if not self.with_conv:
        # 如果 with_conv 为 False，则直接使用 FIR 滤波器进行上采样
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        # 如果 with_conv 为 True，则使用自定义的卷积层进行上采样
        h = self.Conv2d_0(x)

    return h  # 返回上采样后的特征图


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    """
    初始化下采样模块。

    参数:
      in_ch (int): 输入特征的通道数。如果未指定，则默认为 out_ch。
      out_ch (int): 输出特征的通道数。如果未指定，则默认为 in_ch。
      with_conv (bool): 是否在下采样后添加卷积层，默认为 False。
      fir (bool): 是否使用 FIR 滤波器进行下采样，默认为 False。
      fir_kernel (tuple): FIR 滤波器的核大小，默认为 (1, 3, 3, 1)。
    """
    super().__init__()
    # 如果未指定 out_ch，则默认与 in_ch 相同
    out_ch = out_ch if out_ch else in_ch

    # 如果不使用 FIR 滤波器
    if not fir:
      # 如果 with_conv 为 True，则添加一个 3x3 卷积层，步幅为 2
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)  # 3x3 卷积层，步幅为 2
    else:
      # 如果使用 FIR 滤波器且 with_conv 为 True，则添加一个自定义的卷积层
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())  # 自定义卷积层

    # 保存参数
    self.fir = fir  # 是否使用 FIR 滤波器
    self.fir_kernel = fir_kernel  # FIR 滤波器核
    self.with_conv = with_conv  # 是否添加卷积层
    self.out_ch = out_ch  # 输出通道数

  def forward(self, x):
    """
    前向传播函数。

    参数:
      x (torch.Tensor): 输入张量，形状为 (B, C, H, W)，其中
        B 是 batch size，C 是通道数，H 和 W 是高度和宽度。

    返回:
      torch.Tensor: 下采样后的输出张量，形状为 (B, out_ch, H // 2, W // 2)。
    """
    B, C, H, W = x.shape  # 获取输入张量的形状

    # 如果不使用 FIR 滤波器
    if not self.fir:
      # 如果 with_conv 为 True，则使用卷积层进行下采样
      if self.with_conv:
        # 对输入进行填充，确保卷积后的尺寸正确
        x = F.pad(x, (0, 1, 0, 1))  # 在右侧和底部各填充 1 个像素
        x = self.Conv_0(x)  # 3x3 卷积层，步幅为 2
      else:
        # 如果 with_conv 为 False，则使用平均池化进行下采样
        x = F.avg_pool2d(x, 2, stride=2)  # 2x2 平均池化，步幅为 2
    else:
      # 如果使用 FIR 滤波器
      if not self.with_conv:
        # 如果 with_conv 为 False，则直接使用 FIR 滤波器进行下采样
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        # 如果 with_conv 为 True，则使用自定义的卷积层进行下采样
        x = self.Conv2d_0(x)

    return x  # 返回下采样后的特征图


class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    """
    初始化 DDPM 风格的残差块。

    参数:
      act: 激活函数。
      in_ch (int): 输入特征的通道数。
      out_ch (int): 输出特征的通道数。如果未指定，则默认为 in_ch。
      temb_dim (int): 时间嵌入的维度。如果未指定，则不使用时间嵌入。
      conv_shortcut (bool): 是否使用卷积作为快捷连接（shortcut），默认为 False。
      dropout (float): Dropout 概率，默认为 0.1。
      skip_rescale (bool): 是否对跳跃连接的输出进行缩放（除以 sqrt(2)），默认为 False。
      init_scale (float): 初始化缩放因子，用于卷积层的权重初始化，默认为 0。
    """
    super().__init__()
    out_ch = out_ch if out_ch else in_ch  # 如果未指定 out_ch，则默认为 in_ch

    # 第一个 GroupNorm 和 3x3 卷积层
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3(in_ch, out_ch)

    # 时间嵌入的全连接层
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)  # 初始化权重
      nn.init.zeros_(self.Dense_0.bias)  # 初始化偏置

    # 第二个 GroupNorm、Dropout 和 3x3 卷积层
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)

    # 如果输入通道数和输出通道数不同，则添加快捷连接
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)  # 使用 3x3 卷积作为快捷连接
      else:
        self.NIN_0 = NIN(in_ch, out_ch)  # 使用 NIN（Network In Network）作为快捷连接

    # 保存参数
    self.skip_rescale = skip_rescale  # 是否对跳跃连接的输出进行缩放
    self.act = act  # 激活函数
    self.out_ch = out_ch  # 输出通道数
    self.conv_shortcut = conv_shortcut  # 是否使用卷积作为快捷连接

  def forward(self, x, temb=None):
    """
    前向传播函数。

    参数:
      x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
      temb (torch.Tensor): 时间嵌入张量，形状为 (B, temb_dim)。

    返回:
      torch.Tensor: 输出张量，形状为 (B, out_ch, H, W)。
    """
    # 第一个 GroupNorm 和激活函数
    h = self.act(self.GroupNorm_0(x))
    # 第一个 3x3 卷积
    h = self.Conv_0(h)

    # 如果提供了时间嵌入，则将其加到特征上
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]  # 将时间嵌入广播到特征图的空间维度

    # 第二个 GroupNorm、激活函数和 Dropout
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    # 第二个 3x3 卷积
    h = self.Conv_1(h)

    # 如果输入通道数和输出通道数不同，则调整快捷连接的通道数
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)  # 使用 3x3 卷积调整通道数
      else:
        x = self.NIN_0(x)  # 使用 NIN 调整通道数

    # 跳跃连接
    if not self.skip_rescale:
      return x + h  # 直接相加
    else:
      return (x + h) / np.sqrt(2.)  # 缩放跳跃连接的输出


class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    """
    初始化 BigGAN 风格的残差块。

    参数:
      act: 激活函数。
      in_ch (int): 输入特征的通道数。
      out_ch (int): 输出特征的通道数。如果未指定，则默认为 in_ch。
      temb_dim (int): 时间嵌入的维度。如果未指定，则不使用时间嵌入。
      up (bool): 是否进行上采样，默认为 False。
      down (bool): 是否进行下采样，默认为 False。
      dropout (float): Dropout 概率，默认为 0.1。
      fir (bool): 是否使用 FIR 滤波器进行上/下采样，默认为 False。
      fir_kernel (tuple): FIR 滤波器的核大小，默认为 (1, 3, 3, 1)。
      skip_rescale (bool): 是否对跳跃连接的输出进行缩放（除以 sqrt(2)），默认为 True。
      init_scale (float): 初始化缩放因子，用于卷积层的权重初始化，默认为 0。
    """
    super().__init__()
    out_ch = out_ch if out_ch else in_ch  # 如果未指定 out_ch，则默认为 in_ch

    # 第一个 GroupNorm
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)

    # 上采样和下采样标志
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    # 第一个 3x3 卷积
    self.Conv_0 = conv3x3(in_ch, out_ch)

    # 时间嵌入的全连接层
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)  # 初始化权重
      nn.init.zeros_(self.Dense_0.bias)  # 初始化偏置

    # 第二个 GroupNorm、Dropout 和 3x3 卷积
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)

    # 如果输入通道数和输出通道数不同，或者需要进行上/下采样，则添加 1x1 卷积
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    # 保存参数
    self.skip_rescale = skip_rescale  # 是否对跳跃连接的输出进行缩放
    self.act = act  # 激活函数
    self.in_ch = in_ch  # 输入通道数
    self.out_ch = out_ch  # 输出通道数

  def forward(self, x, temb=None):
    """
    前向传播函数。

    参数:
      x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
      temb (torch.Tensor): 时间嵌入张量，形状为 (B, temb_dim)。

    返回:
      torch.Tensor: 输出张量，形状为 (B, out_ch, H', W')。
    """
    # 第一个 GroupNorm 和激活函数
    h = self.act(self.GroupNorm_0(x))

    # 上采样或下采样
    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)  # FIR 上采样
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)  # 最近邻上采样
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)  # FIR 下采样
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)  # 平均池化下采样
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    # 第一个 3x3 卷积
    h = self.Conv_0(h)

    # 如果提供了时间嵌入，则将其加到特征上
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]  # 将时间嵌入广播到特征图的空间维度

    # 第二个 GroupNorm、激活函数和 Dropout
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    # 第二个 3x3 卷积
    h = self.Conv_1(h)

    # 如果输入通道数和输出通道数不同，或者需要进行上/下采样，则调整快捷连接的通道数
    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)  # 使用 1x1 卷积调整通道数

    # 跳跃连接
    if not self.skip_rescale:
      return x + h  # 直接相加
    else:
      return (x + h) / np.sqrt(2.)  # 缩放跳跃连接的输出
