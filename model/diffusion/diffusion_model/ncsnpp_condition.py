from . import utils
from . import layers
from . import normalization
from . import layerspp
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnpp_condition')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config  # 保存配置
        self.act = act = get_act(config)  # 获取激活函数
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))  # 注册噪声级别（sigmas）为缓冲区

        # 从配置中提取模型参数
        self.nf = nf = config['diffusion_model']['nf']  # 基础通道数
        ch_mult = config['diffusion_model']['ch_mult']  # 通道数倍增因子
        self.num_res_blocks = num_res_blocks = config['diffusion_model']['num_res_blocks']  # 每个分辨率的残差块数
        self.attn_resolutions = attn_resolutions = config['diffusion_model']['attn_resolutions']  # 使用注意力机制的分辨率
        dropout = config['diffusion_model']['dropout']  # dropout 概率
        resamp_with_conv = config['diffusion_model']['resamp_with_conv']  # 是否使用卷积进行上下采样
        self.num_resolutions = num_resolutions = len(ch_mult)  # 分辨率数量
        self.all_resolutions = all_resolutions = [config['diffusion_data']['image_size'] // (2 ** i) for i in range(num_resolutions)]  # 所有分辨率

        # 条件设置
        self.conditional = conditional = config['diffusion_model']['conditional']  # 是否使用噪声条件
        fir = config['diffusion_model']['fir']  # 是否使用 FIR 滤波器
        fir_kernel = config['diffusion_model']['fir_kernel']  # FIR 滤波器核
        self.skip_rescale = skip_rescale = config['diffusion_model']['skip_rescale']  # 是否对跳跃连接进行缩放
        self.resblock_type = resblock_type = config['diffusion_model']['resblock_type'].lower()  # 残差块类型
        self.progressive = progressive = config['diffusion_model']['progressive'].lower()  # 渐进式上采样类型
        self.progressive_input = progressive_input = config['diffusion_model']['progressive_input'].lower()  # 渐进式下采样类型
        self.embedding_type = embedding_type = config['diffusion_model']['embedding_type'].lower()  # 嵌入类型（Fourier 或 positional）
        init_scale = config['diffusion_model']['init_scale']  # 初始化缩放因子

        # 检查参数合法性
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']

        # 组合方法（用于渐进式采样）
        combine_method = config['diffusion_model']['progressive_combine'].lower()
        combiner = functools.partial(Combine, method=combine_method)

        # 模块列表
        modules = []

        # 时间步/噪声级别嵌入
        if embedding_type == 'fourier':
            # 高斯 Fourier 特征嵌入
            assert config['diffusion_training']['continuous'], "Fourier features are only used for continuous training."
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config['diffusion_model']['fourier_scale']
            ))
            embed_dim = 2 * nf  # Fourier 嵌入维度

        elif embedding_type == 'positional':
            embed_dim = nf  # 位置嵌入维度

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        # 条件嵌入
        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))  # 线性层
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)  # 初始化权重
            nn.init.zeros_(modules[-1].bias)  # 初始化偏置
            modules.append(nn.Linear(nf * 4, nf * 4))  # 另一个线性层
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        # 定义注意力块、上采样和下采样
        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                    with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        Downsample = functools.partial(layerspp.Downsample,
                                    with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        # 渐进式上采样
        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        # 渐进式下采样
        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        # 定义残差块类型
        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                        act=act,
                                        dropout=dropout,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        temb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                        act=act,
                                        dropout=dropout,
                                        fir=fir,
                                        fir_kernel=fir_kernel,
                                        init_scale=init_scale,
                                        skip_rescale=skip_rescale,
                                        temb_dim=nf * 4)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # 下采样模块
        channels = config['diffusion_data']['num_channels']  # 输入图像的通道数
        if progressive_input != 'none':
            input_pyramid_ch = channels  # 渐进式下采样的通道数

        modules.append(conv3x3(channels, nf))  # 初始卷积层
        hs_c = [nf]  # 保存每个分辨率的通道数

        in_ch = nf
        for i_level in range(num_resolutions):
            # 每个分辨率的残差块
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]  # 输出通道数
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))  # 添加残差块
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))  # 添加注意力块
                hs_c.append(in_ch)

            # 下采样
            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        # 中间模块
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # 上采样模块
        pyramid_ch = 0
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                        out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            # 渐进式上采样
            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            # 上采样
            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c  # 确保所有分辨率通道数已用完

        # 最终模块
        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                    num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)  # 将所有模块组合为 ModuleList

    def forward(self, x, time_cond,y=None):
        if y == None:
            ValueError("y=None")
        # 时间步/噪声级别嵌入
        modules = self.all_modules
        m_idx = 0
        if self.embedding_type == 'fourier':
            # 高斯 Fourier 特征嵌入
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # 正弦位置嵌入
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        # 条件嵌入
        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        # 数据归一化
        if not self.config['diffusion_data']['centered']:
            x = 2 * x - 1.  # 将输入数据从 [0, 1] 转换为 [-1, 1]

        # 下采样模块
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]  # 初始卷积
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # 每个分辨率的残差块
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            # 下采样
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                # 渐进式下采样
                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        # 中间模块
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # 上采样模块
        pyramid = None
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                tmp = hs.pop()
                h = modules[m_idx](torch.cat([h, tmp], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            # 渐进式上采样
            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            # 上采样
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs  # 确保所有分辨率通道数已用完

        # 最终输出
        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        # 根据噪声级别缩放输出
        if self.config['diffusion_model']['scale_by_sigma']:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h