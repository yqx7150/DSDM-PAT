import math
import os
import random
from collections import namedtuple
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
from utilis.Nerf.Nerf_utils import get_psnr, get_ssim ,get_mse ,get_psnr_3d
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataset.RDDM.get_dataset import dataset,dataset_pickle
from einops import rearrange, reduce
from ema_pytorch import EMA
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])


def set_seed(SEED):
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

def normalize_to_neg_one_to_one(img,cfg):
    if cfg['diffusion_data']['norm_type'] == 'no_norm':
        return img
    else:
        if isinstance(img, list):
            return [img[k] * 2 - 1 for k in range(len(img))]
        else:
            return img * 2 - 1

def unnormalize_to_zero_to_one(img,cfg):
    if cfg['diffusion_data']['norm_type'] == 'no_norm' :
        return img
    else:
        if isinstance(img, list):
            return [(img[k] + 1) * 0.5 for k in range(len(img))]
        else:
            return (img + 1) * 0.5
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


class WeightStandardizedConv2d(nn.Conv2d):

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
        input_condition=False,
        img_to_img_translation=False
    ):
        super().__init__()

        # --- 1. 输入通道计算 ---
        self.channels = channels
        self.self_condition = self_condition

        # 计算实际输入通道数（考虑各种条件）
        # input_condition# 附加条件，condition主条件，self_condition自条件
        input_channels = channels + channels * \
            (1 if self_condition else 0) + channels * \
            (1 if condition and (not img_to_img_translation) else 0) + channels * (1 if input_condition else 0)

        # --- 2. 初始卷积层 ---
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        # --- 3. 构建下采样维度序列 --
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # --- 4. ResNet块定义 ---
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # --- 5. 时间嵌入处理 ---
        time_dim = dim * 4  # 时间嵌入维度
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        # 选择位置编码类型
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # 时间编码MLP
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # --- 6. 下采样路径 ---
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1) # 是否最后一层

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))


        # --- 7. 中间瓶颈层 ---
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)


        # --- 8. 上采样路径 ---
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))


        # --- 9. 输出层 ---
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        # 前向传播过程
        # 参数:
        #     x: 输入张量 [B,C,H,W]
        #     time: 时间步 [B,]
        #     x_self_cond: 自条件张量 [B,C,H,W] (可选)

        # --- 1. 自条件处理 ---
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1) # 通道维度拼接

        # --- 2. 初始卷积 ---
        x = self.init_conv(x)
        r = x.clone()

        # --- 3. 时间嵌入 ---
        t = self.time_mlp(time)

        # --- 4. 下采样路径 ---
        h = [] # 存储跳连特征
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # --- 5. 瓶颈层 ---
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        

        # --- 6. 上采样路径 ---
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)


        # --- 7. 输出处理 ---
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class UnetRes(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        ##获取配置
        dim=cfg['diffusion_model']['dim']
        channels = cfg['diffusion_data']['num_channels']
        condition=cfg['diffusion_training']['condition']
        input_condition=cfg['diffusion_training']['input_condition']
        objective=cfg['diffusion_training']['objective']
        test_res_or_noise=cfg['diffusion_training']['test_res_or_noise']
        img_to_img_translation=cfg['diffusion_data']['img_to_img_translation']
        self_condition=cfg['diffusion_model']['self_condition']
        resnet_block_groups=cfg['diffusion_model']['resnet_block_groups']
        learned_variance=cfg['diffusion_model']['learned_variance']
        learned_sinusoidal_cond=cfg['diffusion_model']['learned_sinusoidal_cond']
        random_fourier_features=cfg['diffusion_model']['random_fourier_features']
        learned_sinusoidal_dim=cfg['diffusion_model']['learned_sinusoidal_dim']
        num_unet=cfg['diffusion_model']['num_unet']
        init_dim=cfg['diffusion_model']['init_dim']
        out_dim=cfg['diffusion_model']['out_dim']
        dim_mults=tuple(cfg['diffusion_model']['dim_mults'])

        ##配置模型
        self.condition = condition
        self.input_condition = input_condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.self_condition = self_condition
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        self.img_to_img_translation = img_to_img_translation
        if self.num_unet == 2:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition,
                              img_to_img_translation=img_to_img_translation)
            self.unet1 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition,
                              img_to_img_translation=img_to_img_translation)
        elif self.num_unet == 1:
            self.unet0 = Unet(dim,
                              init_dim=init_dim,
                              out_dim=out_dim,
                              dim_mults=dim_mults,
                              channels=channels,
                              self_condition=self_condition,
                              resnet_block_groups=resnet_block_groups,
                              learned_variance=learned_variance,
                              learned_sinusoidal_cond=learned_sinusoidal_cond,
                              random_fourier_features=random_fourier_features,
                              learned_sinusoidal_dim=learned_sinusoidal_dim,
                              condition=condition,
                              input_condition=input_condition,
                              img_to_img_translation=img_to_img_translation)

    def forward(self, x, time, x_self_cond=None):
        if self.num_unet == 2:
            if self.test_res_or_noise == "res_noise":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), self.unet1(x, time[1], x_self_cond=x_self_cond)
            elif self.test_res_or_noise == "res":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), 0
            elif self.test_res_or_noise == "noise":
                return 0, self.unet1(x, time[1], x_self_cond=x_self_cond)
            if self.test_res_or_noise == "x0_noise":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), self.unet1(x, time[1], x_self_cond=x_self_cond)
            elif self.test_res_or_noise == "x0":
                return self.unet0(x, time[0], x_self_cond=x_self_cond), 0
            elif self.test_res_or_noise == "noise":
                return 0, self.unet1(x, time[1], x_self_cond=x_self_cond)
        elif self.num_unet == 1:
            if self.objective == 'pred_res_noise':
                # num_unet=2
                pass
            elif self.objective == 'pred_x0_noise':
                # num_unet=2
                pass
            elif self.objective == "pred_noise":
                time = time[1]
            elif self.objective == "pred_res":
                time = time[0]
            elif self.objective == "pred_x0":
                time = time[0]
            return [self.unet0(x, time, x_self_cond=x_self_cond)]



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    assert (alphas.sum()-1).abs() < 1e-6

    return alphas*sum_scale



def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class ResidualDiffusion(nn.Module):
    def __init__(self,model,cfg):
        super().__init__()

        ###获取配置
        image_size=cfg['diffusion_data']['image_size']  
        timesteps=cfg['diffusion_model']['num_scales']  
        sampling_timesteps=cfg['diffusion_model']['num_scales']   
        objective=cfg['diffusion_training']['objective']  
        condition=cfg['diffusion_training']['condition']  
        sum_scale=cfg['diffusion_training']['sum_scale']  
        input_condition=cfg['diffusion_training']['input_condition']
        input_condition_mask=cfg['diffusion_training']['input_condition_mask']  
        test_res_or_noise=cfg['diffusion_training']['test_res_or_noise'] 
        img_to_img_translation=cfg['diffusion_data']['img_to_img_translation']  
        loss_type = cfg['diffusion_model']['loss_type'] 
        ddim_sampling_eta = cfg['diffusion_model']['ddim_sampling_eta'] 

        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        ###配置模型
        self.cfg = cfg
        self.model = model 
        self.channels = self.model.channels  # 输入通道数
        self.self_condition = self.model.self_condition  # 自条件生成标志
        self.image_size = image_size  # 图像尺寸
        self.objective = objective  # 预测目标(res_noise/x0等)
        self.condition = condition  # 条件生成标志
        self.input_condition = input_condition  # 输入条件标志
        self.input_condition_mask = input_condition_mask  # 输入条件mask
        self.test_res_or_noise = test_res_or_noise  # 测试模式
        self.img_to_img_translation = img_to_img_translation  # 图像翻译任务标志

        # 残差缩放因子设置
        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
            ddim_sampling_eta = 0.
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        ##扩散系数生成
        convert_to_ddim = False
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
             # DDIM系数生成逻辑
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = 0
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = 0
        else:
            # 自定义系数生成
            alphas = gen_coefficients(timesteps, schedule="decreased")
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        #   注册缓冲区（固定参数）
        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6


    #初始化扩散系数（备用方法）
    def init(self):

        timesteps = 1000

        convert_to_ddim = True
        if convert_to_ddim:
            beta_schedule = "linear"
            beta_start = 0.0001
            beta_end = 0.02
            if beta_schedule == "linear":
                betas = torch.linspace(
                    beta_start, beta_end, timesteps, dtype=torch.float32)
            elif beta_schedule == "scaled_linear":
                # this schedule is very specific to the latent diffusion model.
                betas = (
                    torch.linspace(beta_start**0.5, beta_end**0.5,
                                   timesteps, dtype=torch.float32) ** 2
                )
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide cosine schedule
                betas = betas_for_alpha_bar(timesteps)
            else:
                raise NotImplementedError(
                    f"{beta_schedule} does is not implemented for {self.__class__}")

            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumsum = 1-alphas_cumprod ** 0.5
            betas2_cumsum = 1-alphas_cumprod

            alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
            betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
            alphas = alphas_cumsum-alphas_cumsum_prev
            alphas[0] = alphas[1]
            betas2 = betas2_cumsum-betas2_cumsum_prev
            betas2[0] = betas2[1]

        else:
            alphas = gen_coefficients(timesteps, schedule="average", ratio=1)
            betas2 = gen_coefficients(
                timesteps, schedule="increased", sum_scale=self.sum_scale, ratio=3)

            alphas_cumsum = alphas.cumsum(dim=0).clip(0, 1)
            betas2_cumsum = betas2.cumsum(dim=0).clip(0, 1)

            alphas_cumsum_prev = F.pad(
                alphas_cumsum[:-1], (1, 0), value=alphas_cumsum[1])
            betas2_cumsum_prev = F.pad(
                betas2_cumsum[:-1], (1, 0), value=betas2_cumsum[1])

        betas_cumsum = torch.sqrt(betas2_cumsum)
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        # 从预测残差计算噪声分量
        # 参数:
        #     x_t: 当前噪声图像
        #     t: 时间步
        #     x_input: 输入条件
        #     pred_res: 模型预测的残差
        # 返回:
        #     估计的噪声分量

        return (
            (x_t-x_input-(extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        #从输入条件和噪声预测起始图像
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        #从残差和噪声预测起始图像
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        #计算后验分布均值（从残差和噪声）
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        #计算后验分布参数
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=True):
        # 模型预测主函数，根据配置返回不同预测结果
        
        # 参数:
        #     x_input: 条件输入图像（如低分辨率图像/带掩码图像）
        #     x: 当前噪声图像x_t
        #     t: 当前时间步（控制噪声强度）
        #     x_input_condition: 额外的条件输入（默认0表示无）
        #     x_self_cond: 自条件输入（来自前一步的预测结果）
        #     clip_denoised: 是否将输出裁剪到[-1,1]范围（默认True）
        
        # 返回:
        #     ModelResPrediction对象，包含预测的残差/噪声/去噪结果
        
        # ==================== 1. 构建模型输入 ====================
        if not self.condition:
            # 无条件生成：仅使用噪声图像x作为输入
            x_in = x
        else:
            if self.img_to_img_translation:
                # 图像翻译任务（如图像超分辨率/风格迁移）
                if self.input_condition:
                    # 如果有额外条件输入，沿通道维度拼接
                    x_in = torch.cat((x, x_input_condition), dim=1)
                else:
                    x_in = x
            else:
                # 常规条件生成（如图像修复）
                if self.input_condition:
                    # 拼接噪声图像+条件图像+额外条件
                    x_in = torch.cat((x, x_input, x_input_condition), dim=1)
                else:
                    # 拼接噪声图像+条件图像
                    x_in = torch.cat((x, x_input), dim=1)

        # ==================== 2. 模型前向传播 ====================
        # 输入说明：
        #   x_in: 组合后的输入图像
        #   [alphas_cumsum[t], betas_cumsum[t]]: 时间步t对应的噪声调度系数
        #   x_self_cond: 自条件输入（可能为None）
        model_output = self.model(
            x_in,
            [self.alphas_cumsum[t] * self.num_timesteps,
            self.betas_cumsum[t] * self.num_timesteps],
            x_self_cond
        )

        # 定义裁剪函数（如需裁剪到[-1,1]范围）
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_denoised else identity


        # ==================== 3. 处理不同预测目标 ====================
        if self.objective == 'pred_res_noise':
            # 目标：同时预测残差(pred_res)和噪声(pred_noise)
            if self.test_res_or_noise == "res_noise":
                # 直接使用模型输出的两个通道
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res) # 裁剪残差预测
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise) # 噪声预测
                x_start = maybe_clip(x_start) # # 裁剪去噪结果


            elif self.test_res_or_noise == "res":
                # 仅使用残差预测，计算推导噪声
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res # 计算去噪结果
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                # 仅使用噪声预测，计算推导残差
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)

        elif self.objective == 'pred_x0_noise':
            # 目标：预测去噪图像(x0)和噪声
            if self.test_res_or_noise == "x0_noise":  # 通过x0计算残差
                pred_res = x_input-model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = maybe_clip(model_output[0]) # 直接使用模型输出的x0
            elif self.test_res_or_noise == "x0": 
                pred_res = x_input-model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = maybe_clip(model_output[0])
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_noise":
            # 目标：仅预测噪声
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            # 目标：仅预测残差
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)
        elif self.objective == "pred_x0":
            # 目标：仅预测去噪图像(x0)
            pred_res = x_input-model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        # ==================== 4. 返回结构化结果 ====================
        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t, x_input_condition=0, x_self_cond=None):

        # 计算反向扩散过程的后验分布参数（均值、方差）和预测的起始图像x_start
        # 参数:
        #     x_input: 条件输入图像（如低分辨率/带掩码的图像）
        #     x: 当前噪声图像x_t（需要去噪的图像）
        #     t: 当前时间步（控制噪声强度）
        #     x_input_condition: 额外的条件输入（默认0表示无）
        #     x_self_cond: 自条件输入（来自前一步的预测结果）
        # 返回:
        #     model_mean: 后验分布的均值（公式推导见下文）
        #     posterior_variance: 后验分布的方差
        #     posterior_log_variance: 方差的对数（数值稳定性处理）
        #     x_start: 模型预测的去噪结果（x_0）

        # ==================== 1. 获取模型预测结果 ====================
        # 调用model_predictions获取三个关键预测：
        #   pred_res: 预测的残差（x_input - x_start）
        #   pred_noise: 预测的噪声（未直接使用）
        #   x_start: 预测的去噪结果x_0
        preds = self.model_predictions(
            x_input, x, t, x_input_condition, x_self_cond)

        # 解构预测结果
        pred_res = preds.pred_res      # 残差预测：x_input与x_start的差异
        x_start = preds.pred_x_start   # 去噪结果预测

        # ==================== 2. 计算后验分布参数 ====================
        # 调用q_posterior计算三个关键参数：
        #   根据扩散过程理论推导的后验分布参数（公式见下方注释）    
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
        pred_res=pred_res,  # 模型预测的残差
        x_start=x_start,    # 模型预测的x_0
        x_t=x,              # 当前噪声图像
        t=t                 # 当前时间步
        )

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()# 禁用梯度计算（采样过程不需要反向传播）
    def p_sample(self, x_input, x, t: int, x_input_condition=0, x_self_cond=None):
        # 单步采样过程：从x_t生成x_{t-1}
        
        # 参数:
        #     x_input: 条件输入图像（如低分辨率/带掩码的图像）
        #     x: 当前噪声图像x_t（形状[b,c,h,w]）
        #     t: 当前时间步（整数）
        #     x_input_condition: 额外的条件输入（默认0）
        #     x_self_cond: 自条件输入（来自前一步的预测）
        
        # 返回:
        #     pred_img: 采样得到的x_{t-1} 
        #     x_start: 模型预测的干净图像x_0
        

        # --- 1. 准备时间步参数 ---
        # 获取batch大小和设备信息（b: batch_size, device: cuda/cpu）

        b, *_, device = *x.shape, x.device

        # 创建与batch大小匹配的时间步张量（全部填充为t）
        # 例如：如果x.shape[0]=8且t=10 → tensor([10,10,...,10])
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        
        # --- 2. 计算后验分布参数 ---
        # 获取：
        #   model_mean: 后验均值μ_θ(x_t,t)
        #   _: 方差（未使用）
        #   model_log_variance: 方差的对数（数值稳定性）
        #   x_start: 模型预测的x_0
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x_input, x=x, t=batched_times, x_input_condition=x_input_condition, x_self_cond=x_self_cond)
        
        # --- 3. 生成随机噪声 ---
        # 当t>0时添加噪声，t=0时不加噪声（最后一步）
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0

        # --- 4. 计算采样结果 ---
        # 根据反向扩散公式：x_{t-1} = μ_θ + σ_θ * z （z~N(0,I)）
        # 其中 exp(0.5 * log_variance) = sqrt(variance)
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()# 禁用梯度计算（采样过程不需要反向传播）
    def p_sample_loop(self, x_input, shape, last=True):
        # 完整的DDPM反向扩散采样循环（从噪声到图像）
        
        # 参数:
        #     x_input: 条件输入（可能是元组，包含主条件和附加条件）
        #     shape: 目标图像形状（batch_size, channels, height, width）
        #     last: 是否只返回最终结果（True）或所有中间结果（False）
        
        # 返回:
        #     归一化到[0,1]范围的图像（单张或列表）
        # 处理条件输入 
        if self.input_condition:
            # 如果有附加条件（如分割图/边缘图等）
            x_input_condition = x_input[1]  # 第二个元素作为附加条件
        else:
            x_input_condition = 0  # 无附加条件
        x_input = x_input[0]  # 主条件输入（如低分辨率图像）

        # 获取batch大小和设备信息
        batch, device = shape[0], self.betas.device
        # 初始化噪声图像 
        if self.condition:
            # 条件生成：从条件输入加噪声开始
            img = x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img  # 保存加噪后的初始图像（用于后续可视化）
        else:
            # 无条件生成：从纯噪声开始
            img = torch.randn(shape, device=device)

        x_start = None # 初始化自条件变量

        if not last:
            img_list = []    #准备结果存储 

        # --- 反向扩散循环 ---
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(
                x_input, img, t, x_input_condition, self_cond)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list,self.cfg)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list,self.cfg)


    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True):
        # DDIM (Denoising Diffusion Implicit Models) 采样过程
        # 相比DDPM可以跳过部分时间步，加速采样过程
        
        # 参数:
        #     x_input: 条件输入（可能是元组，包含主条件和附加条件）
        #     shape: 目标图像形状（batch_size, channels, height, width）
        #     last: 是否只返回最终结果（True）或所有中间结果（False）
        
        # 返回:
        #     归一化到[0,1]范围的图像（单张或列表）
    
        # # --- 1. 处理条件输入 ---
        #DDIM采样过程

        # --- 1. 处理条件输入 ---
        if self.input_condition:
            x_input_condition = x_input[1]  # 第二个元素作为附加条件
        else:
            x_input_condition = 0  # 无附加条件
        x_input = x_input[0]  # 主条件输入

        # 获取设备信息和采样参数
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # --- 2. 构建时间步序列 ---
        # 生成[-1, 0, 1,..., T-1]的时间序列
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        # --- 3. 初始化噪声图像 ---
        if self.condition:
            # 条件生成：从条件输入加噪声开始
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            # 无条件生成：从纯噪声开始
            img = torch.randn(shape, device=device)

        x_start = None  # 初始化自条件变量
        type = "use_pred_noise" # 采样算法类型（支持4种变体）

        if not last:  # 强制η=0（确定性采样），原论文中η∈[0,1]
            img_list = []

        eta = 0
        # --- 4. DDIM采样循环 ---
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            # 准备当前时间步的条件张量（形状[batch_size]）
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            # 自条件处理（如果启用）
            self_cond = x_start if self.self_condition else None
            # 获取模型预测（残差/噪声/去噪图像）
            preds = self.model_predictions(
                x_input, img, time_cond, x_input_condition, self_cond)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            # 终止条件（到达最后时间步）
            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue
            # --- 5. 计算DDIM系数 ---
            # α相关参数（控制数据部分）
            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            # betas2 = 1-(1-betas2_cumsum)/(1-betas2_cumsum_next)
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)
            # --- 6. 执行DDIM更新 ---
            if type == "use_pred_noise":
                # 标准DDIM更新（使用预测噪声）
                img = img - alpha*pred_res - \
                    (betas_cumsum-(betas2_cumsum_next-sigma2).sqrt()) * \
                    pred_noise + sigma2.sqrt()*noise
            elif type == "use_x_start":
                 # 使用x_start的更新方式
                img = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum*img + \
                    (1-sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum)*pred_res + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                # η=0的特殊情况（确定性采样）
                img = img - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                # η=1的特殊情况（类似DDPM）
                img = img - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise
            # 记录中间结果
            if not last:
                img_list.append(img)
        # --- 7. 处理输出 ---
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list,self.cfg)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            # 将图像从[-1,1]归一化到[0,1]范围
            return unnormalize_to_zero_to_one(img_list,self.cfg)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=16, last=True):
        # 扩散模型采样入口函数
        # 根据配置自动选择DDPM或DDIM采样方式，并处理输入标准化
        
        # 参数:
        #     x_input: 条件输入，可以是以下形式：
        #              - 无条件生成：0（默认）
        #              - 单条件生成：Tensor [C,H,W] 或 [B,C,H,W]
        #              - 多条件生成：Tuple (主条件, 附加条件)
        #     batch_size: 生成图像的数量（当无条件生成时使用）
        #     last: 是否只返回最终结果（True）或所有中间结果（False）
        
        # 返回:
        #     归一化到[0,1]范围的图像（Tensor或列表）
        # --- 1. 确定采样方法 ---
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # --- 2. 处理条件输入 ---
        if self.condition:
            # 处理多条件输入的情况
            if self.input_condition and self.input_condition_mask:
                # 当需要条件掩码时，单独标准化主条件（x_input[0]）
                x_input[0] = normalize_to_neg_one_to_one(x_input[0],self.cfg)
            else:
                # 单条件情况：整体标准化输入
                x_input = normalize_to_neg_one_to_one(x_input,self.cfg)
            # 从条件输入获取实际batch大小和图像尺寸
            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            # 无条件生成模式
            size = (batch_size, channels, image_size, image_size)
        # --- 3. 执行采样 ---
        # 调用选择的采样函数（DDPM或DDIM）
        return sample_fn(x_input, size, last=last)

    def q_sample(self, x_start, x_res, t, noise=None):
        # 前向扩散过程 - 将输入图像逐步加噪
        
        # 参数:
        #     x_start: 原始图像（ground truth）[B,C,H,W]
        #     x_res: 残差（条件图像与原始图像的差值）[B,C,H,W]
        #     t: 时间步（可以是标量或形状为[B]的张量）
        #     noise: 可选的外部噪声（默认随机生成）
        
        # 返回:
        #     加噪后的图像 x_t [B,C,H,W]
        
        # 数学形式:
        #     x_t = x_start + α̃_t * x_res + β̃_t * ε
        #     其中：
        #     - α̃_t 是累积残差系数（self.alphas_cumsum）
        #     - β̃_t 是累积噪声系数（self.betas_cumsum）
        #     - ε ~ N(0,I) 是随机噪声
        #前向扩散过程
        # 如果没有提供噪声，则生成标准正态分布噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 执行前向扩散公式（三项相加）：
        # 1. 原始图像 x_start
        # 2. 按时间步加权的残差 α̃_t * x_res
        # 3. 按时间步加权的噪声 β̃_t * noise
        return (
            x_start+extract(self.alphas_cumsum, t, x_start.shape) * x_res +
            extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    @property  # 将方法转为属性，可以通过 .loss_fn 直接访问
    def loss_fn(self):
        # 损失函数选择器（动态返回对应的PyTorch损失函数）
        
        # 根据配置的 loss_type 返回：
        # - 'l1' : 返回 torch.nn.functional.l1_loss
        # - 'l2' : 返回 torch.nn.functional.mse_loss
        
        # 属性:
        #     self.loss_type: 字符串，必须在 ['l1', 'l2'] 中
        
        # 返回:
        #     对应的PyTorch损失函数
        
        # 示例:
        #     >>> model.loss_type = 'l1'
        #     >>> loss = model.loss_fn(pred, target)  # 调用L1损失
        
        # 设计说明:
        #     1. 使用 @property 装饰器使方法可以像属性一样访问
        #     2. 延迟返回函数引用而非立即计算，提高灵活性
        #     3. 集中管理损失函数选择逻辑

        # L1损失（绝对误差），对异常值更鲁棒
        if self.loss_type == 'l1':
            return F.l1_loss
        
        # L2损失（均方误差），对异常值更敏感但梯度更平滑
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, imgs, t, noise=None):
        # 扩散模型的核心损失计算过程
        
        # 参数:
        #     imgs: 输入数据，可能是:
        #         - 无条件生成: [B,C,H,W] 张量
        #         - 条件生成: 列表 [gt_img, cond_img, extra_cond] 
        #     t: 随机采样的时间步 [B,] 
        #     noise: 可选预生成的噪声（默认随机生成）
        
        # 返回:
        #     损失值列表（根据不同的预测目标可能有多个损失项）


        # ==================== 1. 准备输入数据 ====================
        if isinstance(imgs, list):  # Condition
            if self.input_condition:
                x_input_condition = imgs[2] # 额外条件输入（如分割图）
            else:
                x_input_condition = 0
            x_input = imgs[1] # 主条件图像（如低分辨率图）
            x_start = imgs[0] # 真实图像（ground truth）
        else:  # 无条件生成模式
            x_input = 0
            x_start = imgs # 真实图像


        # 生成随机噪声（如果未提供）
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 计算残差（条件图像与真实图像的差异）
        x_res = x_input - x_start

        b, c, h, w = x_start.shape

        # ==================== 2. 前向扩散过程 ====================
        # 通过q_sample计算加噪后的图像x_t
        # 公式: x_t = x_start + α̃_t*x_res + β̃_t*noise
        x = self.q_sample(x_start, x_res, t, noise=noise) #X是加噪后的图像

        # ==================== 3. 自条件生成处理 ====================
        x_self_cond = None

        # 检查是否启用自条件生成，并且以50%的概率执行（随机正则化）
        if self.self_condition and random.random() < 0.5:
            # 使用torch.no_grad()上下文管理器，禁止梯度计算（因为这是辅助生成，不需要反向传播）
            with torch.no_grad():
                # 通过模型预测当前时间步t的起始图像x_start（即去噪后的图像）
                # 参数说明：
                #   x_input: 条件输入图像（如低分辨率图像/带掩码图像）
                #   x: 当前噪声图像x_t
                #   t: 当前时间步
                #   x_input_condition if self.input_condition else 0: 可选的条件输入
                # 返回的pred_x_start是模型预测的去噪结果
                x_self_cond = self.model_predictions(
                    x_input, x, t, x_input_condition if self.input_condition else 0).pred_x_start
                
                # 从计算图中分离预测结果，确保它不会影响梯度计算
                # 这相当于创建一个"冻结"的版本，仅作为条件输入使用
                x_self_cond.detach_()
        # ==================== 4. 构建模型输入 ====================
        if not self.condition:
            x_in = x
        else:
            if self.img_to_img_translation:
                if self.input_condition:
                    x_in = torch.cat((x, x_input_condition), dim=1)
                else:
                    x_in = x
            else:
                if self.input_condition:
                    x_in = torch.cat((x, x_input, x_input_condition), dim=1)
                else:
                    x_in = torch.cat((x, x_input), dim=1)

        # ==================== 5. 模型前向传播 ====================
        # 输入说明:
        #   x_in: 组合后的输入
        #   [alphas_cumsum, betas_cumsum]: 缩放后的时间步嵌入
        #   x_self_cond: 自条件输入
        model_out = self.model(x_in,
                               [self.alphas_cumsum[t]*self.num_timesteps,
                                   self.betas_cumsum[t]*self.num_timesteps],
                               x_self_cond)
        

        # ==================== 6. 准备预测目标和真实目标 ====================
        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)

            pred_res = model_out[0]

        elif self.objective == "pred_x0":
            target.append(x_start)

            pred_x0 = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        # =================== 7. 计算损失 ====================
        u_loss = False # u_loss开关（通常关闭）
        if u_loss: # 计算潜在空间损失（实验性功能）
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else: # 常规损失计算
            loss_list = []
            for i in range(len(model_out)):# 逐项计算损失
                loss = self.loss_fn(model_out[i], target[i], reduction='none') # 先对空间维度取平均，再对batch取平均
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
            return loss_list

    def forward(self, img, *args, **kwargs):

        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
            
        # 无条件生成模式  # 直接从输入张量获取形状信息 # 模型预设的图像尺寸
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size

        # --- 2. 采样随机时间步 ---
        # 为每个样本生成随机时间步（范围0到num_timesteps-1）
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # --- 3. 输入标准化 ---
        if self.input_condition and self.input_condition_mask: # 特殊条件模式：单独标准化每个输入
            img[0] = normalize_to_neg_one_to_one(img[0],self.cfg)
            img[1] = normalize_to_neg_one_to_one(img[1],self.cfg)
        else: # 常规模式：整体标准化输入
            img = normalize_to_neg_one_to_one(img,self.cfg)
        # --- 4. 计算损失 ---
        return self.p_losses(img, t, *args, **kwargs)










class Trainer(object):
    def __init__(self,diffusion_model,cfg,logger,output_folder):
        super().__init__()
        #获取配置
        train_batch_size=cfg['diffusion_training']['batch_size']
        train_lr=cfg['diffusion_optim']['lr']
        train_num_steps=cfg['diffusion_training']['epochs']
        gradient_accumulate_every=cfg['diffusion_training']['gradient_accumulate_every']
        ema_decay=cfg['diffusion_model']['ema_rate']
        amp=cfg['diffusion_training']['isamp']
        condition=cfg['diffusion_training']['condition']
        save_and_sample_every=cfg['diffusion_training']['eval_freq_sampling']
        crop_patch=cfg['diffusion_training']['crop_patch']
        data_path = cfg['diffusion_data']['root']
        ema_update_every=cfg['diffusion_training']['ema_update_every']
        fp16 = cfg['diffusion_training']['fp16']
        split_batches = cfg['diffusion_training']['split_batches']
        sub_dir = cfg['diffusion_training']['sub_dir']
        num_unet = cfg['diffusion_model']['num_unet']
        data_tpye = cfg['diffusion_data']['dataset']
        samples_batch_size = cfg['diffusion_eval']['batch_size']



        #配置训练器
        self.config = cfg
        self.logger = logger
        self.sample_dir = os.path.join(output_folder , "samples") 
        self.checkpoint_dir = os.path.join(output_folder, "checkpoints")  # 检查点保存目录
        self.test_output_folder = output_folder.replace("_Train", "_Test")
        self.device = cfg['device']

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',        
            )

        self.sub_dir = sub_dir
        self.crop_patch = crop_patch

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            samples_batch_size), 'number of samples must have an integer square root'
        self.samples_batch_size = samples_batch_size
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        self.num_unet = num_unet

        if cfg['phase'] != 'Iteration':
            self.train_dataset = dataset_pickle(self.config , phese="train", condition=self.condition)
            self.test_dataset  = dataset_pickle(self.config , phese="val", condition=self.condition)
            self.train_loader = self.accelerator.prepare(
                DataLoader(self.train_dataset,
                        batch_size=train_batch_size,
                        shuffle=True,        # 训练集需要shuffle
                        pin_memory=True,
                        num_workers=4)
            )
            self.sample_loader = self.accelerator.prepare(
                DataLoader(self.test_dataset,
                        batch_size=samples_batch_size,
                        shuffle=False,       # 验证集通常不shuffle
                        pin_memory=True,
                        num_workers=4)
            )

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


        self.step = 0

        if self.num_unet == 1:
            self.model, self.opt = self.accelerator.prepare(
                self.model, self.opt0)
        elif self.num_unet == 2:
            self.model, self.opt0, self.opt1 = self.accelerator.prepare(
                self.model, self.opt0, self.opt1)


    def save(self, epoch):
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
        torch.save(data, f"{self.checkpoint_dir}/model-{str(epoch)}.pt")

    def load(self):
        path = Path(os.path.join(self.config['diffusion_eval']['pre_model']))
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
        else :
            raise FileNotFoundError(f"模型文件不存在: {str(path)}")


    def train(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
        Path(self.sample_dir).mkdir(parents=True, exist_ok=True)  
        accelerator = self.accelerator
        # 创建数据标准化器和其逆操作
        for epoch in range(1, self.config['diffusion_training']['epochs']):
            self.logger.info('=================================================')
            self.logger.info(f"[EVAL] epoch: {epoch}/{self.config['diffusion_training']['epochs']}")
            self.logger.info('=================================================')

            ################################################################
            ##########################  训练  ##############################
            ################################################################
            for step, batch in enumerate(self.train_loader, start=1):
                
                for i in range(len(batch)):
                    batch[i] = batch[i].to(self.config['device'])  # 对数据进行标准化并移动到指定设备

                if self.num_unet == 1:
                    total_loss = [0]
                elif self.num_unet == 2:
                    total_loss = [0, 0]

                with self.accelerator.autocast():
                    loss = self.model(batch)
                    for i in range(self.num_unet):
                        loss[i] = loss[i] 
                        total_loss[i] = total_loss[i] + loss[i].item()

                for i in range(self.num_unet):
                    self.accelerator.backward(loss[i])

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                accelerator.wait_for_everyone()

                if self.num_unet == 1:
                    self.opt0.step()
                    self.opt0.zero_grad()
                elif self.num_unet == 2:
                    self.opt0.step()
                    self.opt0.zero_grad()
                    self.opt1.step()
                    self.opt1.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()
                    if step % self.config['diffusion_training']['log_freq'] == 0:
                        if self.num_unet == 1:
                            self.logger.info(f'step:{step}---loss_unet0: {total_loss[0]:.4f}')
                        elif self.num_unet == 2:
                            self.logger.info(
                                f'step:{step}---loss_unet0: {total_loss[0]:.4f}---loss_unet1: {total_loss[1]:.4f}')
                            
            ################################################################
            #############################保存样本###########################
            ################################################################
            if epoch % self.config['diffusion_training']['eval_freq_sampling'] == 0 and self.config['diffusion_training']['eval_freq_sampling'] !=0:
                self.sample(epoch)

            ################################################################
            ########################保存一个检查点###########################
            ################################################################
            if epoch % self.config['diffusion_training']['eval_freq'] == 0:
                self.save(epoch)


        self.logger.info('training complete')

    def sample(self,epoch, last=True):
        self.ema.ema_model.eval()
        with torch.no_grad():
            batches = self.samples_batch_size
            for step_eval, eval_batch in enumerate(self.sample_loader, start=1):
                if step_eval==1:

                    input_image = eval_batch[1] #b,c,h,w
                    gt_image = eval_batch[0]
                    pred_image = self.ema.ema_model.sample(eval_batch[1:], batch_size=batches, last=last)

                    this_sample_dir = os.path.join(self.sample_dir, "epoch_{}".format(epoch))  # 样本保存目录
                    Path(this_sample_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
                    real_path = os.path.join(this_sample_dir,
                                            f"sample{step_eval:03d}_target.png")
                    pred_path = os.path.join(this_sample_dir,
                                            f"sample{step_eval:03d}_predict.png")
                    input_path = os.path.join(this_sample_dir,
                                            f"sample{step_eval:03d}_input.png")
                    # 保存为独立文件
                    plt.imsave(real_path, gt_image[0][0].cpu(), cmap=plt.cm.Greys_r)
                    plt.imsave(pred_path, pred_image[1][0][0].cpu(), cmap=plt.cm.Greys_r)
                    plt.imsave(input_path, input_image[0][0].cpu(), cmap=plt.cm.Greys_r)


                    # 计算指标（逐样本计算）
                    batch_mse = get_mse(pred_image[1][0], gt_image[0])
                    batch_psnr = get_psnr(pred_image[1][0], gt_image[0])
                    batch_ssim = get_ssim(pred_image[1][0], gt_image[0])

                    # 计算input和GT的指标
                    batch_mse2 = get_mse(input_image[0], gt_image[0])
                    batch_psnr2 = get_psnr(input_image[0], gt_image[0])
                    batch_ssim2 = get_ssim(input_image[0], gt_image[0])
                    self.logger.info(f"epoch_: {epoch} ")
                    self.logger.info(f" ----pre and GT----  batch_mse: {batch_mse} batch_psnr: {batch_psnr} batch_ssim: {batch_ssim}")
                    self.logger.info(f"----input and GT---- batch_mse: {batch_mse2} batch_psnr: {batch_psnr2} batch_ssim: {batch_ssim2}")

            self.ema.ema_model.train()
            return epoch

    def test(self, last=True,iter_data=None):
        if self.config['phase'] == 'Iteration' :
            sample_loader = Creat_Iter_dataset(iter_data)
            self.sample_loader = self.accelerator.prepare(
                DataLoader(sample_loader,
                        batch_size=self.samples_batch_size,
                        shuffle=False,        # 训练集需要shuffle
                        pin_memory=True,
                        num_workers=4)
            )
            
        
        list_return = []
        #初始化
        Path(self.test_output_folder).mkdir(parents=True, exist_ok=True)  # 创建目录
        self.ema.to(self.device)
        self.ema.ema_model.eval()
        
        total_mse = 0.
        total_psnr = 0.
        total_ssim = 0.

        total_mse2 = 0.
        total_psnr2 = 0.
        total_ssim2 = 0.
        num_proj = 0
        with torch.no_grad():
            batches = self.samples_batch_size
            for step_eval, eval_batch in enumerate(self.sample_loader):
                    input_image = eval_batch[1] #b,c,h,w
                    gt_image = eval_batch[0]
                    pred_image = self.ema.ema_model.sample(eval_batch[1:], batch_size=batches, last=last)
                    # maxi = pred_image[1].max()
                    # mini = pred_image[1].min()

                    # maxs = input_image.max()
                    # mins = input_image.min()

                    # maxg = gt_image.max()
                    # ming = gt_image.min()
                    this_sample_dir = os.path.join(self.test_output_folder,'sample')  # 样本保存目录
                    Path(this_sample_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
                    real_path = os.path.join(this_sample_dir,
                                            f"sample{step_eval:03d}_target.png")

                    pred_path = os.path.join(this_sample_dir,
                                            f"sample{step_eval:03d}_predict.png")
                    input_path = os.path.join(this_sample_dir,
                                            f"sample{step_eval:03d}_input.png")
                    # 保存为独立文件
                    plt.imsave(real_path, gt_image[0][0].cpu(), cmap=plt.cm.Greys_r)
                    plt.imsave(pred_path, pred_image[1][0][0].cpu(), cmap=plt.cm.Greys_r)
                    plt.imsave(input_path, input_image[0][0].cpu(), cmap=plt.cm.Greys_r)

                    # 计算指标（逐样本计算）
                    batch_mse = get_mse(pred_image[1][0], gt_image[0])
                    batch_psnr = get_psnr(pred_image[1][0], gt_image[0])
                    batch_ssim = get_ssim(pred_image[1][0], gt_image[0])

                    # 计算input和GT的指标
                    batch_mse2 = get_mse(input_image[0], gt_image[0])
                    batch_psnr2 = get_psnr(input_image[0], gt_image[0])
                    batch_ssim2 = get_ssim(input_image[0], gt_image[0])


                    # 累加结果
                    total_mse += batch_mse.item() * batches
                    total_psnr += batch_psnr.item() * batches
                    total_ssim += batch_ssim.item() * batches

                    # 累加结果2
                    total_mse2 += batch_mse2.item() * batches
                    total_psnr2 += batch_psnr2.item() * batches
                    total_ssim2 += batch_ssim2.item() * batches

                    num_proj += batches
                    
                    self.logger.info(f"----proj_: {step_eval}----")
                    self.logger.info(f" ----pre and GT----   batch_mse: {batch_mse} batch_psnr: {batch_psnr} batch_ssim: {batch_ssim}")
                    self.logger.info(f"----input and GT----  batch_mse: {batch_mse2} batch_psnr: {batch_psnr2} batch_ssim: {batch_ssim2}")
                    list_return.append(pred_image[1][0].cpu().numpy())

            # 计算平均指标
            avg_mse = total_mse / num_proj
            avg_psnr = total_psnr / num_proj
            avg_ssim = total_ssim / num_proj

            # 计算平均指标2
            avg_mse2 = total_mse2 / num_proj
            avg_psnr2 = total_psnr2 / num_proj
            avg_ssim2 = total_ssim2 / num_proj
            self.logger.info(f"---------------------------------------------------------------------------------------------")
            self.logger.info(f"-----num_proj:{num_proj}-----")
            self.logger.info(f" ----pre and GT----    avg_mse: {avg_mse} avg_psnr: {avg_psnr} avg_ssim: {avg_ssim}")
            self.logger.info(f"----input and GT----   avg_mse: {avg_mse2} avg_psnr: {avg_psnr2} avg_ssim: {avg_ssim2}")
        return np.concatenate(list_return, axis=0)

class Creat_Iter_dataset(Dataset):
    def __init__(self, data_tensor):

        self.input = np.expand_dims(data_tensor[0], axis=1)  # 在 axis=1 处插入新维度
        self.gt = np.expand_dims(data_tensor[1], axis=1)     # 同上

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.gt[idx] ,self.input[idx]




'''
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
        samples_batch_size=1,           # 每次生成的样本数量
        output_folder='./results/sample',  # 结果保存路径
        amp=False,                # 是否启用自动混合精度
        fp16=False,               # 是否使用FP16半精度
        split_batches=False,       # 是否在多GPU训练时拆分批次
        convert_image_to=None,    # 图像格式转换（如RGB/L等）
        condition=False,          # 是否使用条件生成模式
        sub_dir=False,            # 是否在子目录中查找数据
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
        self.sub_dir = sub_dir
        self.crop_patch = crop_patch

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            samples_batch_size), 'number of samples must have an integer square root'
        self.samples_batch_size = samples_batch_size
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        self.num_unet = num_unet

        if self.condition:

            if len(folder) == 3:
                self.condition_type = 1
                # test_input
                ds = dataset(folder[-1], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, condition=0, equalizeHist=equalizeHist, crop_patch=crop_patch, sample=True, generation=generation)
                trian_folder = folder[0:2]

                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=samples_batch_size, shuffle=True,
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
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=samples_batch_size, shuffle=True,
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
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=samples_batch_size, shuffle=True,
                                                                               pin_memory=True, num_workers=4)))  # cpu_count()

                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=2, equalizeHist=equalizeHist, crop_patch=crop_patch, generation=generation)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)))
            elif len(folder) == 2:
                self.condition_type = 3
                ds = dataset(folder[0], self.image_size)
                self.sample_dataset = ds #
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=samples_batch_size, shuffle=True,pin_memory=True, num_workers=4)))  
                ds = dataset(folder[1], self.image_size)
                self.dl = cycle(self.accelerator.prepare(DataLoader(ds, batch_size=train_batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)))

            else:
                self.condition_type = 3
                ds = dataset(folder[0], self.image_size)
                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(DataLoader(self.sample_dataset, batch_size=samples_batch_size, shuffle=True,pin_memory=True, num_workers=4)))  
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

            self.set_output_folder(output_folder)

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
        torch.save(data, str(self.output_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.output_folder / f'model-{milestone}.pt')

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
        accelerator = self.accelerator

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                if self.num_unet == 1:
                    total_loss = [0]
                elif self.num_unet == 2:
                    total_loss = [0, 0]
                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        for i in range(self.num_unet):
                            loss[i] = loss[i] / self.gradient_accumulate_every
                            total_loss[i] = total_loss[i] + loss[i].item()

                    for i in range(self.num_unet):
                        self.accelerator.backward(loss[i])

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                accelerator.wait_for_everyone()

                if self.num_unet == 1:
                    self.opt0.step()
                    self.opt0.zero_grad()
                elif self.num_unet == 2:
                    self.opt0.step()
                    self.opt0.zero_grad()
                    self.opt1.step()
                    self.opt1.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.sample(milestone)

                        if self.step != 0 and self.step % (self.save_and_sample_every*10) == 0:
                            self.save(milestone)
                if self.num_unet == 1:
                    pbar.set_description(f'loss_unet0: {total_loss[0]:.4f}')
                elif self.num_unet == 2:
                    pbar.set_description(
                        f'loss_unet0: {total_loss[0]:.4f},loss_unet1: {total_loss[1]:.4f}')
                pbar.update(1)

        accelerator.print('training complete')

    def sample(self, milestone, last=True, FID=False):

        self.ema.ema_model.eval()

        with torch.no_grad():
            batches = self.samples_batch_size
        # --- 条件输入处理 ---
        # 根据条件类型准备输入数据（x_input_sample）和可视化数据（show_x_input_sample）
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
                nrow = int(math.sqrt(self.samples_batch_size))
            else:
                nrow = all_images.shape[0]

            if FID:
                for i in range(batches):
                    file_name = f'sample-{milestone}.png'
                    utils.save_image(
                        all_images_list[0][i].unsqueeze(0), os.path.join(self.output_folder, file_name), nrow=1)
                    milestone += 1
                    if milestone >= self.total_n_samples:
                        break
            else:
                file_name = f'sample-{milestone}.png'
                utils.save_image(all_images, str(
                    self.output_folder / file_name), nrow=nrow)
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
                batch_size=self.samples_batch_size)
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
                    batches = self.samples_batch_size

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
                    nrow = int(math.sqrt(self.samples_batch_size))
                else:
                    nrow = all_images.shape[0]

                utils.save_image(all_images, str(
                    self.output_folder / file_name), nrow=nrow)
                print("test-save "+file_name)
        else:
            if FID:
                self.total_n_samples = 50000
                img_id = len(glob.glob(f"{self.output_folder}/*"))
                n_rounds = (self.total_n_samples -
                            img_id) // self.samples_batch_size+1
            else:
                n_rounds = 100
            for i in range(n_rounds):
                if FID:
                    i = img_id
                img_id = self.sample(i, last=last, FID=FID)
        print("test end")

    def set_output_folder(self, path):
        self.output_folder = Path(path)
        if not self.output_folder.exists():
            os.makedirs(self.output_folder)

'''