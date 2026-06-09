import functools
import time
import torch
from numpy.testing._private.utils import measure
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model.diffusion.diffusion_model import utils as mutils
from utilis.diffusion_utils.sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
from utilis.diffusion_utils.utils import fft2, ifft2, fft2_m, ifft2_m
from .ct import *  # 导入CT相关物理模型
from utilis.diffusion_utils.utils import show_samples, show_samples_gray, clear, clear_color, batchfy

# λ调度器基类
class lambda_schedule:
    def __init__(self, total=2000):
        self.total = total  # 总迭代次数

    def get_current_lambda(self, i):
        pass  # 抽象方法，由子类实现

# 线性λ调度器
class lambda_schedule_linear(lambda_schedule):
    def __init__(self, start_lamb=1.0, end_lamb=0.0):
        super().__init__()
        self.start_lamb = start_lamb  # 起始λ值
        self.end_lamb = end_lamb  # 结束λ值

    def get_current_lambda(self, i):
        # 线性插值计算当前λ
        return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)

# 恒定λ调度器
class lambda_schedule_const(lambda_schedule):
    def __init__(self, lamb=1.0):
        super().__init__()
        self.lamb = lamb  # 固定λ值

    def get_current_lambda(self, i):
        return self.lamb  # 始终返回固定值

# 沿z轴(批量维度)的差分算子
def _Dz(x): # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]  # 前移一位
    y[-1] = x[0]    # 循环边界
    return y - x     # 返回差分

# 沿z轴的差分转置算子
def _DzT(x): # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]

    tempt = -(y-x)
    difft = tempt[:-1]
    y[1:] = difft
    y[0] = x[-1] - x[0]  # 处理边界条件
    return y

# 沿x轴的差分算子 (空间维度)
def _Dx(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]  # 沿x轴前移
    y[:, :, -1, :] = x[:, :, 0, :]    # 循环边界
    return y - x

# 沿x轴的差分转置算子
def _DxT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    tempt = -(y - x)
    difft = tempt[:, :, :-1, :]
    y[:, :, 1:, :] = difft
    y[:, :, 0, :] = x[:, :, -1, :] - x[:, :, 0, :]  # 边界处理
    return y

# 沿y轴的差分算子 (空间维度)
def _Dy(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]  # 沿y轴前移
    y[:, :, :, -1] = x[:, :, :, 0]    # 循环边界
    return y - x

# 沿y轴的差分转置算子
def _DyT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    tempt = -(y - x)
    difft = tempt[:, :, :, :-1]
    y[:, :, :, 1:] = difft
    y[:, :, :, 0] = x[:, :, :, -1] - x[:, :, :, 0]  # 边界处理
    return y

# 获取带ADMM和TV正则化的Radon采样器 (单切片版本)
def get_pc_radon_ADMM_TV(sde, predictor, corrector, inverse_scaler, snr,
                         n_steps=1, probability_flow=False, continuous=False,
                         denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                         final_consistency=False, img_cache=None, img_shape=None, lamb_1=5, rho=10):
    """ 稀疏应用测量一致性的ADMM-TV采样器 """
    # 定义预测器和校正器更新函数
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    # 初始化ADMM变量
    if img_cache != None:
        img_shape[0] += 1
    del_z = torch.zeros(img_shape)  # 分裂变量
    udel_z = torch.zeros(img_shape)  # 对偶变量
    eps = 1e-10

    # Radon变换和反变换
    def _A(x):
        return radon.A(x)  # 前向投影
    
    def _AT(sinogram):
        return radon.AT(sinogram)  # 反投影

    # Kaczmarz方法实现数据一致性
    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None, norm_const=None):
        x = x + lamb * _AT(measurement - _A(x))/norm_const  # 更新公式
        x_mean = x
        return x, x_mean
    
    # 共轭梯度法中的A算子
    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))  # 包含TV正则项

    # 共轭梯度法求解
    def CG(A_fn,b_cg,x,n_inner=10):
        r = b_cg - A_fn(x)  # 残差
        p = r  # 搜索方向
        rs_old = torch.matmul(r.view(1,-1),r.view(1,-1).T)  # 残差平方

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old/torch.matmul(p.view(1,-1),Ap.view(1,-1).T)  # 步长
    
            x += a * p  # 更新解
            r -= a * Ap  # 更新残差

            rs_new = torch.matmul(r.view(1,-1),r.view(1,-1).T)
            if torch.sqrt(rs_new) < eps:  # 收敛检查
                break
            p = r + (rs_new/rs_old) * p  # 更新搜索方向
            rs_old = rs_new
        return x

    # ADMM-TV主循环
    def CS_routine(x, ATy, niter=20):
        if img_cache != None:
            x = torch.cat([img_cache,x],dim=0)
            idx = list(range(len(x),0,-1))
            x = x[idx]

        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
            
        for i in range(niter):
            # 构建CG的右端项
            b_cg = ATy + rho * (_DzT(del_z)-_DzT(udel_z))
            # 共轭梯度求解
            x = CG(A_cg, b_cg, x, n_inner=1)

            # 更新分裂变量和对偶变量
            del_z = shrink(_Dz(x) + udel_z, lamb_1/rho)
            udel_z = _Dz(x) - del_z + udel_z
            
        if img_cache != None:
            x = x[idx]
            x = x[1:]
            del_z[-1] = 0
            udel_z[-1] = 0
            
        x_mean = x
        return x, x_mean

    # 获取更新函数
    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean
        return radon_update_fn

    # 获取校正器更新函数
    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean
        return radon_update_fn

    # 创建预测器和校正器更新函数
    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    # 主采样函数
    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            # 初始化采样
            x = sde.prior_sampling(data.shape).to(data.device)

            # 计算归一化常数
            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            
            # 时间步设置
            timesteps = torch.linspace(sde.T, eps, sde.N)
            
            # 主采样循环
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 预测器步骤
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                # 校正器步骤
                x, x_mean = corrector_radon_update_fn(model, data, x, t, measurement=measurement)
                
                # 保存进度
                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(os.path.join(save_root, 'recon', 'progress', f'progress{i}.png'), clear(x_mean[0:1]), cmap='gray')
            
            # 最终一致性步骤
            if final_consistency:
                x, x_mean = kaczmarz(x, x_mean, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon

# 获取带ADMM和TV正则化的Radon采样器 (体积版本)
def get_pc_radon_ADMM_TV_vol(sde, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                             final_consistency=False, img_shape=None, lamb_1=5, rho=10):
    """ 体积数据的ADMM-TV采样器 """
    # 定义预测器和校正器更新函数
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    # 初始化ADMM变量
    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    # Radon变换和反变换
    def _A(x):
        return radon.A(x)
    
    def _AT(sinogram):
        return radon.AT(sinogram)

    # Kaczmarz方法
    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None, norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    # 共轭梯度法中的A算子
    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))

    # 共轭梯度法求解
    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    # ADMM-TV主循环
    def CS_routine(x, ATy, niter=20):
        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_z = _Dz(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    # 获取更新函数
    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean
        return radon_update_fn

    # 获取ADMM-TV更新函数
    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, Sinogram=None,image_constraints=None):
            with torch.no_grad():
                ATy = _AT(Sinogram)#是否可以直接使用y
                x, x_mean = CS_routine(x, ATy, niter=1)
                #x, x_mean = CS_routine(x, image_constraints, niter=1)
                return x, x_mean
        return ADMM_TV_fn

    # 创建预测器、校正器和ADMM-TV更新函数
    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    # 主采样函数
    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            # 初始化采样
            x = sde.prior_sampling(data.shape).to(data.device)

            # 计算归一化常数
            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            
            # 时间步设置
            timesteps = torch.linspace(sde.T, eps, sde.N)
            
            # 主采样循环
            for i in tqdm(range(sde.N)):
                if i >= 0:
                    t = timesteps[i]
                    # 1. 分批处理以适应GPU内存
                    x_batch = batchfy(x, 12)
                    # 2. 对每批执行PC步骤
                    x_agg = list()
                    for idx, x_batch_sing in enumerate(x_batch):
                        x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                        x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                        x_agg.append(x_batch_sing)
                    # 3. 聚合结果执行ADMM TV
                    x = torch.cat(x_agg, dim=0)
                    # 4. 执行ADMM TV
                    x, x_mean = mc_update_fn(x, Sinogram=measurement,image_constraints=data)

                    # 保存进度
                    if save_progress:
                        if (i % 50) == 0:
                            print(f'iter: {i}/{sde.N}')
                            plt.imsave(os.path.join(save_root, 'recon', 'progress', f'progress{i}.png'), clear(x_mean[0:1]), cmap='gray')
            # 最终一致性步骤
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon

# 各向异性TV的ADMM实现
def get_ADMM_TV(eps=1e-5, radon=None, save_progress=False, save_root=None,
                img_shape=None, lamb_1=5, rho=10, outer_iter=30, inner_iter=20):
    """ 各向异性TV的ADMM实现 """
    # 初始化ADMM变量
    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    # Radon变换和反变换
    def _A(x):
        return radon.A(x)
    
    def _AT(sinogram):
        return radon.AT(sinogram)

    # 共轭梯度法中的A算子
    def A_cg(x):
        return _AT(_A(x)) + rho * (_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x)))

    # 共轭梯度法求解
    def CG(A_fn, b_cg, x, n_inner=20):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    # ADMM-TV主循环
    def CS_routine(x, ATy, niter=30):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
            
        for i in tqdm(range(niter)):
            # 构建CG右端项
            b_cg = ATy + rho * ((_DxT(del_x) - _DxT(udel_x))
                                + (_DyT(del_y) - _DyT(udel_y))
                                + (_DzT(del_z) - _DzT(udel_z)))
            # 共轭梯度求解
            x = CG(A_cg, b_cg, x, n_inner=inner_iter)
            
            # 保存进度
            if save_progress:
                plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x[0:1]), cmap='gray')

            # 更新分裂变量和对偶变量
            del_x = shrink(_Dx(x) + udel_x, lamb_1 / rho)
            del_y = shrink(_Dy(x) + udel_y, lamb_1 / rho)
            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_x = _Dx(x) - del_x + udel_x
            udel_y = _Dy(x) - del_y + udel_y
            udel_z = _Dz(x) - del_z + udel_z
        return x

    # 获取ADMM-TV更新函数
    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=outer_iter)
                return x, x_mean
        return ADMM_TV_fn

    mc_update_fn = get_ADMM_TV_fn()

    # ADMM-TV主函数
    def ADMM_TV(data, measurement=None):
        with torch.no_grad():
            x = torch.zeros(data.shape).to(data.device)
            x = mc_update_fn(x, measurement=measurement)
            return x

    return ADMM_TV

# 各向同性TV的ADMM实现
def get_ADMM_TV_isotropic(eps=1e-5, radon=None, save_progress=False, save_root=None,
                          img_shape=None, lamb_1=5, rho=10, outer_iter=30, inner_iter=20):
    """ 各向同性TV的ADMM实现 """
    # 初始化ADMM变量
    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    # Radon变换和反变换
    def _A(x):
        return radon.A(x)
    
    def _AT(sinogram):
        return radon.AT(sinogram)

    # 共轭梯度法中的A算子
    def A_cg(x):
        return _AT(_A(x)) + rho * (_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x)))

    # 共轭梯度法求解
    def CG(A_fn, b_cg, x, n_inner=20):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    # ADMM-TV主循环 (各向同性版本)
    def CS_routine(x, ATy, niter=30):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
            
        for i in tqdm(range(niter)):
            # 构建CG右端项
            b_cg = ATy + rho * ((_DxT(del_x) - _DxT(udel_x))
                                + (_DyT(del_y) - _DyT(udel_y))
                                + (_DzT(del_z) - _DzT(udel_z)))
            # 共轭梯度求解
            x = CG(A_cg, b_cg, x, n_inner=inner_iter)
            
            # 保存进度
            if save_progress:
                plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x[0:1]), cmap='gray')

            # 各向同性TV处理
            _Dxx = _Dx(x)
            _Dyx = _Dy(x)
            _Dzx = _Dz(x)
            # 合并梯度维度
            _Dxa = torch.cat((_Dxx, _Dyx, _Dzx), dim=1)
            udel_a = torch.cat((udel_x, udel_y, udel_z), dim=1)

            # 各向同性prox操作
            del_a = prox_l21(_Dxa + udel_a, lamb_1 / rho, dim=1)

            # 分离回各维度
            del_x, del_y, del_z = torch.split(del_a, 1, dim=1)

            # 更新对偶变量
            udel_x = _Dxx - del_x + udel_x
            udel_y = _Dyx - del_y + udel_y
            udel_z = _Dzx - del_z + udel_z
        return x

    # 获取ADMM-TV更新函数
    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x = CS_routine(x, ATy, niter=outer_iter)
                return x
        return ADMM_TV_fn

    mc_update_fn = get_ADMM_TV_fn()

    # ADMM-TV主函数
    def ADMM_TV(data, measurement=None):
        with torch.no_grad():
            x = torch.zeros(data.shape).to(data.device)
            x = mc_update_fn(x, measurement=measurement)
            return x

    return ADMM_TV

# L21范数的proximal算子 (用于各向同性TV)
def prox_l21(src, lamb, dim):
    """
    src.shape = [448(z), 1, 256(x), 256(y)]
    """
    # 计算L2范数
    weight_src = torch.linalg.norm(src, dim=dim, keepdim=True)
    # 收缩操作
    weight_src_shrink = shrink(weight_src, lamb)

    # 计算权重
    weight = weight_src_shrink / weight_src
    return src * weight  # 应用权重

# 软阈值收缩函数
def shrink(weight_src, lamb):
    return torch.sign(weight_src) * torch.max(torch.abs(weight_src) - lamb, torch.zeros_like(weight_src))

# MRI应用的ADMM-TV采样器
def get_pc_radon_ADMM_TV_mri(sde, predictor, corrector, inverse_scaler, snr, mask=None,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, save_progress=True, save_root=None,
                             img_shape=None, lamb_1=5, rho=10):
    """ MRI应用的ADMM-TV采样器 """
    # 定义预测器和校正器更新函数
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    # 初始化ADMM变量
    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    # MRI前向模型 (傅里叶采样)
    def _A(x):
        return fft2(x) * mask  # 傅里叶变换加掩码
    
    def _AT(kspace):
        return torch.real(ifft2(kspace))  # 反傅里叶变换取实部

    # 共轭梯度法中的A算子
    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))

    # 共轭梯度法求解
    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    # ADMM-TV主循环
    def CS_routine(x, ATy, niter=20):
        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_z = _Dz(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    # 获取更新函数
    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean
        return radon_update_fn

    # 获取ADMM-TV更新函数
    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean
        return ADMM_TV_fn

    # 创建预测器、校正器和ADMM-TV更新函数
    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    # 主采样函数
    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            # 初始化采样
            x = sde.prior_sampling(data.shape).to(data.device)
            
            # 时间步设置
            timesteps = torch.linspace(sde.T, eps, sde.N)
            
            # 主采样循环
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. 分批处理以适应GPU内存
                x_batch = batchfy(x, 20)
                # 2. 对每批执行PC步骤
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_agg.append(x_batch_sing)
                # 3. 聚合结果执行ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. 执行ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)

                # 保存进度
                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(os.path.join(save_root, 'recon', 'progress', f'progress{i}.png'), clear(x_mean[0:1]), cmap='gray')

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon





# MRI应用的ADMM-TV采样器
def get_pc_radon_ADMM_TV_temp(sde, predictor, corrector, inverse_scaler, snr, mask=None,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, save_progress=True, save_root=None,
                             img_shape=None, lamb_1=5, rho=10):
    """ MRI应用的ADMM-TV采样器 """
    # 定义预测器和校正器更新函数
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    eps = 1e-10

    # 获取更新函数
    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean
        return radon_update_fn


    # 创建预测器、校正器和ADMM-TV更新函数
    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)


    # 主采样函数
    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            # 初始化采样
            x = data.clone()           
            # 时间步设置
            timesteps = torch.linspace(sde.T, eps, sde.N)
            
            # 主采样循环
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. 分批处理以适应GPU内存
                x_batch = batchfy(x, 20)
                # 2. 对每批执行PC步骤
                x_agg = list()
                for idx, x_batch_ in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing = x_batch_sing + x_batch_
                    x_agg.append(x_batch_sing)
                # 3. 聚合结果执行ADMM TV
                x = torch.cat(x_agg, dim=0)

                # 保存进度
                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(os.path.join(save_root, 'recon', 'progress', f'progress{i}.png'), clear(x[0:1]), cmap='gray')

            return inverse_scaler(x)

    return pc_radon

