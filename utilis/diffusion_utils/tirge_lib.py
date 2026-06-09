from tigre.utilities.pytorch_bindings import create_pytorch_operator
import tigre
import numpy as np
import torch
import tigre.utilities.gpu as gpu

def tigre_sin_consist(geo,angles,x,y,tigre_devices):
  geo.nVoxel[0] = 1      # 单层
  geo.sVoxel[0] = 1.0    # z 方向物理尺寸 = 1 mm（或其他小值）
  geo.dVoxel[0] = 1.0    # z 方向体素间距 = 1 mm/voxel
  tigre_devices   = gpu.getGpuIds()
  tigre_devices = tigre_devices[0]
  for i,j in zip(x,y):
    ax, atb = create_pytorch_operator(geo, angles, tigre_devices)
    i_update = finite_difference_update_vectorized(i, j, ax)
    pass

  return 1



def finite_difference_update_vectorized(x, y, ax, lambda_consistency=0.1, eps=1e-3):
    grad = torch.zeros_like(x)
    original_x = x.detach()
    
    # 一次性生成所有扰动
    for dim in [1, 2]:  # 仅对H,W维度扰动（假设C=1）
        # 正向扰动
        x_plus = original_x.clone()
        x_plus[:, dim] += eps
        loss_plus = torch.mean((ax(x_plus) - ax(y)) ** 2, dim=[1,2])
        
        # 负向扰动
        x_minus = original_x.clone()
        x_minus[:, dim] -= eps
        loss_minus = torch.mean((ax(x_minus) - ax(y)) ** 2, dim=[1,2])
        
        # 计算梯度
        grad[:, dim] = (loss_plus - loss_minus) / (2 * eps)
    
    return original_x - lambda_consistency * grad