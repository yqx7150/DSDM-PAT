from torch.utils.data import Dataset, DataLoader
import numpy as np
#from pathlib import Path
from skimage.transform import resize
#import tensorflow as tf
#from skimage.transform import iradon
import pickle
import torch.nn.functional as F
import tigre
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize_dr(dr_tensor):

    # 计算最小值和最大值
    min_val = dr_tensor.min()
    max_val = dr_tensor.max()
    
    # 避免除以零
    if max_val - min_val > 0:
        # 归一化到[0,1]范围
        normalized = (dr_tensor - min_val) / (max_val - min_val)
    else:
        # 如果所有值相同，则设为0.5
        normalized = torch.ones_like(dr_tensor) * 0.5
    
    return normalized

class ConeGeometry(object):
    def __init__(self, data):
        self.geo = tigre.geometry(mode="cone") 

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.geo.DSD = data["DSD"]/1000 # Distance Source to Detector      (m) x射线发射源到x射线接收器之间的距离
        self.geo.DSO = data["DSO"]/1000  # Distance Source Origin        (m) 发射源到起点之间的距离

        # Detector parameters
        self.geo.nDetector = np.array(data["nDetector"])  # number of pixels              (px)
        self.geo.dDetector = np.array(data["dDetector"])/1000  # size of each pixel            (m)
        self.geo.sDetector = self.geo.nDetector * self.geo.dDetector  # total size of the detector    (m)
        
        # Image parameters
        self.geo.nVoxel = np.array(data["nVoxel"])  # number of voxels              (vx)
        self.geo.dVoxel = np.array(data["dVoxel"])/1000  # size of each voxel            (m)
        self.geo.sVoxel = self.geo.nVoxel * self.geo.dVoxel  # total size of the image       (m)

        # Offsets
        self.geo.offOrigin = np.array(data["offOrigin"])/1000  # Offset of image from origin   (m)
        self.geo.offDetector = np.array(data["offDetector"])/1000  # Offset of Detector            (m)

        # Auxiliary
        self.geo.accuracy = data["accuracy"]  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.geo.mode = data["mode"]  # parallel, cone                ...
        self.geo.filter = data["filter"]
def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config['diffusion_data']['centered']:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x
def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config['diffusion_data']['centered']:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x
def resize_data(data, target_size, mode='reflect', anti_aliasing=True):
    """
    对输入的数据进行缩放。

    参数:
        data (np.ndarray): 输入数据，可以是 2D 或 3D 数组。
        target_size (tuple): 目标大小，例如 (256, 256)。
        mode (str): 缩放时的边界处理模式，默认为 'reflect'。
        anti_aliasing (bool): 是否启用抗锯齿，默认为 True。

    返回:
        np.ndarray: 缩放后的数据。
    """
    # 如果数据是 2D（单通道图像），直接缩放
    if len(data.shape) == 2:
        return resize(data, target_size, mode=mode, anti_aliasing=anti_aliasing)
    # 如果数据是 3D（多通道图像），对每个通道分别缩放
    elif len(data.shape) == 3:
        resized_data = np.zeros((data.shape[0], target_size[0], target_size[1]))
        for i in range(data.shape[0]):
            resized_data[i] = resize(data[i], target_size, mode=mode, anti_aliasing=anti_aliasing)
        return resized_data
    else:
        raise ValueError("输入数据的维度必须是 2D 或 3D")
################################################################
####################AAPM数据集训练图像域#########################
################################################################
class AAPM(Dataset):
    def __init__(self, path, type="train", device="cuda", split_ratio=0.9):    
        super().__init__()
        self.device = device
        self.type = type
        
        # 加载数据并转置维度为(样本数, 高度, 宽度)
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        self.all_image = np.transpose(data['image'], (2, 0, 1))  # 形状变为(560, 512, 512)
        self.geo = ConeGeometry(data) # 把数据处理成ConeGeometry
        self.angles = data["all"]["angles"]

        # 数据集分割
        num_samples = self.all_image.shape[0]
        split_idx = int(num_samples * split_ratio)
        
        if type == "train":
            self.images = self.all_image[:]  
        elif type == "val":
            self.images = self.all_image[split_idx:]  
        else:
            raise ValueError("type must be either 'train' or 'val'")
        
        # 转换为torch张量并添加通道维度 (B,H,W) -> (B,1,H,W)
        self.images = torch.from_numpy(self.images).float().unsqueeze(1).to(device)
        self.images = F.interpolate(self.images, size=(256, 256), mode='bilinear', align_corners=False)
        pass

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 直接返回原始图像数据
        return self.images[idx]

################################################################
####################AAPM数据集训练DR投影图#######################
################################################################
class AAPMDR(Dataset):
    def __init__(self, path, type="train", device="cuda", split_ratio=0.9):    
        super().__init__()
        self.device = device
        self.type = type
        
        # 加载数据并转置维度为(样本数, 高度, 宽度)
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        self.all_DR = data['all']['projections']
        self.geo = ConeGeometry(data) # 把数据处理成ConeGeometry
        self.angles = data["all"]["angles"]

        # 数据集分割
        num_samples = self.all_DR.shape[0]
        split_idx = int(num_samples * split_ratio)
        
        if type == "train":
            self.DR = self.all_DR[:]  
        elif type == "val":
            self.DR = self.all_DR[split_idx:]  
        else:
            raise ValueError("type must be either 'train' or 'val'")
        
        # 转换为torch张量并添加通道维度 (B,H,W) -> (B,1,H,W)
        self.DR = torch.from_numpy(self.DR).float().unsqueeze(1).to(device)
        pass

    def __len__(self):
        return len(self.DR)
    
    def __getitem__(self, idx):
        # 获取原始图像数据并归一化
        dr = self.DR[idx]
        normalized_dr = normalize_dr(dr)
        return normalized_dr
    

################################################################
##########################载入数据###############################
################################################################
class AAPM_new(Dataset):
    def __init__(self, path, type="train",condition=0, split_ratio=0.9,data_type='AAMP_DR'):    
        super().__init__()
        #self.device = device
        self.type = type
        self.data_type =data_type
        self.condition = condition
        
        # 加载数据并转置维度为(样本数, 高度, 宽度)
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        
        self.geo = ConeGeometry(data) # 把数据处理成ConeGeometry
        self.angles = data["all"]["angles"]
        
        self.all_DR_gt = data['all']['projections']
        self.all_DR_input = data['all']['pred_projections']

        self.image_gt = np.transpose(data['image'], (2, 0, 1))  
        self.image_input = np.transpose(data['noise_image'], (2, 0, 1))   

        if data_type == 'AADM_DR':
            # 数据集分割
            num_samples = self.all_DR.shape[0]
            split_idx = int(num_samples * split_ratio)
            
            if type == "train":
                self.DR_gt = self.all_DR_gt[:]  
                self.DR_input = self.all_DR_input[:]  
            elif type == "val":
                self.DR_gt = self.all_DR_gt[split_idx:]  
                self.DR_input = self.all_DR_input[split_idx:]  
            else:
                raise ValueError("error: not 'train' or 'val'")
            
            # 转换为torch张量并添加通道维度 (B,H,W) -> (B,1,H,W)
            self.DR_gt = torch.from_numpy(self.DR_gt).float().unsqueeze(1)
            self.DR_input = torch.from_numpy(self.DR_input).float().unsqueeze(1)
            pass

        elif data_type == 'AADM_image':
            # 数据集分割
            num_samples = self.image_gt.shape[0]
            split_idx = int(num_samples * split_ratio)
            
            if type == "train":
                self.images_gt = self.image_gt[:]  
                self.images_input = self.image_input[:]  
            elif type == "val":
                self.images_gt = self.image_gt[split_idx:]  
                self.images_input = self.image_input[split_idx:]  
            else:
                raise ValueError("not 'train' or 'val'")
            
            # 转换为torch张量并添加通道维度 (B,H,W) -> (B,1,H,W)
            self.images = torch.from_numpy(self.images).float().unsqueeze(1)
            pass


    def __len__(self):
        if self.data_type == 'AAPM_DR' :
            return len(self.DR_gt)
        elif self.data_type == 'AAPM_image' :
            return len(self.images_gt)
        else :
            raise ValueError("data_type error")

    
    def __getitem__(self, idx):
        # 获取原始图像数据并归一化
        if self.data_type == 'AAPM_DR' :
            gt = normalize_dr(self.DR_gt[idx])
            input =normalize_dr(self.DR_input[idx]) 
        elif self.data_type == 'AAPM_image' :
            gt = normalize_dr(self.images_gt[idx])
            input =normalize_dr(self.image_input[idx]) 
        else:
            raise ValueError("data_type error")
        if self.condition == 0:
            return gt
        elif self.condition == 1:
            return input,gt

################################################################
##########################载入数据###############################
################################################################
def create_dataloader(cofigs, batch_size,type):
  if cofigs['diffusion_data']['dataset'] == 'AAPM' :
    train_dataset = AAPM(cofigs["diffusion_data"]["root"],  "train", cofigs['device'])
    val_dataset = AAPM(cofigs["diffusion_data"]["root"],  "val", cofigs['device'])
  elif cofigs['diffusion_data']['dataset'] == 'AAPMDR' :
    train_dataset = AAPMDR(cofigs["diffusion_data"]["root"],  "train", cofigs['device'])
    val_dataset = AAPMDR(cofigs["diffusion_data"]["root"],  "val", cofigs['device'])
  else :
     raise ValueError("未知的数据集名称:"+ cofigs['diffusion_data']['dataset'])
  
  if type == 'train':
    data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
    )

  elif type == 'test':
    data_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
  )
  else :
    raise ValueError(f"错误:{type}")

  return data_loader 