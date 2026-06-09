import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import tigre


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
def normalize_dr(dr_tensor):
    eps = 1e-12
    # 计算最小值和最大值
    min_val = dr_tensor.min()
    max_val = dr_tensor.max()

    normalized = (dr_tensor - min_val) / (max_val - min_val + eps)
    
    return normalized
class BaseDataset(Dataset):
    def __init__(self, config , type="train",condition=0,split_ratio=0.9):    
        super().__init__()
        self.type = type
        self.condition = condition
        self.config = config
        ####################################################
        pred =np.load(config['diffusion_data']['root'] + 'pred.npy') 
        label = np.load(config['diffusion_data']['root'] + 'label.npy') 
        ####################################################
        self.all_input = pred
        self.all_gt = label
        max1 = pred.max()
        min1 = pred.min()

        max2 = label.max()
        min2 = label.min()
        ####################################################
        num_samples = self.all_gt.shape[0]
        split_idx = int(num_samples * split_ratio)

        
        if type == "train":
            self.gt = self.all_gt[:]  
            self.input = self.all_input[:]  
        elif type == "val":
            self.all_gt = self.all_gt[split_idx:]   
            self.all_input = self.all_input[split_idx:]  
        else:
            raise ValueError("error: not 'train' or 'val'")

        self.all_gt = np.expand_dims(self.all_gt.astype(np.float32), axis=1)  
        self.all_input = np.expand_dims(self.all_input.astype(np.float32), axis=1)

        if len(self.all_gt) != len(self.all_input):
            raise ValueError("error: len(all_gt) != len(all_input) ")
        
        if config['diffusion_data']['norm_type'] == 'globe_norm':
           self.all_gt = normalize_dr(self.all_gt)
           self.all_input = normalize_dr(self.all_input)
        elif config['diffusion_data']['norm_type'] == 'Increase_tenfold': 
            self.all_gt =  self.all_gt * config['diffusion_data']['scale_factor']
            self.all_input =  self.all_input * config['diffusion_data']['scale_factor']


    def __len__(self):
        return len(self.all_gt)


    def __getitem__(self, idx):
        # 获取原始图像数据并归一化
        if self.config['diffusion_data']['norm_type'] == 'single_norm':
            input =normalize_dr(self.all_input[idx]) 
            gt = normalize_dr(self.all_gt[idx])
        else :
            input =self.all_input[idx]
            gt = self.all_gt[idx]
        if self.condition == 0:
            return gt
        else :
            return [gt,input]
################################################################
##########################载入数据###############################
################################################################
def create_dataloader(cofigs, batch_size,type):

    train_dataset = BaseDataset(cofigs["diffusion_data"]["root"],  "train",condition=cofigs['diffusion_training']['condition_type'],data_type=cofigs['diffusion_data']['dataset'] )
    val_dataset = BaseDataset(cofigs["diffusion_data"]["root"],  "val",condition=cofigs['diffusion_training']['condition_type'],data_type=cofigs['diffusion_data']['dataset'] )


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





























# 
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
#class InpaintDataset(data.Dataset):
#     """
#     图像修复（Inpainting）任务专用数据集类
#     功能：生成带随机掩码的图像数据，模拟图像部分缺失的场景
#     典型应用：训练图像修复模型（如EdgeConnect, DeepFill等）
#     """
#     def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
#         imgs = make_dataset(data_root)
#         if data_len > 0:
#             self.imgs = imgs[:int(data_len)]
#         else:
#             self.imgs = imgs
#         self.tfs = transforms.Compose([
#                 transforms.Resize((image_size[0], image_size[1])),
#                 transforms.ToTensor(),
#                 # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
#         ])
#         self.loader = loader
#         self.mask_config = mask_config
#         self.mask_mode = self.mask_config['mask_mode']
#         self.image_size = image_size

#     def __getitem__(self, index):
#         ret = {}
#         path = self.imgs[index]
#         img = self.tfs(self.loader(path))
#         mask = self.get_mask()
#         cond_image = img*(1. - mask) + mask*torch.randn_like(img)
#         mask_img = img*(1. - mask) + 0.5*mask

#         mask=mask.repeat(3,1,1)
#         # return [img,mask_img,mask]
#         return [img,mask_img]

#     def __len__(self):
#         return len(self.imgs)

#     def get_mask(self):
#         if self.mask_mode == 'bbox':
#             mask = bbox2mask(self.image_size, random_bbox())
#         elif self.mask_mode == 'center':
#             h, w = self.image_size
#             mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
#         elif self.mask_mode == 'irregular':
#             mask = get_irregular_mask(self.image_size)
#         elif self.mask_mode == 'free_form':
#             mask = brush_stroke_mask(self.image_size)
#         elif self.mask_mode == 'hybrid':
#             regular_mask = bbox2mask(self.image_size, random_bbox())
#             irregular_mask = brush_stroke_mask(self.image_size, )
#             mask = regular_mask | irregular_mask
#         elif self.mask_mode == 'file':
#             pass
#         else:
#             raise NotImplementedError(
#                 f'Mask mode {self.mask_mode} has not been implemented.')
#         return torch.from_numpy(mask).permute(2,0,1)

#     def load_name(self, index, sub_dir=False):
#             # condition
#         name = self.imgs[index]
#         if sub_dir == 0:
#             return os.path.basename(name)
#         elif sub_dir == 1:
#             path = os.path.dirname(name)
#             sub_dir = (path.split("/"))[-1]
#             return sub_dir+"_"+os.path.basename(name)

# class UncroppingDataset(data.Dataset):
#     """
#     图像非裁剪修复数据集类（用于图像补全/修复任务）
#     功能：生成带随机掩码的图像数据，模拟图像部分缺失的场景
#     典型应用：训练图像修复模型（如Partial Convolution, GAN-based inpainting）
#     """
#     def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
#         """
#         参数说明:
#             data_root:   字符串, 图像数据根目录路径
#             mask_config: 字典, 掩码生成配置，关键键值:
#                 - mask_mode: 掩码类型 ('manual', 'fourdirection', 'onedirection', 'hybrid')
#                 - shape:     当mask_mode='manual'时指定的固定掩码形状
#             data_len:    整数, 限制加载的数据量（-1表示全部）
#             image_size:  列表, 输出图像尺寸 [高, 宽]
#             loader:      函数, 图像加载方法（默认PIL.Image.open）
#         """
#         imgs = make_dataset(data_root)
#         if data_len > 0:
#             self.imgs = imgs[:int(data_len)]
#         else:
#             self.imgs = imgs
#         self.tfs = transforms.Compose([
#                 transforms.Resize((image_size[0], image_size[1])),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
#         ])
#         self.loader = loader
#         self.mask_config = mask_config
#         self.mask_mode = self.mask_config['mask_mode']
#         self.image_size = image_size

#     def __getitem__(self, index):
#         """
#         获取带掩码的训练样本
#         返回字典包含:
#             - gt_image:    原始完整图像
#             - cond_image:  带噪声的掩码区域+原始可见区域
#             - mask_image:  二值化掩码可视化（调试用）
#             - mask:        掩码矩阵（1表示缺失区域）
#             - path:        文件名（用于调试）
#         """
#         ret = {}
#         path = self.imgs[index]  # 获取图像路径
        
#         # 1. 加载并预处理图像
#         img = self.tfs(self.loader(path))  # [C,H,W]范围[-1,1]

#         # 2. 生成掩码（1表示缺失区域，0表示保留区域）
#         mask = self.get_mask()  # [1,H,W]

#         # 3. 构造条件图像（在掩码区域添加噪声）
#         cond_image = img * (1. - mask) + mask * torch.randn_like(img)

#         # 4. 构造掩码可视化图像（白色表示缺失区域）
#         mask_img = img * (1. - mask) + mask  # 掩码区=1（白色），保留区=原图

#         # 5. 组装返回数据
#         ret['gt_image'] = img          # 原始图像 [3,256,256]
#         ret['cond_image'] = cond_image  # 带噪声的破损图像 [3,256,256]
#         ret['mask_image'] = mask_img    # 可视化掩码 [3,256,256]
#         ret['mask'] = mask              # 纯掩码 [1,256,256]
#         ret['path'] = path.split("/")[-1].split("\\")[-1]  # 提取纯文件名

#         return ret

#     def __len__(self):
#         return len(self.imgs)

#     def get_mask(self):
#         if self.mask_mode == 'manual':
#             mask = bbox2mask(self.image_size, self.mask_config['shape'])
#         elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
#             mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
#         elif self.mask_mode == 'hybrid':
#             if np.random.randint(0,2)<1:
#                 mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
#             else:
#                 mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
#         elif self.mask_mode == 'file':
#             pass
#         else:
#             raise NotImplementedError(
#                 f'Mask mode {self.mask_mode} has not been implemented.')
#         return torch.from_numpy(mask).permute(2,0,1)