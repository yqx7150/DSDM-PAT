import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    """
    图像修复（Inpainting）任务专用数据集类
    功能：生成带随机掩码的图像数据，模拟图像部分缺失的场景
    典型应用：训练图像修复模型（如EdgeConnect, DeepFill等）
    """
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + 0.5*mask

        mask=mask.repeat(3,1,1)
        # return [img,mask_img,mask]
        return [img,mask_img]

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

    def load_name(self, index, sub_dir=False):
            # condition
        name = self.imgs[index]
        if sub_dir == 0:
            return os.path.basename(name)
        elif sub_dir == 1:
            path = os.path.dirname(name)
            sub_dir = (path.split("/"))[-1]
            return sub_dir+"_"+os.path.basename(name)

class UncroppingDataset(data.Dataset):
    """
    图像非裁剪修复数据集类（用于图像补全/修复任务）
    功能：生成带随机掩码的图像数据，模拟图像部分缺失的场景
    典型应用：训练图像修复模型（如Partial Convolution, GAN-based inpainting）
    """
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        """
        参数说明:
            data_root:   字符串, 图像数据根目录路径
            mask_config: 字典, 掩码生成配置，关键键值:
                - mask_mode: 掩码类型 ('manual', 'fourdirection', 'onedirection', 'hybrid')
                - shape:     当mask_mode='manual'时指定的固定掩码形状
            data_len:    整数, 限制加载的数据量（-1表示全部）
            image_size:  列表, 输出图像尺寸 [高, 宽]
            loader:      函数, 图像加载方法（默认PIL.Image.open）
        """
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        """
        获取带掩码的训练样本
        返回字典包含:
            - gt_image:    原始完整图像
            - cond_image:  带噪声的掩码区域+原始可见区域
            - mask_image:  二值化掩码可视化（调试用）
            - mask:        掩码矩阵（1表示缺失区域）
            - path:        文件名（用于调试）
        """
        ret = {}
        path = self.imgs[index]  # 获取图像路径
        
        # 1. 加载并预处理图像
        img = self.tfs(self.loader(path))  # [C,H,W]范围[-1,1]

        # 2. 生成掩码（1表示缺失区域，0表示保留区域）
        mask = self.get_mask()  # [1,H,W]

        # 3. 构造条件图像（在掩码区域添加噪声）
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)

        # 4. 构造掩码可视化图像（白色表示缺失区域）
        mask_img = img * (1. - mask) + mask  # 掩码区=1（白色），保留区=原图

        # 5. 组装返回数据
        ret['gt_image'] = img          # 原始图像 [3,256,256]
        ret['cond_image'] = cond_image  # 带噪声的破损图像 [3,256,256]
        ret['mask_image'] = mask_img    # 可视化掩码 [3,256,256]
        ret['mask'] = mask              # 纯掩码 [1,256,256]
        ret['path'] = path.split("/")[-1].split("\\")[-1]  # 提取纯文件名

        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

#图像染色任务
class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        # 参数说明：
        # data_root: 数据集根目录（包含color/和gray/子目录）
        # data_flist: 指定使用的文件列表（如train.txt）
        # data_len: 限制加载的数据量（-1表示全部）
        # image_size: 统一调整的图像尺寸
        # loader: 图像加载函数（默认PIL.Image.open）
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        # 4. 组装返回数据
        ret['gt_image'] = img         # 彩色目标图像
        ret['cond_image'] = cond_image # 灰度条件图像
        ret['path'] = file_name       # 原始文件名（用于结果可视化时追踪）
        return ret

    def __len__(self):
        return len(self.flist)




