import os
import random
from pathlib import Path
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Dataset(Dataset):
    """支持多种模式的图像数据集加载器，包括条件生成、无条件生成和多条件生成"""
    
    def __init__(
        self,
        folder,                # 数据路径（单个路径或路径列表）
        image_size,            # 输出图像尺寸
        exts=['jpg', 'jpeg', 'png', 'tiff'],  # 支持的图像格式
        augment_flip=False,     # 是否启用随机水平翻转
        convert_image_to=None,  # 图像转换模式（如'L'转灰度）
        condition=0,           # 0=无条件生成, 1=单条件, 2=双条件
        equalizeHist=False,     # 是否应用直方图均衡化
        crop_patch=True,        # 是否随机裁剪小块
        sample=False           # 是否为采样模式（禁用随机裁剪）
    ):
        super().__init__()
        # 初始化参数
        self.equalizeHist = equalizeHist
        self.exts = exts
        self.augment_flip = augment_flip
        self.condition = condition
        self.crop_patch = crop_patch
        self.sample = sample

        # 根据条件模式加载不同数据
        if condition == 1:
            # 单条件模式：folder[0]=GT, folder[1]=输入
            self.gt = self.load_flist(folder[0])    # 加载真实图像路径列表
            self.input = self.load_flist(folder[1]) # 加载条件图像路径列表
        elif condition == 0:
            # 无条件生成模式
            self.paths = self.load_flist(folder)    # 加载单组图像路径
        elif condition == 2:
            # 双条件模式：folder[0]=GT, folder[1]=条件1, folder[2]=条件2
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self.input_condition = self.load_flist(folder[2])

        self.image_size = image_size
        self.convert_image_to = convert_image_to

    def __len__(self):
        """返回数据集大小（根据条件模式选择不同计数）"""
        if self.condition:
            return len(self.input)  # 条件模式以输入图像数为准
        else:
            return len(self.paths)   # 无条件模式使用完整数据集

    def __getitem__(self, index):
        """核心方法：加载并处理指定索引的图像数据"""
        
        # 模式1：单条件生成（如图像修复）
        if self.condition == 1:
            # 加载图像对：GT和条件输入
            img0 = Image.open(self.gt[index])    # 高质量目标图像
            img1 = Image.open(self.input[index])  # 退化后的输入图像
            
            # 格式转换（如转灰度）
            img0 = self.convert_image(img0)
            img1 = self.convert_image(img1)

            # 图像填充到目标尺寸
            img0, img1 = self.pad_img([img0, img1], self.image_size)

            # 随机裁剪小块（训练时启用）
            if self.crop_patch and not self.sample:
                img0, img1 = self.get_patch([img0, img1], self.image_size)

            # 直方图均衡化（可选）
            img1 = self.apply_equalizeHist(img1)

            # 数据增强流程
            images = self.apply_augmentations([[img0, img1]])
            return [self.to_tensor(img) for img in images[0]]

        # 模式0：无条件生成（如标准图像生成）
        elif self.condition == 0:
            path = self.paths[index]
            img = Image.open(path)
            img = self.convert_image(img)
            
            # 单图像处理流程
            img = self.pad_img([img], self.image_size)[0]
            if self.crop_patch and not self.sample:
                img = self.get_patch([img], self.image_size)[0]
            img = self.apply_equalizeHist(img)
            
            images = self.apply_augmentations([[img]])
            return self.to_tensor(images[0][0])

        # 模式2：双条件生成（如多模态输入）
        elif self.condition == 2:
            # 加载三组图像：GT + 两个条件
            img0, img1, img2 = map(Image.open, 
                                 [self.gt[index], 
                                  self.input[index], 
                                  self.input_condition[index]])
            
            # 统一处理流程
            img0, img1, img2 = map(self.convert_image, [img0, img1, img2])
            img0, img1, img2 = self.pad_img([img0, img1, img2], self.image_size)
            
            if self.crop_patch and not self.sample:
                img0, img1, img2 = self.get_patch([img0, img1, img2], self.image_size)
            
            img1 = self.apply_equalizeHist(img1)
            images = self.apply_augmentations([[img0, img1, img2]])
            return [self.to_tensor(img) for img in images[0]]

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]
