import os
import numpy as np
from PIL import Image
from typing import Optional

def png_folder_to_3d_gray_npy(
    input_folder: str,
    output_npy_path: str,
    target_size: Optional[tuple] = None,
    sort_files: bool = True
) -> None:
    """
    将文件夹中的所有PNG图片转换为**灰度图**，并整合为3D Numpy数组保存为.npy文件
    最终数组维度：[样本数, 高度, 宽度]（单通道灰度，无额外通道维度）

    参数:
        input_folder: str - 存放PNG图片的文件夹路径
        output_npy_path: str - 输出.npy文件的路径（如 "./gray_images.npy"）
        target_size: tuple - 可选，统一所有图片的尺寸 (height, width)，None则使用原图尺寸
        sort_files: bool - 是否按文件名排序，默认True（保证样本顺序可预测）

    异常:
        FileNotFoundError - 输入文件夹不存在
        ValueError - 文件夹中无PNG图片、图片尺寸不统一（未指定target_size时）
        RuntimeError - 单张图片读取/处理失败
    """
    # 1. 基础校验：文件夹是否存在
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"输入文件夹不存在，请检查路径：{input_folder}")
    
    # 2. 筛选所有PNG文件（兼容大小写：.png/.PNG）
    png_file_paths = [
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.lower().endswith(".png")
    ]
    
    # 校验是否有有效PNG文件
    if len(png_file_paths) == 0:
        raise ValueError(f"文件夹 {input_folder} 中未找到任何PNG图片！")
    
    # 3. 按文件名排序（保证样本顺序稳定）
    if sort_files:
        png_file_paths.sort()
    print(f"共找到 {len(png_file_paths)} 张PNG图片，开始转换为灰度图...")
    
    # 4. 批量读取并转换为灰度图
    gray_image_arrays = []
    for idx, img_path in enumerate(png_file_paths):
        try:
            # 打开图片并强制转换为8位灰度图（L模式）
            with Image.open(img_path) as img:
                gray_img = img.convert("L")  # L模式：8位灰度图（0-255）
                
                # 统一尺寸（如需）：resize参数是 (width, height)，需反转target_size
                if target_size is not None:
                    gray_img = gray_img.resize(
                        target_size[::-1],  # 反转：(h, w) → (w, h)
                        Image.Resampling.LANCZOS  # 高质量缩放
                    )
                
                # 转换为numpy数组（形状：[h, w]）并加入列表
                gray_arr = np.array(gray_img, dtype=np.uint8)  # 灰度图默认用uint8存储
                gray_image_arrays.append(gray_arr)
                
                # 打印进度（每10张/最后1张反馈）
                if (idx + 1) % 10 == 0 or idx + 1 == len(png_file_paths):
                    print(f"已处理：{idx + 1}/{len(png_file_paths)} 张")
        
        except Exception as e:
            raise RuntimeError(f"处理图片 {img_path} 时出错：{str(e)}")
    
    # 5. 校验所有灰度图尺寸是否统一（未指定target_size时）
    if target_size is None:
        first_shape = gray_image_arrays[0].shape
        for idx, arr in enumerate(gray_image_arrays):
            if arr.shape != first_shape:
                raise ValueError(
                    f"灰度图尺寸不统一！\n"
                    f"第1张图片尺寸：{first_shape}（h×w）\n"
                    f"第{idx+1}张图片 {png_file_paths[idx]} 尺寸：{arr.shape}（h×w）\n"
                    f"请指定 target_size 参数统一所有图片尺寸。"
                )
    
    # 6. 拼接为3D数组（第一维为样本）
    # 最终形状：[样本数, 高度, 宽度]
    final_3d_array = np.stack(gray_image_arrays, axis=0)
    
    # 7. 保存为npy文件
    np.save(output_npy_path, final_3d_array)
    
    # 输出结果信息
    print("\n===== 转换完成 =====")
    print(f"输出文件路径：{output_npy_path}")
    print(f"3D数组形状：{final_3d_array.shape} → [样本数, 高度, 宽度]")
    print(f"数组数据类型：{final_3d_array.dtype}（灰度图默认uint8，0-255）")




png_folder_to_3d_gray_npy(
    input_folder="/home/b/code-new/NERV/NERF_MINI/data/dataset_xiaoshu/train1/64png_back",  # 替换为你的PNG文件夹路径
    output_npy_path="/home/b/code-new/NERV/NERF_MINI/data/dataset_xiaoshu/train1/64png_back"    # 输出的npy文件路径
)