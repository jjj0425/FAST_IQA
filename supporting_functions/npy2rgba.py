import os
import numpy as np
from PIL import Image

# 定义输入文件夹路径和输出文件夹路径
input_folder = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\UNet\val\result_test"
output_folder = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\UNet\val\result_test_png"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有 .npy 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.npy'):
        # 构建文件路径
        file_path = os.path.join(input_folder, file_name)
        
        # 从 .npy 文件中加载数据
        data = np.load(file_path)
        
        # 确保数据形状为 4x256x256
        if data.shape != (4, 256, 256):
            raise ValueError(f"Invalid shape {data.shape} for file {file_name}. Expected shape is (4, 256, 256).")
        
        # 转换数据到 0-255 范围
        data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        
        # 创建一个空白的 RGBA 图像
        image = np.zeros((256, 256, 4), dtype=np.uint8)
        
        # 将每个通道的数据分别赋值给 RGBA 图像的 R、G、B、A 通道
        for i in range(4):
            image[:, :, i] = data[i]
        
        # 构建输出文件路径
        output_file_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".png")
        
        # 保存图像为 .png 文件
        Image.fromarray(image).save(output_file_path)

print("Conversion complete.")
