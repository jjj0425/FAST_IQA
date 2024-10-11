import os
from PIL import Image

# 定义要处理的目录路径
input_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\mask'
output_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\mask_256'

# 确保输出目录存在，如果不存在则创建
os.makedirs(output_dir, exist_ok=True)

# 定义目标大小
target_size = (256, 256)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        # 打开图像文件
        img_path = os.path.join(input_dir, filename)
        with Image.open(img_path) as img:
            # 调整图像大小
            resized_img = img.resize(target_size)
            
            # 保存调整后的图像到输出目录
            resized_img_path = os.path.join(output_dir, filename)
            resized_img.save(resized_img_path)

print("所有图像已调整大小并保存。")