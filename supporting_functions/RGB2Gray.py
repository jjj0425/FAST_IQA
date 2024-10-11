import os
import cv2
import numpy as np

# 定义路径
input_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\results\deeplabv3p_b2\img'
output_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\results\deeplabv3p_b2\img_gray'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 定义颜色映射
color_mapping = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3
}

def rgb_to_gray(image, color_mapping):
    h, w, _ = image.shape
    gray_image = np.zeros((h, w), dtype=np.uint8)
    for rgb, gray in color_mapping.items():
        mask = np.all(image == np.array(rgb), axis=-1)
        gray_image[mask] = gray
    return gray_image

# 处理文件夹中的每张图片
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取RGB图像
        rgb_image = cv2.imread(input_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # 转为灰度图像
        gray_image = rgb_to_gray(rgb_image, color_mapping)
        
        # 保存灰度图像
        cv2.imwrite(output_path, gray_image)

print('转换完成并保存到目标文件夹。')