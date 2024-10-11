import os
from PIL import Image

def convert_grayscale_to_rgb(image_path, output_path):
    # 打开灰度图像
    with Image.open(image_path) as img_gray:
        # 创建一个新的RGB图像
        img_rgb = Image.new("RGB", img_gray.size)
        
        # 遍历每个像素，并根据灰度值转换为RGB值
        width, height = img_gray.size
        for y in range(height):
            for x in range(width):
                pixel_value = img_gray.getpixel((x, y))
                if pixel_value == 0:
                    img_rgb.putpixel((x, y), (0, 0, 0))
                elif pixel_value == 1:
                    img_rgb.putpixel((x, y), (255, 0, 0))
                elif pixel_value == 2:
                    img_rgb.putpixel((x, y), (0, 255, 0))
                elif pixel_value == 3:
                    img_rgb.putpixel((x, y), (0, 0, 255))
                else:
                    img_rgb.putpixel((x, y), (255, 255, 255))
        
        # 保存转换后的RGB图像
        img_rgb.save(output_path)

# 定义输入和输出目录路径
input_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\results\GLFR_b2'
output_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\results\GLFR_b2_RGB'

# 如果输出目录不存在，则创建
os.makedirs(output_folder, exist_ok=True)

# 遍历输入目录中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        # 获取图像文件的完整路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # 将灰度图像转换为RGB图像
        convert_grayscale_to_rgb(input_path, output_path)

print("所有灰度图像已转换为RGB图像并保存。")
