import os
import numpy as np
from PIL import Image

def calculate_mean_std(image_dir):
    pixel_values = []

    # 遍歷資料夾中的所有影像
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            with Image.open(image_path) as img:
                img = img.convert('RGB')  # 確保影像為RGB格式
                img_array = np.array(img)
                pixel_values.append(img_array)

    # 確保有影像被讀取
    if not pixel_values:
        raise ValueError("未找到任何影像或影像格式不支援")

    # 將所有影像像素值串聯成一個大的陣列
    all_pixels = np.concatenate([img.reshape(-1, 3) for img in pixel_values], axis=0)

    # 計算平均值和標準差
    mean = np.mean(all_pixels, axis=0)
    std = np.std(all_pixels, axis=0)

    return mean, std

# 設定影像資料夾路徑
image_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\validation\result_deeplabv3p_nocrf\img'  # 設定為您的影像資料夾

mean, std = calculate_mean_std(image_dir)
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
