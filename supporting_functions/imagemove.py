import os
import cv2
import numpy as np

# 定義路徑
train_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\validation\mask_rgb_256'

# 定義顏色
colors = {
    "(0, 0, 0)": (0, 0, 0),
    "(255, 0, 0)": (255, 0, 0),
    "(0, 255, 0)": (0, 255, 0),
    "(0, 0, 255)": (0, 0, 255)
}

# 初始化像素計數器
pixel_counts = {
    "(0, 0, 0)": 0,
    "(255, 0, 0)": 0,
    "(0, 255, 0)": 0,
    "(0, 0, 255)": 0
}

# 讀取資料夾中的每張影像並統計像素數量
for filename in os.listdir(train_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):  # 根據實際影像格式進行調整
        image_path = os.path.join(train_folder, filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        if image is not None:
            for color_name, color_value in colors.items():
                # 計算每個顏色的像素數量
                mask = cv2.inRange(image, np.array(color_value), np.array(color_value))
                pixel_counts[color_name] += cv2.countNonZero(mask)
        else:
            print(f'Failed to read image: {image_path}')
            
# 打印統計結果
for color_name, count in pixel_counts.items():
    print(f'Color {color_name}: {count} pixels')

print('Pixel counting completed.')
