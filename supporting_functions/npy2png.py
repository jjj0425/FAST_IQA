import os
import numpy as np
from PIL import Image

# 來源目錄和目標目錄
source_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid'
target_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\sigmoid_png'

# 確保目標目錄存在
os.makedirs(target_dir, exist_ok=True)

# 遍歷所有 .npy 檔案
for filename in os.listdir(source_dir):
    if filename.endswith('.npy'):
        # 讀取 .npy 檔案
        npy_path = os.path.join(source_dir, filename)
        data = np.load(npy_path)
        
        # 確保形狀為 3*256*256
        if data.shape == (3, 256, 256):
            # 轉置為 256*256*3
            data = np.transpose(data, (1, 2, 0))
            
            # 轉換為 PIL Image 並保存為 PNG
            image = Image.fromarray((data * 255).astype(np.uint8))
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(target_dir, png_filename)
            image.save(png_path)

print("所有 .npy 檔案已成功轉換為 PNG 檔案並保存。")
