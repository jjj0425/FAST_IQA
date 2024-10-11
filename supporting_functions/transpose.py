import os
import numpy as np

# 指定文件夹路径
folder_path = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\UNet\val\result_test"

# 遍历文件夹中的所有.npy文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.npy'):
        file_path = os.path.join(folder_path, file_name)
        
        # 加载 .npy 文件
        data = np.load(file_path)
        
        # 检查数据形状是否为 (4, 256, 256)
        if data.shape == (4, 256, 256):
            # 转置数组形状为 (256, 256, 4)
            transformed_data = np.transpose(data, (1, 2, 0))
            
            # 将转换后的数组保存回文件
            np.save(file_path, transformed_data)
            print(f"Processed {file_name} and saved transformed data.")
        else:
            print(f"File {file_name} has an unexpected shape: {data.shape}")

print("All files have been processed.")
