import os
import numpy as np

# 指定原始数据所在文件夹和目标文件夹
source_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\result\sm_prediction'
target_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\result\sm_prediction'

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的所有.npy文件
for file_name in os.listdir(source_folder):
    if file_name.endswith('.npy'):
        file_path = os.path.join(source_folder, file_name)
        
        # 加载.npy文件
        data = np.load(file_path)
        
        # 删除第一个通道
        data = np.delete(data, 0, axis=0)
        
        # 保存修改后的数据到新文件夹
        target_file_path = os.path.join(target_folder, file_name)
        np.save(target_file_path, data)
        print(f"Processed {file_name} and saved to {target_file_path}")

print("All files have been processed.")
