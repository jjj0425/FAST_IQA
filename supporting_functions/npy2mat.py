import os
import numpy as np
import scipy.io

# 輸入和輸出文件夾路徑
input_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\UNet\val\result_test_no_channel1'
output_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\UNet\val\result_test_no_channel1_mat'

# 確保輸出文件夾存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍歷文件夾中的所有.npy檔案
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        # 構建輸入和輸出文件的完整路徑
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename[:-4] + '.mat')

        # 讀取.npy檔案
        data = np.load(input_file)

        # 將數據保存為.mat檔案
        scipy.io.savemat(output_file, {'data': data})

        print(f"Converted {input_file} to {output_file}")
