import os
import shutil

# 定義路徑
origin_path = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\img_origin'
resize_path = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\segmentation'

# 定義類別
categories = ['class1', 'class2', 'class3', 'class4', 'class5']

# 逐一檢查每個類別資料夾
for category in categories:
    origin_dir = os.path.join(origin_path, category)
    resize_dir = resize_path
    new_dir = os.path.join(resize_dir, category)
    
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    # 讀取原始資料夾中的所有檔名
    for filename in os.listdir(origin_dir):
        origin_file = os.path.join(origin_dir, filename)
        resize_file = os.path.join(resize_dir, filename)
        new_file = os.path.join(new_dir, filename)
        
        # 如果縮放資料夾中有同樣的檔名，則移動該檔案
        if os.path.exists(resize_file):
            shutil.move(resize_file, new_dir)

print("移動完成！")
