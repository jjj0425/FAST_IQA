import os
import shutil

# 定義來源資料夾和目標資料夾
origin_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\img_resize\origin'
sigmoid_png_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\deeplabv3'
target_base_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification\deeplabv3'

# 確保目標基礎資料夾存在
os.makedirs(target_base_dir, exist_ok=True)

# 迭代 origin 資料夾中的每個子資料夾 (class1, class2, ...)
for class_name in os.listdir(origin_dir):
    class_origin_path = os.path.join(origin_dir, class_name)
    
    # 確認這是個資料夾
    if os.path.isdir(class_origin_path):
        # 在目標基礎資料夾中創建相同的子資料夾
        class_target_path = os.path.join(target_base_dir, class_name)
        os.makedirs(class_target_path, exist_ok=True)
        
        # 迭代該子資料夾中的每個檔案
        for filename in os.listdir(class_origin_path):
            file_stem = os.path.splitext(filename)[0]
            # 尋找相應的 sigmoid_png 檔案
            sigmoid_png_path = os.path.join(sigmoid_png_dir, file_stem + '.png')
            if os.path.exists(sigmoid_png_path):
                # 將檔案複製到相應的目標子資料夾中
                target_path = os.path.join(class_target_path, file_stem + '.png')
                shutil.move(sigmoid_png_path, target_path)

print("所有影像已成功歸類。")
