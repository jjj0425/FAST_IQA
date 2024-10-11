import os
import shutil
from pathlib import Path

# 設定路徑
train_txt_path = Path(r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\ImageSet\train.txt")
src_img_dir = Path(r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\mask_256")
dst_img_dir = Path(r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\train\mask")

# 創建目標資料夾如果不存在的話
dst_img_dir.mkdir(parents=True, exist_ok=True)

# 讀取 train.txt 檔案中的檔名
with open(train_txt_path, 'r') as file:
    image_names = file.read().splitlines()

# 複製符合的影像檔
for image_name in image_names:
    src_img_path = src_img_dir / image_name
    dst_img_path = dst_img_dir / image_name

    # 確認檔案存在後再進行複製
    if src_img_path.exists():
        shutil.copy(src_img_path, dst_img_path)
        print(f"Copied {src_img_path} to {dst_img_path}")
    else:
        print(f"File {src_img_path} does not exist")

print("Finished copying images.")
