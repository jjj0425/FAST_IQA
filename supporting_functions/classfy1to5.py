import os
import shutil
import pandas as pd

# 定義文件和目錄路徑
csv_path = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\score\score.csv"
img_source_dir = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\img_all"
classification_dir = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\classification"
others_dir = os.path.join(classification_dir, "others")

# 讀取CSV文件
df = pd.read_csv(csv_path)

# 檢查分類目錄是否存在，不存在則創建
if not os.path.exists(classification_dir):
    os.makedirs(classification_dir)
if not os.path.exists(others_dir):
    os.makedirs(others_dir)

# 遍歷每一行數據
for index, row in df.iterrows():
    score1 = row['score1']
    score2 = row['score2']
    filename = row['filename']
    
    # 確定影像文件路徑
    source_file = os.path.join(img_source_dir, filename)
    
    if score1 == score2:
        # 如果分數一致，將影像存入對應分數的資料夾
        score_dir = os.path.join(classification_dir, str(score1))
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        destination_file = os.path.join(score_dir, filename)
    else:
        # 如果分數不一致，將影像存入others資料夾
        destination_file = os.path.join(others_dir, filename)
    
    # 複製影像文件
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)
    else:
        print(f"File not found: {source_file}")
