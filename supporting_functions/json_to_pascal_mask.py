import os
import json
import numpy as np
import cv2

# 定义路径
json_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\label_new'
output_dir = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\sem_mask'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 定义类别与像素值的映射
label_map = {
    'Liver': 1,
    'RightKidney': 2,
    'diaphgram': 3
}

# 处理每个 JSON 文件
for json_file in os.listdir(json_dir):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_dir, json_file)

        # 读取 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取图像的宽和高
        img_height = data['imageHeight']
        img_width = data['imageWidth']

        # 创建一个空白的灰度图像
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # 解析标注的形状
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            # 将多边形转换为整数类型的 numpy 数组
            pts = np.array(points, dtype=np.int32)

            # 获取对应的像素值
            pixel_value = label_map.get(label, 0)

            # 填充多边形
            cv2.fillPoly(mask, [pts], pixel_value)

        # 保存生成的灰度图像
        output_path = os.path.join(output_dir, json_file.replace('.json', '.png'))
        cv2.imwrite(output_path, mask)

print('转换完成！')
