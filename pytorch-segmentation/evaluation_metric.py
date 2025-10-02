import numpy as np
import cv2
import os

def calculate_metrics_per_image(gt_labels, pred_labels, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    conf_matrix += calculate_confusion_matrix(gt_labels, pred_labels, num_classes)

    global_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    ious = np.zeros(num_classes)
    valid_classes = []
    for cls in range(num_classes):
        intersection = conf_matrix[cls, cls]
        union = np.sum(conf_matrix[cls, :]) + np.sum(conf_matrix[:, cls]) - intersection
        if np.sum(conf_matrix[cls, :]) > 0:
            ious[cls] = intersection / union if union != 0 else 0
            valid_classes.append(cls)
    
    mean_iou = np.mean([ious[cls] for cls in valid_classes])
    
    return global_accuracy, ious, mean_iou

def rgb_to_class_indices(rgb_img):
    # 定义每种类别的RGB值
    rgb_to_class = {
        (0, 0, 0): 0,   #Background
        (0, 0, 128): 1, #Right kidney
        (0, 128, 0): 2, #Diaphragm
        (0, 128, 128): 3, #LIVER
        # 可以根据需要添加更多类别
    }
    
    height, width, _ = rgb_img.shape
    class_indices = np.zeros((height, width), dtype=np.int64)
    
    for rgb, cls_idx in rgb_to_class.items():
        mask = (rgb_img == rgb).all(axis=2)
        class_indices[mask] = cls_idx
    
    return class_indices

def calculate_confusion_matrix(gt_labels, pred_labels, num_classes):
    mask = (gt_labels >= 0) & (gt_labels < num_classes)
    conf_matrix = np.bincount(
        num_classes * gt_labels[mask] + pred_labels[mask],
        minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return conf_matrix

# 定义ground truth和预测结果的文件夹路径
gt_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\UNET\val\mask'
pred_folder = r'C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\UNet\val\result_SegNet'

# 定义类别数量
num_classes = 4  # 例如，4个类别

# 遍历所有的图像文件并计算每张图像的全局准确率和每个类别的IoU
for filename in os.listdir(gt_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        gt_path = os.path.join(gt_folder, filename)
        pred_path = os.path.join(pred_folder, filename)
        
        # 读取图像
        gt_img = cv2.imread(gt_path)
        pred_img = cv2.imread(pred_path)
        
        # 检查图像尺寸是否匹配
        if gt_img.shape != pred_img.shape:
            print(f"Error: Shape mismatch for {filename}")
            continue
        
        # 将RGB图像转换为类别索引图
        gt_labels = rgb_to_class_indices(gt_img)
        pred_labels = rgb_to_class_indices(pred_img)
        
        # 计算每张图像的全局准确率和每个类别的IoU
        global_accuracy, ious, mean_iou = calculate_metrics_per_image(gt_labels, pred_labels, num_classes)
        
        # 输出结果
        print(f"Results for {filename}:")
        print(f"  Global Accuracy: {global_accuracy:.4f}")
        for cls in range(num_classes):
            print(f"  IoU for class {cls}: {ious[cls]:.4f}")
        print(f"  Mean IoU: {mean_iou:.4f}")





