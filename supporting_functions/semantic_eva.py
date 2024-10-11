import os
import numpy as np
import cv2
import pandas as pd

# Define class colors
CLASS_COLORS = {
    "Liver": (255, 0, 0),
    "Kidney": (0, 255, 0),
    "Diaphragm": (0, 0, 255),
    "Background": (0, 0, 0)
}

# Function to calculate IoU
def calculate_iou(pred, gt, class_color):
    pred_mask = np.all(pred == class_color, axis=-1)
    gt_mask = np.all(gt == class_color, axis=-1)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return np.nan
    else:
        return intersection / union

# Function to calculate global accuracy
def calculate_global_accuracy(pred, gt):
    correct = np.all(pred == gt, axis=-1).sum()
    total = pred.shape[0] * pred.shape[1]
    return correct / total

# Main function to evaluate segmentation
def evaluate_segmentation(pred_folder, gt_folder):
    results = []
    pred_files = os.listdir(pred_folder)
    for pred_file in pred_files:
        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, pred_file)
        
        if os.path.exists(gt_path):
            pred_img = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB)
            gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
            
            ious = {class_name: np.nan for class_name in CLASS_COLORS}
            for class_name, class_color in CLASS_COLORS.items():
                iou = calculate_iou(pred_img, gt_img, class_color)
                if not np.isnan(iou):
                    ious[class_name] = iou
            
            mean_iou = np.nanmean(list(ious.values()))
            global_accuracy = calculate_global_accuracy(pred_img, gt_img)
            
            result = {
                "filename": pred_file,
                "Liver": ious["Liver"],
                "Kidney": ious["Kidney"],
                "Diaphragm": ious["Diaphragm"],
                "Background": ious["Background"],
                "mean_iou": mean_iou,
                "global_accuracy": global_accuracy
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Calculate mean and std for each metric
    mean_metrics = results_df.mean(axis=0, numeric_only=True).to_dict()
    std_metrics = results_df.std(axis=0, numeric_only=True).to_dict()
    
    # Convert mean and std metrics to DataFrames
    mean_metrics_df = pd.DataFrame(mean_metrics, index=[0])
    std_metrics_df = pd.DataFrame(std_metrics, index=[0])
    
    # Add filename columns to mean and std DataFrames
    mean_metrics_df["filename"] = "mean"
    std_metrics_df["filename"] = "std"
    
    # Concatenate results_df with mean and std DataFrames
    results_df = pd.concat([results_df, mean_metrics_df, std_metrics_df], ignore_index=True)
    
    # Ensure the order of columns
    columns_order = ["filename", "Liver", "Kidney", "Diaphragm", "Background", "mean_iou", "global_accuracy"]
    results_df = results_df[columns_order]
    
    output_path = os.path.join(pred_folder, "segmentation_evaluation.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# Example usage
pred_folder = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\results\GLFR_b2_RGB"
gt_folder = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\validation\mask_rgb_256"
evaluate_segmentation(pred_folder, gt_folder)


# import os
# import numpy as np
# import cv2
# import pandas as pd

# # Define class colors
# CLASS_COLORS = {
#     "Liver": (255, 0, 0),
#     "Kidney": (0, 255, 0),
#     "Diaphragm": (0, 0, 255),
#     "Background": (0, 0, 0)
# }

# # Function to calculate IoU
# def calculate_iou(pred, gt, class_color):
#     pred_mask = np.all(pred == class_color, axis=-1)
#     gt_mask = np.all(gt == class_color, axis=-1)
#     intersection = np.logical_and(pred_mask, gt_mask).sum()
#     union = np.logical_or(pred_mask, gt_mask).sum()
#     if union == 0:
#         return np.nan
#     else:
#         return intersection / union

# # Function to calculate global accuracy
# def calculate_global_accuracy(pred, gt):
#     correct = np.all(pred == gt, axis=-1).sum()
#     total = pred.shape[0] * pred.shape[1]
#     return correct / total

# # Main function to evaluate segmentation
# def evaluate_segmentation(pred_folder, gt_folder):
#     results = []
#     pred_files = os.listdir(pred_folder)
#     for pred_file in pred_files:
#         pred_path = os.path.join(pred_folder, pred_file)
#         gt_path = os.path.join(gt_folder, pred_file)
        
#         if os.path.exists(gt_path):
#             pred_img = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB)
#             gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
            
#             ious = {}
#             for class_name, class_color in CLASS_COLORS.items():
#                 if np.any(np.all(gt_img == class_color, axis=-1)):  # Check if the class exists in GT
#                     iou = calculate_iou(pred_img, gt_img, class_color)
#                     ious[class_name] = iou
#                 else:
#                     ious[class_name] = np.nan
            
#             valid_ious = [iou for iou in ious.values() if not np.isnan(iou)]
#             mean_iou = np.mean(valid_ious) if valid_ious else np.nan
#             global_accuracy = calculate_global_accuracy(pred_img, gt_img)
            
#             result = {
#                 "filename": pred_file,
#                 "Liver": ious["Liver"],
#                 "Kidney": ious["Kidney"],
#                 "Diaphragm": ious["Diaphragm"],
#                 "Background": ious["Background"],
#                 "mean_iou": mean_iou,
#                 "global_accuracy": global_accuracy
#             }
#             results.append(result)
    
#     results_df = pd.DataFrame(results)
    
#     # Calculate mean and std for each metric
#     mean_metrics = results_df.mean(axis=0, numeric_only=True).to_dict()
#     std_metrics = results_df.std(axis=0, numeric_only=True).to_dict()
    
#     # Convert mean and std metrics to DataFrames
#     mean_metrics_df = pd.DataFrame(mean_metrics, index=[0])
#     std_metrics_df = pd.DataFrame(std_metrics, index=[0])
    
#     # Add filename columns to mean and std DataFrames
#     mean_metrics_df["filename"] = "mean"
#     std_metrics_df["filename"] = "std"
    
#     # Concatenate results_df with mean and std DataFrames
#     results_df = pd.concat([results_df, mean_metrics_df, std_metrics_df], ignore_index=True)
    
#     # Ensure the order of columns
#     columns_order = ["filename", "Liver", "Kidney", "Diaphragm", "Background", "mean_iou", "global_accuracy"]
#     results_df = results_df[columns_order]
    
#     output_path = os.path.join(pred_folder, "../segmentation_evaluation2.csv")
#     results_df.to_csv(output_path, index=False)
#     print(f"Results saved to {output_path}")

# # Example usage
# pred_folder = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\results\UNet_b2\img"
# gt_folder = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\validation\mask_rgb_256"
# evaluate_segmentation(pred_folder, gt_folder)
