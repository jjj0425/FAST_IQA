import numpy as np

# Load the .npy file
file_path = r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\result\sigmoid_prediction\R08302.npy"
data = np.load(file_path)

# Print the shape of the array
print("Shape of data:", data.shape)

# Print the total number of elements in the array
print("Total number of elements in data:", data.size)