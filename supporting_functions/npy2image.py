import numpy as np
import matplotlib.pyplot as plt

# 載入 NumPy 檔案
data = np.load(r"C:\Users\JJJ\my\data\.images\FAST\rawdata\RUQ\segmentation\result\sigmoid_prediction\R08302.npy")

# 隨機選取幾個位置
num_positions = 3
positions = np.random.randint(0, data.shape[1], size=(num_positions, 2))  # 在影像範圍內隨機選取 (y, x) 座標

# 打印每個位置在 4 個 channel 中的值
for pos in positions:
    y, x = pos
    print("位置 ({}, {}):".format(y, x))
    for channel in range(data.shape[0]):
        value = data[channel, y, x]
        print("  Channel {}: {}".format(channel, value))
        
# 找到最大值和最小值
max_value = np.max(data)
min_value = np.min(data)

print("最大值:", max_value)
print("最小值:", min_value)


# 選擇其中一張影像
for i in range(data.shape[0]):
    image_to_show = data[i]
    # 顯示影像
    plt.imshow(image_to_show, cmap='gray')
    plt.colorbar()  # 顯示色條
    plt.show()