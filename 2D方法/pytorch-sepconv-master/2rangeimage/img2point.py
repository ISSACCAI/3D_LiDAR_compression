import os
import numpy as np
from scipy import optimize
import math
import matplotlib.pyplot as plt
from PIL import Image
from rangeimg import LaserScan

H = 64
W = 1800
fov_up = 2.0
fov_down = -24.8
fov_up = fov_up / 180.0 * np.pi  # field of view up in rad
fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

yaw=0
z=0
depth=0

def f(x):
    x0, x1 = x.tolist()
    return [-math.atan2(x1, x0) - yaw, x0 ** 2 + x1 ** 2 + z ** 2 - depth ** 2]


path = "Range_image/"  # 待读取的文件夹
path_list = os.listdir(path)
path_list.sort(key=lambda x: int(x[:-4]))  # 对读取的路径进行排序
# print(path_list)
path_add = []
output_txt = []
for filename in path_list:
    path_add.append(os.path.join(path, filename))
    output_txt.append(filename[0:-4] + ".txt")

total_result = []
for i in range(len(path_add)):
    img = Image.open(path_add[i])
    img_array = np.array(img)
    for row in range(img_array.shape[0]):
        for col in range(img_array.shape[1]):
            if img_array[row][col] != 0:
                depth = img_array[row][col]
                row_new = row / W
                col_new = col / H
                yaw = (row_new * 2 - 1) * math.pi
                pitch = (1 - col_new) * fov - abs(fov_down)
                z = math.sin(pitch) * depth
                result = optimize.fsolve(f, [1, 1])
                total_result.append(result)
    print(0)

print(0)
