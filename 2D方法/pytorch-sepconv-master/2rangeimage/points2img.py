import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rangeimg import LaserScan
import pre_process

path = 'C:/Users/13743/Desktop/2rangeimage/raw_lidar_data/'  # 待读取的文件夹
path_list = os.listdir(path)
# path_list.sort(key=lambda x: int(x[:-4]))  # 对读取的路径进行排序
# print(path_list)
path_add = []
output_img = []
for filename in path_list:
    path_add.append(os.path.join(path, filename))
    output_img.append(filename[0:-4] + ".png")

# 对于115200个点，水平分辨率大致是0.2，360/0.2=1800
trans2rangeimg = LaserScan(project=True, H=64, W=1800, fov_up=2.0, fov_down=-24.8)
for i in range(len(path_add)):
    trans2rangeimg.open_scan(path_add[i])
    xianshi = np.copy(trans2rangeimg.proj_range)
    nrom_grayimage = (trans2rangeimg.proj_range/np.max(trans2rangeimg.proj_range))*255
    image_o = np.where(nrom_grayimage > 0, nrom_grayimage, 0)
    new_image_o = np.zeros(nrom_grayimage.shape)
    new_image = pre_process.pre_process(image_o, new_image_o)
    img_convert = Image.fromarray(new_image)
    img_convert = img_convert.convert('RGB')  # 能转为灰度图，彩色图则改L为‘RGB’
    img_convert.save("C:/Users/13743/Desktop/2rangeimage/Range_image/"+output_img[i])
    print(0)

# for i in range(len(path_add)):
#     trans2rangeimg.open_scan(path_add[i])
#     xianshi = np.copy(trans2rangeimg.proj_xyz)
#     # xianshi = xianshi.transpose((2, 0, 1))
#     # nrom_grayimage = (trans2rangeimg.proj_range/np.max(trans2rangeimg.proj_range))*255
#     img_convert = Image.fromarray(np.uint8(xianshi))
#     img_convert = img_convert.convert('L')  # 能转为灰度图，彩色图则改L为‘RGB’
#
#     img_convert.save("D:/2rangeimage/Range_image/"+output_img[i])
#     print(0)






