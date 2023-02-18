import torch
import torch.nn as nn
import numpy as np

from models.models import PointINet
from models.utils import chamfer_loss, EMD
import mayavi.mlab as mlab

import argparse
from tqdm import tqdm
import os

import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_lidar(fn, npoints):
    points = np.fromfile(fn, dtype=np.float32, count=-1).reshape([-1, 4])
    raw_num = points.shape[0]
    if raw_num >= npoints:
        sample_idx = np.random.choice(raw_num, npoints, replace=False)
    else:
        sample_idx = np.concatenate((np.arange(raw_num), np.random.choice(raw_num, npoints - raw_num, replace=True)),
                                    axis=-1)

    pc = points[sample_idx, :]
    pc = torch.from_numpy(pc).t()
    color = np.zeros([npoints, 3]).astype('float32')
    color = torch.from_numpy(color).t()

    pc = pc.unsqueeze(0).cuda()
    color = color.unsqueeze(0).cuda()

    return pc, color

def create_pic():
    # fn_ini = '/home/cauli/Desktop/3D时间压缩图像与数据/原始点云文件/origianl11/0000000000.bin'
    # fn_end = '/home/cauli/Desktop/3D时间压缩图像与数据/原始点云文件/origianl11/0000000005.bin'
    fn_ini = '/home/cauli/Desktop/3D时间压缩图像与数据/原始点云文件/origianl11/0000000002.bin'
    fn_end = '/home/cauli/Desktop/3D时间压缩图像与数据/点云结果图/3元组/0.5.bin'
    # fn_mid_pc1 = './data/demo_data/original11/0000000001.bin'
    # fn_mid_pc2 = './data/demo_data/original11/0000000002.bin'
    # fn_mid_pc3 = './data/demo_data/original11/0000000003.bin'
    # fn_mid_pc4 = './data/demo_data/original11/0000000004.bin'
    # fn_mid_pc5 = './data/demo_data/original11/0000000005.bin'
    # fn_mid_pc6 = './data/demo_data/original11/0000000006.bin'
    # fn_mid_pc7 = './data/demo_data/original11/0000000007.bin'
    # fn_mid_pc8 = './data/demo_data/original11/0000000008.bin'
    # fn_mid_pc9 = './data/demo_data/original11/0000000009.bin'

    npoints=32768
    pc1, color1 = get_lidar(fn_ini, npoints)
    pc2, color2 = get_lidar(fn_end, npoints)  # color这里是全0，pc表示的是点的坐标和属性[1，4,点的个数]
    # mid_pc1, mid_color1 = get_lidar(fn_mid_pc1, npoints)
    # mid_pc2, mid_color2 = get_lidar(fn_mid_pc2, npoints)
    # mid_pc3, mid_color3 = get_lidar(fn_mid_pc3, npoints)
    # mid_pc4, mid_color4 = get_lidar(fn_mid_pc4, npoints)
    # mid_pc5, mid_color5 = get_lidar(fn_mid_pc5, npoints)
    # mid_pc6, mid_color6 = get_lidar(fn_mid_pc6, npoints)
    # mid_pc7, mid_color7 = get_lidar(fn_mid_pc7, npoints)
    # mid_pc8, mid_color4 = get_lidar(fn_mid_pc8, npoints)
    # mid_pc9, mid_color9 = get_lidar(fn_mid_pc8, npoints)

    ini_pc = pc1.squeeze(0).permute(1, 0).cpu().numpy()
    end_pc = pc2.squeeze(0).permute(1, 0).cpu().numpy()  # permute表示调换维度
    fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), engine=None, size=(1600, 1000))
    mlab.points3d(ini_pc[:, 0], ini_pc[:, 1], ini_pc[:, 2], color=(0, 1, 0), scale_factor=0.2, figure=fig,
                  mode='sphere')
    #mlab.show()
    mlab.points3d(end_pc[:, 0], end_pc[:, 1], end_pc[:, 2], color=(0, 0, 1), scale_factor=0.2, figure=fig,
                  mode='sphere')
    mlab.show()


if __name__ == '__main__':
    create_pic()