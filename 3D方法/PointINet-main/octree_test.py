import torch
import torch.nn as nn
import numpy as np
from models.utils import chamfer_loss, EMD
import os
from plyfile import PlyData
import mayavi.mlab as mlab

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_lidar(path, npoints):
    # points = np.fromfile(fn, dtype=np.float32, count=-1).reshape([-1, 4])

    """
    load object vertices
    :param pth: str
    :return: pts: (N, 3)
    """
    # ply就是得到的对象
    ply = PlyData.read(path)
    # vtx相当于得到的所有的点的信息，每个点有10个属性
    vtx = ply['vertex']
    # 只要有用的3个属性，stack得到的列表是一堆x，一堆y，一堆z,
    # 我们要对其进行转换，axis=-1是在列的维度拼接，
    # 变成一个x，一个y，一个z这种的格式，
    points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)

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


def octree_test():
    fn_ini = '/home/cauli/Desktop/八叉树编码前后点云/不同分辨率下的点云数据/1e-5/0000000001.plyenc.ply'
    fn_end = '/home/cauli/Desktop/八叉树编码前后点云/不同分辨率下的点云数据/1e-5/0000000001.plydec.ply'

    pc1, color1 = get_lidar(fn_ini, 32768)
    pc2, color2 = get_lidar(fn_end, 32768)  # color这里是全0，pc表示的是点的坐标和属性[1，4,点的个数]
    pc1 = torch.tensor(pc1, dtype=torch.float)
    pc1 = pc1/100000  # 不理解

    ini_pc = pc1.squeeze(0).permute(1, 0).cpu().numpy()
    end_pc = pc2.squeeze(0).permute(1, 0).cpu().numpy()  # permute表示调换维度
    fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), engine=None, size=(1600, 1000))
    mlab.points3d(ini_pc[:, 0], ini_pc[:, 1], ini_pc[:, 2], color=(0, 1, 0), scale_factor=0.2, figure=fig,
                  mode='sphere')
    mlab.points3d(end_pc[:, 0], end_pc[:, 1], end_pc[:, 2], color=(0, 0, 1), scale_factor=0.2, figure=fig,
                  mode='sphere')
    mlab.show()
    # 计算EMD和CD
    cd = chamfer_loss(pc1, pc2)
    emd = EMD(pc1, pc2)
    cd = cd.squeeze().cpu().numpy()
    emd = emd.squeeze().cpu().numpy()
    print("chamfer distance: ", cd)
    print("earth mover's distance: ", emd)


if __name__ == '__main__':
    octree_test()
