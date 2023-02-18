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


def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--npoints', type=int, default=32768)
    parser.add_argument('--pretrain_model', type=str, default='./pretrain_model/interp_kitti.pth')
    parser.add_argument('--pretrain_flow_model', type=str,
                        default='./pretrain_model/flownet3d_kitti_odometry_maxbias1.pth')
    parser.add_argument('--is_save', type=int, default=1)
    parser.add_argument('--visualize', type=int, default=1)

    return parser.parse_args()


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


def demo(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fn_ini = './data/demo_data/original11/0000000000.bin'
    fn_end = './data/demo_data/original11/0000000002.bin'
    fn_mid_pc1 = './data/demo_data/original11/0000000001.bin'
    fn_mid_pc2 = './data/demo_data/original11/0000000002.bin'
    fn_mid_pc3 = './data/demo_data/original11/0000000003.bin'
    fn_mid_pc4 = './data/demo_data/original11/0000000004.bin'
    fn_mid_pc5 = './data/demo_data/original11/0000000005.bin'
    fn_mid_pc6 = './data/demo_data/original11/0000000006.bin'
    fn_mid_pc7 = './data/demo_data/original11/0000000007.bin'
    fn_mid_pc8 = './data/demo_data/original11/0000000008.bin'
    fn_mid_pc9 = './data/demo_data/original11/0000000009.bin'

    net = PointINet()
    net.load_state_dict(torch.load(args.pretrain_model))
    net.flow.load_state_dict(torch.load(args.pretrain_flow_model))
    net.eval()
    net.cuda()

    interp_scale = 10  # 可选10
    t_array = np.arange(1.0 / interp_scale, 1.0, 1.0 / interp_scale, dtype=np.float32)  # 起点，终点，步长

    with torch.no_grad():
        pc1, color1 = get_lidar(fn_ini, args.npoints)
        pc2, color2 = get_lidar(fn_end, args.npoints)  # color这里是全0，pc表示的是点的坐标和属性[1，4,点的个数]
        mid_pc1, mid_color1 = get_lidar(fn_mid_pc1, args.npoints)
        mid_pc2, mid_color2 = get_lidar(fn_mid_pc2, args.npoints)
        mid_pc3, mid_color3 = get_lidar(fn_mid_pc3, args.npoints)
        mid_pc4, mid_color4 = get_lidar(fn_mid_pc4, args.npoints)
        mid_pc5, mid_color5 = get_lidar(fn_mid_pc5, args.npoints)
        mid_pc6, mid_color6 = get_lidar(fn_mid_pc6, args.npoints)
        mid_pc7, mid_color7 = get_lidar(fn_mid_pc7, args.npoints)
        mid_pc8, mid_color4 = get_lidar(fn_mid_pc8, args.npoints)
        mid_pc9, mid_color9 = get_lidar(fn_mid_pc8, args.npoints)

        mid_pc_list = []
        mid_pc_list.append(mid_pc1)
        mid_pc_list.append(mid_pc2)
        mid_pc_list.append(mid_pc3)
        mid_pc_list.append(mid_pc4)
        mid_pc_list.append(mid_pc5)
        mid_pc_list.append(mid_pc6)
        mid_pc_list.append(mid_pc7)
        mid_pc_list.append(mid_pc8)
        mid_pc_list.append(mid_pc9)

        # pc1=pc1.cpu().numpy()
        # color1=color1.cpu().numpy()

        # 中间插多个点云
        chamfer_loss_list = []
        emd_loss_list = []
        run_time_total = 0

        for i in range(interp_scale - 1):
            t = t_array[i]
            t = torch.tensor([t])
            t = t.cuda().float()

            start_time = time.time()  # 程序开始时间
            pred_mid_pc = net(pc1, pc2, color1, color2, t)
            end_time = time.time()  # 程序结束时间
            run_time = end_time - start_time  # 程序的运行时间，单位为秒
            run_time_total = run_time+run_time_total

            # 计算EMD和CD
            cd = chamfer_loss(pred_mid_pc[:, :3, :], mid_pc_list[i][:, :3, :])
            emd = EMD(pred_mid_pc[:, :3, :], mid_pc_list[i][:, :3, :])

            cd = cd.squeeze().cpu().numpy()
            emd = emd.squeeze().cpu().numpy()

            chamfer_loss_list.append(cd)
            emd_loss_list.append(emd)

            ini_pc = pc1.squeeze(0).permute(1,
                                            0).cpu().numpy()  # squeeze降维，squeeze(0)表示将第0维去掉，一个tensor的维度从前到后为[0，1，2，3]
            end_pc = pc2.squeeze(0).permute(1, 0).cpu().numpy()  # permute表示调换维度

            pred_mid_pc = pred_mid_pc.squeeze(0).permute(1, 0).cpu().numpy()

            # if args.visualize == 1:
            #     fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), engine=None, size=(1600, 1000))
            #     mlab.points3d(ini_pc[:, 0], ini_pc[:, 1], ini_pc[:, 2], color=(0, 0, 1), scale_factor=0.2, figure=fig,
            #                   mode='sphere')
            #     mlab.points3d(end_pc[:, 0], end_pc[:, 1], end_pc[:, 2], color=(0, 1, 0), scale_factor=0.2, figure=fig,
            #                   mode='sphere')
            #     # 前后两帧的颜色不要变
            #     mlab.points3d(pred_mid_pc[:, 0], pred_mid_pc[:, 1], pred_mid_pc[:, 2], color=(1, 0, 0),
            #                   scale_factor=0.2, figure=fig, mode='sphere')
            #     mlab.show()

            if args.is_save == 1:
                save_dir = './data/demo_data/interpolated11'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_name = os.path.join(save_dir, str(t.squeeze().cpu().numpy()) + '.bin')
                pred_mid_pc.tofile(save_name)
                print("save interpolated point clouds to:", save_name)

        print("run_time: ", run_time_total)
        print("chamfer distance: ", chamfer_loss_list)
        print("earth mover's distance: ", emd_loss_list)

        # 三元组
        # t = torch.tensor([0.5])  # 在这里设置时间
        # t = t.cuda().float()
        #
        # start_time = time.time()  # 程序开始时间
        # pred_mid_pc = net(pc1, pc2, color1, color2, t)
        # end_time = time.time()  # 程序结束时间
        # run_time = end_time - start_time  # 程序的运行时间，单位为秒
        # print("run_time: ", run_time)
        #
        # # 计算EMD和CD
        # cd = chamfer_loss(pred_mid_pc[:, :3, :], mid_pc1[:, :3, :])
        # emd = EMD(pred_mid_pc[:, :3, :], mid_pc1[:, :3, :])
        #
        # cd = cd.squeeze().cpu().numpy()
        # emd = emd.squeeze().cpu().numpy()
        #
        # print("chamfer distance: ", cd)
        # print("earth mover's distance: ", emd)
        #
        # ini_pc = pc1.squeeze(0).permute(1, 0).cpu().numpy()  # squeeze降维，squeeze(0)表示将第0维去掉，一个tensor的维度从前到后为[0，1，2，3]
        # end_pc = pc2.squeeze(0).permute(1, 0).cpu().numpy()  # permute表示调换维度
        #
        # pred_mid_pc = pred_mid_pc.squeeze(0).permute(1, 0).cpu().numpy()
        # mid_pc1 = mid_pc1.squeeze(0).permute(1, 0).cpu().numpy()
        #
        # # if args.visualize == 1:
        # #     fig = mlab.figure(figure=None, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), engine=None, size=(1600, 1000))
        # #     # mlab.points3d(ini_pc[:, 0], ini_pc[:, 1], ini_pc[:, 2], color=(0, 0, 1), scale_factor=0.2, figure=fig,
        # #     #               mode='sphere')
        # #     # mlab.show()
        # #     mlab.points3d(end_pc[:, 0], end_pc[:, 1], end_pc[:, 2], color=(0, 1, 0), scale_factor=0.2, figure=fig,
        # #                   mode='sphere')
        # #     #mlab.show()
        # #     mlab.points3d(pred_mid_pc[:, 0], pred_mid_pc[:, 1], pred_mid_pc[:, 2], color=(1, 0, 0),
        # #                   scale_factor=0.2, figure=fig, mode='sphere')
        # #     mlab.show()
        # #     mlab.points3d(mid_pc1[:, 0], mid_pc1[:, 1], mid_pc1[:, 2], color=(1, 0, 0),
        # #                   scale_factor=0.2, figure=fig, mode='sphere')
        # #     mlab.show()
        # if args.is_save == 1:
        #     save_dir = './data/demo_data/interpolated_3'
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     save_name = os.path.join(save_dir, str(t.squeeze().cpu().numpy()) + '.bin')
        #     pred_mid_pc.tofile(save_name)
        #     print("save interpolated point clouds to:", save_name)


if __name__ == '__main__':
    args = parse_args()
    demo(args)
