from TorchDB import DBreader_frame_interpolation
from torch.utils.data import DataLoader
from model import SepConvNet
import argparse
from torchvision import transforms
import torch
from torch.autograd import Variable
import os
import random
from TestModule import Middlebury_other

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--train', type=str, default='./db_train')
parser.add_argument('--kernel_vertical', type=int, default=25)
parser.add_argument('--kernel_horizontal', type=int, default=128)
parser.add_argument('--out_dir', type=str, default='./output_sepconv_pytorch')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--load_model', type=str, default='./pretrain_model/model_epoch021.pth')  # './pretrain_model/sepconv-lf'
parser.add_argument('--test_input', type=str, default='./db_val_small/input')
parser.add_argument('--gt', type=str, default='./db_val_small/gt')

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main():
    args = parser.parse_args()
    db_dir = args.train

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    logfile = open(args.out_dir + '/log.txt', 'w')
    logfile.write('batch_size: ' + str(args.batch_size) + '\n')

    total_epoch = args.epochs
    batch_size = args.batch_size

    # 按照输入的KITTI数据集的大小调整了裁剪的尺寸
    # dataset = DBreader_frame_interpolation(db_dir, resize=(64, 2048), randomresizedcrop=(64, 2048))
    dataset = DBreader_frame_interpolation(db_dir)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    TestDB = Middlebury_other(args.test_input, args.gt)
    test_output_dir = args.out_dir + '/result'

    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        kernel_size_Vertical = args.kernel_vertical
        kernel_size_Horizontal = args.kernel_horizontal
        model = SepConvNet(kernel_size_Vertical=kernel_size_Vertical, kernel_size_Horizontal=kernel_size_Horizontal)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        # model.epoch = checkpoint['epoch']


        # kernel_size_Vertical = args.kernel_vertical
        # kernel_size_Horizontal = args.kernel_horizontal
        # model = SepConvNet(kernel_size_Vertical=kernel_size_Vertical, kernel_size_Horizontal=kernel_size_Horizontal)
        # model_dict = model.state_dict()
        # model_pre = torch.load(args.load_model)
        # keys = []
        # for k, v in model_pre.items():
        #     keys.append(k)
        # i = 0
        #
        # # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
        # for k, v in model_dict.items():
        #     if v.size() == model_pre[keys[i]].size():
        #         model_dict[k] = model_pre[keys[i]]
        #         # print(model_dict[k])
        #         i = i + 1
        # model.load_state_dict(model_dict)
        print("pre_train")

    else:
        kernel_size_Vertical = args.kernel_vertical
        kernel_size_Horizontal = args.kernel_horizontal
        model = SepConvNet(kernel_size_Vertical=kernel_size_Vertical, kernel_size_Horizontal=kernel_size_Horizontal)
        print("new_train")

    logfile.write('kernel_size_Vertical: ' + str(kernel_size_Vertical) + '  '+'kernel_size_Horizontal: ' + str(kernel_size_Horizontal) + '\n')

    if torch.cuda.is_available():
        model = model.cuda()

    max_step = train_loader.__len__()  #所有训练集中三元组数目/batch_size

    model.eval()
    TestDB.Test(model, test_output_dir, logfile, str(model.epoch.item()).zfill(3) + '.png')

    while True:
        if model.epoch.item() == total_epoch:
            break
        model.train()
        for batch_idx, (frame0, frame1, frame2) in enumerate(train_loader):  # 这里batch_idx相当于max_step的序号
            # print(batch_idx)
            frame0 = to_variable(frame0)
            frame1 = to_variable(frame1)
            frame2 = to_variable(frame2)
            frame0 = frame0.type(torch.cuda.FloatTensor)
            frame1 = frame1.type(torch.cuda.FloatTensor)
            frame2 = frame2.type(torch.cuda.FloatTensor)
            if random.randint(0, 1):
                loss = model.train_model(frame0, frame2, frame1)
            else:
                loss = model.train_model(frame2, frame0, frame1)
            if batch_idx % 100 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ',
                                                                            '[' + str(model.epoch.item()) + '/' + str(
                                                                                total_epoch) + ']', 'Step: ',
                                                                            '[' + str(batch_idx) + '/' + str(
                                                                                max_step) + ']', 'train loss: ',
                                                                            loss.item()))
        model.increase_epoch()
        if model.epoch.item() % 1 == 0:
            torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size_Vertical': kernel_size_Vertical, 'kernel_size_Horizontal': kernel_size_Horizontal},
                       ckpt_dir + '/model_epoch' + str(model.epoch.item()).zfill(3) + '.pth')
            model.eval()
            TestDB.Test(model, test_output_dir, logfile,
                        str(model.epoch.item()).zfill(3) + '.png')  # zfill表示指定字符串的长度，前面补零
            logfile.write('\n')

    logfile.close()


if __name__ == "__main__":
    main()
