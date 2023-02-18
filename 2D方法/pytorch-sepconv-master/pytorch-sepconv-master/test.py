import argparse
from TestModule import Middlebury_other
from model import SepConvNet
import torch
import time

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--kernel_vertical', type=int, default=25)
parser.add_argument('--kernel_horizontal', type=int, default=128)
parser.add_argument('--input', type=str, default='./db_test/input')
parser.add_argument('--gt', type=str, default='./db_test/gt')
parser.add_argument('--output', type=str, default='./test_output/result')
parser.add_argument('--checkpoint', type=str, default='./pretrain_model/model_epoch008.pth')  #'./pretrain_model/model_epoch037.pth'


def main():
    args = parser.parse_args()
    input_dir = args.input
    gt_dir = args.gt
    output_dir = args.output
    ckpt = args.checkpoint

    print("Reading Test DB...")
    TestDB = Middlebury_other(input_dir, gt_dir)
    print("Loading the Model...")
    # checkpoint = torch.load(ckpt)
    # kernel_size = checkpoint['kernel_size']
    # model = SepConvNet(kernel_size=kernel_size)
    # state_dict = checkpoint['state_dict']
    # model.load_state_dict(torch.load(state_dict))
    # model.epoch = checkpoint['epoch']
    checkpoint = torch.load(ckpt)
    kernel_size_Vertical = args.kernel_vertical
    kernel_size_Horizontal = args.kernel_horizontal
    model = SepConvNet(kernel_size_Vertical=kernel_size_Vertical, kernel_size_Horizontal=kernel_size_Horizontal)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()

    print("Test Start...")
    TestDB.Test(model, output_dir)


if __name__ == "__main__":
    start_time = time.time()    # 程序开始时间
    main()
    end_time = time.time()    # 程序结束时间
    run_time = (end_time - start_time)/1344    # 程序的运行时间，单位为秒,1344表示测试的数据数据量
    print("run_time:" + str(run_time))
